//! HeXO deep prover — a standalone research CLI running several proof-search
//! drivers over the validated VCF kernel (`crate::forcing`).
//!
//! Drivers: `idtt` (the production iterative-deepening solver, baseline), `dfpn`
//! (Nagai depth-first proof-number search), `pdspn` (Winands et al. two-level
//! PDS-PN), and `hybrid` (line-guided verification). A `race` portfolio runs
//! several drivers on OS threads and takes the first definitive verdict.
//!
//! All drivers share one rule set through [`kernel::KernelCtx`]; see that module.
//! This module owns the shared config/verdict types, the `idtt` wrapper, and the
//! top-level dispatch. No PyO3 surface, no production-path changes.

pub mod certificate;
pub mod dfpn;
#[cfg(test)]
mod fixtures;
pub mod hybrid;
pub mod io;
pub(crate) mod kernel;
pub mod pdspn;
pub mod pn;
pub mod portfolio;

use hexo_engine::game::{GameConfig, GameState};
use hexo_engine::types::Coord;
use certificate::ProofCertificate;
use io::{Line, Position, Report, Stats, UnverifiedBranch, Verdict};
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Which proof-search driver to run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DriverKind {
    Idtt,
    Dfpn,
    Pdspn,
    PdspnDepth,
    PdspnShortest,
    Hybrid,
    Race,
}

impl DriverKind {
    pub fn name(self) -> &'static str {
        match self {
            DriverKind::Idtt => "idtt",
            DriverKind::Dfpn => "dfpn",
            DriverKind::Pdspn => "pdspn",
            DriverKind::PdspnDepth => "pdspn-depth",
            DriverKind::PdspnShortest => "pdspn-shortest",
            DriverKind::Hybrid => "hybrid",
            DriverKind::Race => "race",
        }
    }
    pub fn parse(s: &str) -> Option<DriverKind> {
        Some(match s {
            "idtt" => DriverKind::Idtt,
            "dfpn" => DriverKind::Dfpn,
            "pdspn" => DriverKind::Pdspn,
            "pdspn-depth" => DriverKind::PdspnDepth,
            "pdspn-shortest" => DriverKind::PdspnShortest,
            "hybrid" => DriverKind::Hybrid,
            "race" => DriverKind::Race,
            _ => return None,
        })
    }
}

/// Search-strategy configuration shared by every driver.
#[derive(Clone)]
pub struct ProverConfig {
    pub driver: DriverKind,
    /// Attacker-turn generator width. `true` enables the experimental wide-partner
    /// knob (a strict superset of the tight generator; off by default).
    pub wide: bool,
    pub depth_cap: u8,
    pub node_budget: u64,
    pub tt_mb: usize,
    pub pn2_nodes: u64,
    /// Adaptive level-2 budget: when > 0, each leaf's PN² node cap is scaled
    /// by its branching factor. `pn2_scale` is the reference branching factor
    /// that maps to the full `pn2_nodes`; 0 disables scaling (fixed budget).
    /// By default the budget scales UP with branching (`pn2_nodes * branching /
    /// pn2_scale`); set `pn2_scale_inverse` to scale it DOWN instead
    /// (`pn2_nodes * pn2_scale / branching`), matching the observed
    /// puzzle-dependence where low-branching positions want larger budgets.
    pub pn2_scale: u64,
    pub pn2_scale_inverse: bool,
    pub leaf_budget: u64,
    pub leaf_budget_max: u64,
    /// Optional total level-1 node budget for PDS-PN root-attack screening in
    /// the proof-guided shortest-win path. Zero disables the screen.
    pub root_screen_budget: u64,
    /// Maximum level-1 nodes spent trying to classify any one root attack.
    pub root_screen_per_attack: u64,
    /// Wall-clock limit in seconds; 0 disables (no deadline).
    pub time_limit_s: f64,
    pub race_set: Vec<DriverKind>,
}

impl Default for ProverConfig {
    fn default() -> Self {
        ProverConfig {
            driver: DriverKind::Dfpn,
            wide: false,
            depth_cap: 40,
            node_budget: 20_000_000,
            tt_mb: 512,
            pn2_nodes: 50_000,
            pn2_scale: 0,
            pn2_scale_inverse: false,
            leaf_budget: 1_000_000,
            leaf_budget_max: 10_000_000,
            root_screen_budget: 0,
            root_screen_per_attack: 5_000,
            time_limit_s: 1800.0,
            race_set: vec![DriverKind::Idtt, DriverKind::Dfpn, DriverKind::Pdspn],
        }
    }
}

impl ProverConfig {
    /// The width string for reports.
    pub fn width_str(&self) -> &'static str {
        if self.wide { "wide" } else { "tight" }
    }
}

/// Result of conditioning the forcing model on an attacker turn that has
/// already been placed. `Covers` contains every exact two-stone minimum cover;
/// proving the continuation after each cover is sufficient to prove the attack.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DefenseReplies {
    Covers(Vec<[Coord; 2]>),
    /// Covering all threats needs at least three placements.
    AttackerWin,
    /// The fixed move was not forcing, or the defender can complete first.
    NotForcing,
}

/// Enumerate all legal minimum replies to an already-played attacker turn using
/// the same kernel and width as PDS-PN. The position stones must include the
/// fixed attack, and `position.attacker` identifies the attacking player.
pub fn minimum_defenses_after_attack(
    position: &Position,
    wide: bool,
) -> Result<DefenseReplies, String> {
    let mut kernel = kernel::KernelCtx::new_wide(
        &position.stones,
        position.attacker,
        position.config.win_length,
        position.config.placement_radius,
        wide,
    )
    .ok_or("could not build the defense-reply kernel")?;
    Ok(match kernel.and_eval() {
        kernel::AndEval::Covers(covers) => DefenseReplies::Covers(
            covers
                .into_iter()
                .map(|cover| <[Coord; 2]>::try_from(cover.cells())
                    .map_err(|_| "forcing minimum cover did not contain two cells"))
                .collect::<Result<Vec<_>, _>>()?,
        ),
        kernel::AndEval::AttackerWin => DefenseReplies::AttackerWin,
        kernel::AndEval::Loss => DefenseReplies::NotForcing,
    })
}

/// Cooperative search control: a wall-clock deadline plus a cancel flag a racing
/// thread can raise. Every driver checks [`Ctl::expired`] at its budget cadence.
#[derive(Clone)]
pub struct Ctl {
    pub deadline: Option<Instant>,
    pub cancel: Arc<AtomicBool>,
}

impl Ctl {
    pub fn new(time_limit_s: f64) -> Ctl {
        let deadline = if time_limit_s > 0.0 {
            Some(Instant::now() + std::time::Duration::from_secs_f64(time_limit_s))
        } else {
            None
        };
        Ctl { deadline, cancel: Arc::new(AtomicBool::new(false)) }
    }

    /// True once the deadline has passed or another thread cancelled the search.
    #[inline]
    pub fn expired(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
            || self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    /// The `forcing::Limits` view of this control, for the `idtt` driver.
    pub(crate) fn forcing_limits(&self) -> crate::forcing::Limits {
        crate::forcing::Limits {
            deadline: self.deadline,
            cancel: Some(Arc::clone(&self.cancel)),
        }
    }
}

/// A single driver's outcome, assembled into a [`Report`] by [`run`].
#[derive(Clone, Debug)]
pub struct DriverResult {
    pub verdict: Verdict,
    pub depth: Option<u8>,
    pub pv: Vec<Coord>,
    pub stats: Stats,
    pub unverified: Vec<UnverifiedBranch>,
    pub certificate: Option<ProofCertificate>,
}

impl DriverResult {
    pub fn new(verdict: Verdict) -> DriverResult {
        DriverResult {
            verdict,
            depth: None,
            pv: Vec::new(),
            stats: Stats::default(),
            unverified: Vec::new(),
            certificate: None,
        }
    }
}

impl Position {
    /// Build the engine `GameState` for this position (used by the `idtt` driver,
    /// which calls the production solver directly).
    pub fn to_game(&self) -> GameState {
        let cfg = GameConfig {
            win_length: self.config.win_length,
            placement_radius: self.config.placement_radius,
            max_moves: self.config.max_moves,
        };
        GameState::from_state(&self.stones, self.attacker, self.placements_remaining, cfg)
    }
}

/// The `idtt` baseline: the production iterative-deepening + TT solver, with an
/// added wall-clock deadline and race-cancel hook (both no-ops when unset). It
/// proves the *shortest* win (its depth is authoritative); `nodes` is not exposed
/// by the production solver, so it stays 0 in the stats.
pub fn idtt(pos: &Position, cfg: &ProverConfig, ctl: &Ctl) -> DriverResult {
    use crate::forcing::{Outcome, solve_limited};
    let game = pos.to_game();
    let t = Instant::now();
    let outcome = solve_limited(&game, cfg.depth_cap, cfg.node_budget, cfg.wide, ctl.forcing_limits());
    let elapsed = t.elapsed().as_secs_f64();
    let mut r = match outcome {
        Outcome::Win(w) => {
            let mut r = DriverResult::new(Verdict::Win);
            r.depth = Some(w.depth);
            r.pv = w.pv;
            r
        }
        Outcome::No => DriverResult::new(Verdict::No),
        Outcome::BudgetExceeded => DriverResult::new(Verdict::BudgetExceeded),
    };
    r.stats.elapsed_s = elapsed;
    r
}

/// Shortest-win IDTT using an independently verified PDS-PN proof DAG as a
/// positive oracle. Certificate omissions never prune IDTT moves.
pub fn guided_idtt(
    pos: &Position,
    certificate: &ProofCertificate,
    cfg: &ProverConfig,
    ctl: &Ctl,
) -> Result<DriverResult, String> {
    use crate::forcing::{
        ForcingVerdict, Outcome, solve_limited_guided, solve_limited_guided_verdict,
    };
    let certificate_wide = match certificate.width.as_str() {
        "tight" => false,
        "wide" => true,
        other => return Err(format!("invalid proof width {other:?}")),
    };
    if certificate_wide != cfg.wide {
        return Err(format!(
            "proof width {:?} does not match requested {} search",
            certificate.width,
            cfg.width_str(),
        ));
    }
    let (_summary, mut hints, mut guide_nodes) = certificate::verify_with_hints(pos, certificate)?;
    let t = Instant::now();
    let mut total_nodes = 0u64;
    let mut total_hint_hits = 0u64;
    let mut probes = 0u64;
    let mut tightened = 0u64;
    let mut screened_attacks = 0u64;
    let mut screen_refuted_attacks = 0u64;
    let mut screen_winning_attacks = 0u64;
    let mut screen_unresolved_attacks = 0u64;

    // Cheap global refutation screen at the root. The certificate's already
    // verified winning root actions are skipped; every other forcing attack is
    // handed to PDS-PN at its post-attack AND node. A definitive PDS-PN NO is
    // stronger than any finite depth result, so it is sound to close that AND
    // node for every IDTT threshold. WIN and budget-exhausted attacks remain in
    // the ordinary exhaustive IDTT move set.
    if cfg.root_screen_budget > 0 && !ctl.expired() {
        let known_root_moves = guide_nodes
            .iter()
            .find(|node| node.stones.len() == pos.stones.len())
            .and_then(|node| hints.attacker_order(node.hash, node.placements))
            .map(<[crate::forcing::CellSet2]>::to_vec)
            .unwrap_or_default();
        let mut screen_cfg = cfg.clone();
        screen_cfg.node_budget = cfg.root_screen_budget.min(cfg.node_budget);
        let screen = dfpn::screen_root_attacks(
            pos,
            &screen_cfg,
            ctl,
            &known_root_moves,
            cfg.root_screen_per_attack,
        );
        for hash in screen.refuted_and_hashes {
            hints.insert_no_win_within(hash, false, 0, u8::MAX);
        }
        total_nodes = total_nodes.saturating_add(screen.stats.nodes);
        screened_attacks = screen.attacks_total;
        screen_refuted_attacks = screen.refuted;
        screen_winning_attacks = screen.winning;
        screen_unresolved_attacks = screen.unresolved;
    }
    let mut hints = Rc::new(hints);

    // Tighten exact certificate states from the leaves upward before asking the
    // hard root question. A local IDTT win is itself a sound positive fact; a
    // failed/starved probe contributes nothing. Reserve most of the user's node
    // budget and wall time for the final globally-shortest root search.
    guide_nodes.sort_by_key(|node| (!node.primary, node.certified_depth));
    let prepass_budget = cfg.node_budget / 4;
    let prepass_deadline = ctl.deadline.map(|deadline| {
        deadline.min(Instant::now() + std::time::Duration::from_secs(60))
    });
    for node in guide_nodes {
        if total_nodes >= prepass_budget
            || ctl.expired()
            || prepass_deadline.is_some_and(|deadline| Instant::now() >= deadline)
        {
            break;
        }
        if node.certified_depth <= 1
            || node.certified_depth > u8::MAX as u32
            || node.stones.len() == pos.stones.len()
        {
            continue;
        }
        let per_probe_budget = cfg
            .leaf_budget
            .min(prepass_budget.saturating_sub(total_nodes));
        if per_probe_budget == 0 {
            break;
        }
        let game = GameState::from_state(
            &node.stones,
            pos.attacker,
            node.placements,
            GameConfig {
                win_length: pos.config.win_length,
                placement_radius: pos.config.placement_radius,
                max_moves: pos.config.max_moves,
            },
        );
        let limits = crate::forcing::Limits {
            deadline: prepass_deadline,
            cancel: Some(Arc::clone(&ctl.cancel)),
        };
        let (outcome, stats) = solve_limited_guided_verdict(
            &game,
            node.certified_depth.saturating_sub(1) as u8,
            per_probe_budget,
            cfg.wide,
            limits,
            Rc::clone(&hints),
        );
        probes += 1;
        total_nodes = total_nodes.saturating_add(stats.nodes);
        total_hint_hits = total_hint_hits.saturating_add(stats.hint_hits);
        if stats.excluded_through > 0 {
            Rc::make_mut(&mut hints).insert_no_win_within(
                node.hash,
                true,
                node.placements,
                stats.excluded_through,
            );
        }
        if let ForcingVerdict::Win { depth } = outcome {
            Rc::make_mut(&mut hints).insert(node.hash, true, node.placements, depth.into());
            tightened += 1;
        }
    }

    let game = pos.to_game();
    let root_budget = cfg.node_budget.saturating_sub(total_nodes);
    let (outcome, search_stats, best_win) = solve_limited_guided(
        &game,
        cfg.depth_cap,
        root_budget,
        cfg.wide,
        ctl.forcing_limits(),
        hints,
    );
    total_nodes = total_nodes.saturating_add(search_stats.nodes);
    total_hint_hits = total_hint_hits.saturating_add(search_stats.hint_hits);
    let mut r = match outcome {
        Outcome::Win(w) => {
            let mut r = DriverResult::new(Verdict::Win);
            r.depth = Some(w.depth);
            r.pv = w.pv;
            r
        }
        Outcome::No => DriverResult::new(Verdict::No),
        Outcome::BudgetExceeded => {
            let mut r = DriverResult::new(Verdict::BudgetExceeded);
            if let Some(best) = best_win
                && dfpn::pv_replays_win(pos, &best.pv)
            {
                r.depth = Some(best.depth);
                r.pv = best.pv;
            }
            r
        }
    };
    r.stats.nodes = total_nodes;
    r.stats.tt_hits = total_hint_hits;
    r.stats.leaf_solves = probes;
    r.stats.line_steps = tightened;
    r.stats.best_upper_depth = search_stats.best_upper;
    r.stats.excluded_through_depth = search_stats.excluded_through;
    r.stats.screened_attacks = screened_attacks;
    r.stats.screen_refuted_attacks = screen_refuted_attacks;
    r.stats.screen_winning_attacks = screen_winning_attacks;
    r.stats.screen_unresolved_attacks = screen_unresolved_attacks;
    r.stats.elapsed_s = t.elapsed().as_secs_f64();
    r.certificate = Some(certificate.clone());
    Ok(r)
}

/// Find the shortest forced win by binary-searching attacker-turn thresholds
/// with horizon-aware PDS-PN. The independently verified certificate supplies a
/// sound initial winning upper bound; every completed bounded disproof raises the
/// lower bound. Only adjacent bounds produce an authoritative shortest claim.
pub fn guided_pdspn_shortest(
    pos: &Position,
    certificate: &ProofCertificate,
    cfg: &ProverConfig,
    ctl: &Ctl,
) -> Result<DriverResult, String> {
    let certificate_wide = match certificate.width.as_str() {
        "tight" => false,
        "wide" => true,
        other => return Err(format!("invalid proof width {other:?}")),
    };
    if certificate_wide != cfg.wide {
        return Err(format!(
            "proof width {:?} does not match requested {} search",
            certificate.width,
            cfg.width_str(),
        ));
    }
    if cfg.depth_cap == 0 {
        return Err("shortest search depth cap must be at least 1".to_string());
    }
    let summary = certificate::verify(pos, certificate)?;
    let original_upper = u8::try_from(summary.max_attacker_turns)
        .map_err(|_| "certificate depth exceeds the supported 255-turn bound".to_string())?;
    if original_upper == 0 {
        return Err("certificate reported a zero-turn win".to_string());
    }

    #[cfg(not(target_arch = "wasm32"))]
    let started = Instant::now();
    let mut upper = original_upper;
    let mut lower = 0u8;
    let mut current_cert = certificate.clone();
    let mut total_nodes = 0u64;
    let mut total_tt_hits = 0u64;
    let mut total_leaf_solves = 0u64;
    let mut probes = 0u64;
    let mut tt_bytes = 0u64;
    let mut best_pv = Vec::new();

    // Top-down certificate-guided probing. Each probe asks the exact next
    // depth below the current certified upper bound, and the certificate's
    // win-depth hints close every node it already resolves at that horizon —
    // so a probe only re-searches the certificate's critical nodes (the ones
    // whose certified depth exceeds the probe's remaining-turn budget). A WIN
    // probe tightens the bound by at least one turn and adopts the tighter
    // certificate; a NO probe proves the previous bound exact; a budget miss
    // stops the walk and reports the verified interval honestly.
    //
    // If the user's cap is below the certificate, the first probe asks that
    // exact cap: a miss is still valuable (`no win <= cap`) but we must not
    // silently search above the requested ceiling.
    let mut target = if cfg.depth_cap < upper {
        cfg.depth_cap
    } else {
        upper.saturating_sub(1)
    };
    while lower.saturating_add(1) < upper {
        if ctl.expired() || total_nodes >= cfg.node_budget {
            break;
        }
        let (_hint_summary, hints, _guide_nodes) =
            certificate::verify_with_hints(pos, &current_cert)?;
        let hints = Rc::new(hints);
        let mut probe_cfg = cfg.clone();
        probe_cfg.depth_cap = target;
        probe_cfg.node_budget = cfg.node_budget.saturating_sub(total_nodes);
        let probe = pdspn::solve_bounded_guided(pos, &probe_cfg, ctl, Rc::clone(&hints));
        probes += 1;
        total_nodes = total_nodes.saturating_add(probe.stats.nodes);
        total_tt_hits = total_tt_hits.saturating_add(probe.stats.tt_hits);
        total_leaf_solves = total_leaf_solves.saturating_add(probe.stats.leaf_solves);
        tt_bytes = tt_bytes.max(probe.stats.tt_bytes);

        match probe.verdict {
            Verdict::Win => {
                // Adopt the probe's certificate when it re-verifies at the new
                // bound. The probe may have found an even shorter win than
                // `target`; the verified summary is the honest new upper.
                if let Some(candidate) = probe.certificate
                    && let Ok(candidate_summary) = certificate::verify(pos, &candidate)
                {
                    let new_upper = u8::try_from(candidate_summary.max_attacker_turns)
                        .unwrap_or(upper);
                    upper = upper.min(new_upper);
                    current_cert = candidate;
                } else {
                    upper = target;
                }
                if !probe.pv.is_empty() && dfpn::pv_replays_win(pos, &probe.pv) {
                    best_pv = probe.pv;
                }
                target = upper.saturating_sub(1);
            }
            Verdict::No => {
                lower = target;
                break;
            }
            Verdict::BudgetExceeded | Verdict::Unverified => break,
        }
    }

    let exact = lower.saturating_add(1) == upper;
    // The UI may call the result shortest only when the saved certificate
    // itself proves the reported upper. It need not be numerically tighter:
    // an original one-turn certificate, or an original upper followed by an
    // adjacent NO probe, already matches the exact bound without a WIN probe.
    let cert_tightened = exact
        && certificate::verify(pos, &current_cert)
            .is_ok_and(|summary| summary.max_attacker_turns == u32::from(upper));
    // Present the defender's strongest tested resistance. The attacker takes
    // its quickest certified move and the defender takes the reply that delays
    // the win longest. This line witnesses the certificate's upper bound and is
    // much more useful than a cooperative example when explaining refutations.
    if let Ok(pv) = certificate::worst_case_pv(pos, &current_cert)
        && dfpn::pv_replays_win(pos, &pv)
    {
        best_pv = pv;
    } else if best_pv.is_empty()
        && let Ok(pv) = certificate::shortest_pv(pos, &current_cert)
        && dfpn::pv_replays_win(pos, &pv)
    {
        best_pv = pv;
    }
    let mut result = DriverResult::new(if exact { Verdict::Win } else { Verdict::BudgetExceeded });
    result.depth = Some(upper);
    result.pv = best_pv;
    result.certificate = Some(current_cert);
    result.stats.nodes = total_nodes;
    result.stats.tt_hits = total_tt_hits;
    result.stats.tt_bytes = tt_bytes;
    #[cfg(not(target_arch = "wasm32"))]
    {
        result.stats.elapsed_s = started.elapsed().as_secs_f64();
    }
    result.stats.leaf_solves = total_leaf_solves;
    result.stats.line_steps = probes;
    result.stats.best_upper_depth = upper;
    result.stats.excluded_through_depth = lower;
    result.stats.cert_tightened = cert_tightened;
    Ok(result)
}

/// Report-form wrapper used by the research CLI's `--guide-certificate` path.
pub fn run_guided(
    pos: &Position,
    certificate: &ProofCertificate,
    cfg: &ProverConfig,
) -> Result<Report, String> {
    let ctl = Ctl::new(cfg.time_limit_s);
    let (res, driver) = if cfg.driver == DriverKind::PdspnShortest {
        (guided_pdspn_shortest(pos, certificate, cfg, &ctl)?, "pdspn-shortest")
    } else {
        (guided_idtt(pos, certificate, cfg, &ctl)?, "pds-idtt")
    };
    Ok(Report {
        verdict: res.verdict,
        depth: res.depth,
        pv: res.pv,
        driver: driver.to_string(),
        width: cfg.width_str().to_string(),
        stats: res.stats,
        race: Vec::new(),
        unverified: res.unverified,
        certificate: res.certificate,
    })
}

/// Run a single (non-race) driver.
fn run_one(driver: DriverKind, pos: &Position, lines: &[Line], cfg: &ProverConfig, ctl: &Ctl) -> DriverResult {
    match driver {
        DriverKind::Idtt => idtt(pos, cfg, ctl),
        DriverKind::Dfpn => dfpn::solve(pos, cfg, ctl),
        DriverKind::Pdspn => pdspn::solve(pos, cfg, ctl),
        DriverKind::PdspnDepth => {
            let mut result = pdspn::solve_bounded(pos, cfg, ctl);
            match result.verdict {
                Verdict::Win => {
                    // The bounded proof certifies the caller's threshold. Its
                    // cosmetic PV is one defence line and cannot tighten the
                    // all-defence worst-case bound by itself.
                    result.depth = Some(cfg.depth_cap);
                    result.stats.best_upper_depth = cfg.depth_cap;
                }
                Verdict::No => {
                    // A bounded disproof is a shortestness lower bound, not a
                    // global NO. Preserve that distinction in the report.
                    result.verdict = Verdict::BudgetExceeded;
                    result.stats.excluded_through_depth = cfg.depth_cap;
                }
                Verdict::BudgetExceeded | Verdict::Unverified => {}
            }
            result
        }
        DriverKind::PdspnShortest => DriverResult::new(Verdict::BudgetExceeded),
        DriverKind::Hybrid => hybrid::verify(pos, lines, cfg, ctl),
        DriverKind::Race => unreachable!("race is dispatched by run(), not run_one()"),
    }
}

/// Top-level entry: dispatch the configured driver (or the portfolio race) and
/// assemble the JSON report.
pub fn run(pos: &Position, lines: &[Line], cfg: &ProverConfig) -> Report {
    // Default driver selection: hybrid if lines were provided, else dfpn — but the
    // caller (CLI) sets `cfg.driver` explicitly; this is only a safety net.
    let driver = cfg.driver;
    if driver == DriverKind::Race {
        return portfolio::race(pos, lines, cfg);
    }
    let ctl = Ctl::new(cfg.time_limit_s);
    let res = run_one(driver, pos, lines, cfg, &ctl);
    Report {
        verdict: res.verdict,
        depth: res.depth,
        pv: res.pv,
        driver: driver.name().to_string(),
        width: cfg.width_str().to_string(),
        stats: res.stats,
        race: Vec::new(),
        unverified: res.unverified,
        certificate: res.certificate,
    }
}
