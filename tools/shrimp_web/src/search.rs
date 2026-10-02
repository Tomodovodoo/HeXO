//! hexo-bot's lockstep search (`ShrimpMctsSession.search` in packages/shrimp/rust/src/search.rs) for one root, as
//! a resumable step machine, and the evaluator payload handling of payload.rs. The selection, early-stop, LCB and
//! tactical-guard functions are hexo-bot's, unchanged apart from the error type; what differs is only who calls the
//! network: `Session::step` returns when the leaves need evaluations and `Session::fulfill` takes them.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use half::f16;
use half::slice::HalfFloatSliceExt;
use hexo_engine::{
    apply_placement, pack_coord, unpack_coord, HexCoord, HexoState as RustHexoState, PackedCoord, Placement,
};
use hexo_utils::StateHash;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::cache::{
    lock_cache, new_shared_evaluation_cache, state_hash, RustEvaluation, SharedEvaluationCache,
};
use crate::constants::NUM_FEATURES;
use crate::features::build_features;
use crate::support::build_support;
use crate::threats_shared as threats;
use crate::tree::{random_unit, terminal_value, Divergences, RustEdge, RustLeaf, RustNode, RustSearch, Widening};

const ACTIVE_ROOT_LIMIT: usize = 512;
const SEED_STREAM_GUMBEL: u64 = 6;

/// The keyword arguments of `ShrimpMctsSession.search`, with its defaults.
struct Settings {
    c_puct: Option<f32>,
    virtual_batch_size: Option<u32>,
    active_root_limit: usize,
    root_policy_temperature: f32,
    fpu_reduction: f32,
    virtual_loss: f32,
    widening_policy_mass: Option<f32>,
    widening_max_children: Option<u32>,
    widening_min_children: Option<u32>,
    tss_enabled: bool,
    root_fpu_reduction: Option<f32>,
    divergences: Divergences,
}

/// A request for network evaluations: the featurized unique states, and for every state asked for either its cached
/// evaluation or its unique row.
pub struct Request {
    pub nodes: usize,
    pub offsets: Vec<i32>,
    pub legal: Vec<i32>,
    pub features: Vec<f32>,
    pub coords: Vec<i32>,
    pub neighbours: Vec<i32>,
    legal_ids: Vec<Vec<PackedCoord>>,
    keys: Vec<StateHash>,
    slots: Vec<Slot>,
    answer: Option<Vec<Arc<RustEvaluation>>>,
}

enum Slot {
    Ready(Arc<RustEvaluation>),
    Row(usize),
}

/// The chosen stone and the root's statistics.
pub struct SearchResult {
    pub moves: Vec<i32>,
    pub stats: Vec<f32>,
    pub root_value: f32,
    pub visits: u32,
}

struct Job {
    game_key: u64,
    seed: u64,
    visits: u32,
    leaf_batch: u32,
    root_policy_temperature: f32,
    root_fpu_reduction: f32,
    fpu_reduction: f32,
    virtual_loss: f32,
    widening: Widening,
    request_ml: bool,
    request_logits: bool,
    root: Option<RustHexoState>,
    search: Option<RustSearch>,
    baseline: HashMap<PackedCoord, u32>,
    pending: Vec<RustLeaf>,
    request: Option<Request>,
    result: Option<SearchResult>,
}

pub struct Session {
    settings: Settings,
    searches: HashMap<u64, RustSearch>,
    cache: SharedEvaluationCache,
    cache_max_states: usize,
    job: Option<Job>,
}

impl Session {
    pub fn new(max_states: usize, support_radius: i32, search_parity_mode: bool) -> PyResult<Self> {
        if std::env::var("SHRIMP_SUPPORT_RADIUS").is_err() {
            std::env::set_var("SHRIMP_SUPPORT_RADIUS", support_radius.to_string());
        }
        // The featurizer reads its radius once per module; a lone stone has 3R(R+1) legal cells within radius R.
        let mut probe = RustHexoState::new();
        apply_placement(&mut probe, Placement { coord: HexCoord { q: 0, r: 0 } }).map_err(crate::state::move_error)?;
        let legal = build_support(&probe).legal_count as i32;
        if legal != 3 * support_radius * (support_radius + 1) {
            return Err(PyValueError::new_err(format!(
                "the featurizer runs at a different support radius than {support_radius} ({legal} legal cells)"
            )));
        }
        Ok(Self {
            settings: Settings {
                c_puct: None,
                virtual_batch_size: None,
                active_root_limit: ACTIVE_ROOT_LIMIT,
                root_policy_temperature: 1.0,
                fpu_reduction: 0.20,
                virtual_loss: 1.0,
                widening_policy_mass: None,
                widening_max_children: None,
                widening_min_children: None,
                tss_enabled: true,
                root_fpu_reduction: None,
                divergences: if search_parity_mode { Divergences::parity() } else { Divergences::production() },
            },
            searches: HashMap::new(),
            cache: new_shared_evaluation_cache(),
            cache_max_states: validate_positive_usize("max_states", max_states)?,
            job: None,
        })
    }

    /// One keyword of `search` or one key of its `divergence_overrides`, as `resolve_divergences` applies them.
    pub fn set(&mut self, key: &str, value: f64) -> PyResult<()> {
        let s = &mut self.settings;
        let d = &mut s.divergences;
        let flag = value != 0.0;
        let real = value as f32;
        let count = || -> PyResult<u32> {
            if value >= 0.0 && value.fract() == 0.0 && value <= u32::MAX as f64 {
                Ok(value as u32)
            } else {
                Err(PyValueError::new_err(format!("{key} must be a non-negative integer")))
            }
        };
        match key {
            "c_puct" => s.c_puct = Some(real),
            "virtual_batch_size" => s.virtual_batch_size = Some(count()?),
            "active_root_limit" => s.active_root_limit = count()? as usize,
            "root_policy_temperature" => s.root_policy_temperature = real,
            "fpu_reduction" => s.fpu_reduction = real,
            "virtual_loss" => s.virtual_loss = real,
            "widening_policy_mass" => s.widening_policy_mass = Some(real),
            "widening_max_children" => s.widening_max_children = Some(count()?),
            "widening_min_children" => s.widening_min_children = Some(count()?),
            "tss_enabled" => s.tss_enabled = flag,
            "root_fpu_reduction" => s.root_fpu_reduction = Some(real),
            "lcb_move_selection" => d.lcb_move_selection = flag,
            "early_stop" => d.early_stop = flag,
            "moves_left_utility" => d.moves_left_utility = flag,
            "ml_weight" => d.ml_weight = real,
            "ml_scale" => d.ml_scale = real,
            "ml_q_gate" => d.ml_q_gate = real,
            "ml_two_sided" => d.ml_two_sided = flag,
            "ml_final_pick" => d.ml_final_pick = flag,
            "ml_final_pick_band" => d.ml_final_pick_band = real,
            "lcb_z" => d.lcb_z = real,
            "nucleus_f64" => d.nucleus_f64 = flag,
            "new_child_fpu" => d.new_child_fpu = flag,
            "lazy_widening" => d.lazy_widening = flag,
            "clean_root_prior_cache" => d.clean_root_prior_cache = flag,
            "gumbel_target" => d.gumbel_target = flag,
            "gumbel_root" => d.gumbel_root = flag,
            "gumbel_sequential_halving" => d.gumbel_sequential_halving = flag,
            "gumbel_nonroot_select" => d.gumbel_nonroot_select = flag,
            "gumbel_c_visit" => d.gumbel_c_visit = real,
            "gumbel_c_scale" => d.gumbel_c_scale = real,
            "gumbel_target_c_scale" => d.gumbel_target_c_scale = Some(real),
            "gumbel_m" => d.gumbel_m = count()?,
            "gumbel_draw_temperature" => d.gumbel_draw_temperature = real,
            "gumbel_target_min_visits" => d.gumbel_target_min_visits = count()?,
            "gumbel_play_prune" => d.gumbel_play_prune = flag,
            _ => return Err(PyValueError::new_err(format!("unknown shrimp search setting {key:?}"))),
        }
        Ok(())
    }

    /// Starts one greedy (temperature 0) search of `visits` from the position of `moves`, as the driver's
    /// `session.search([game_key], (state,), visits=..., temperature=0.0, seed=..., move_temperatures=[0.0], ...)`.
    pub fn begin(&mut self, moves: &[i32], visits: u32, seed: u64, game_key: u64) -> PyResult<()> {
        self.job = None;
        let s = &self.settings;
        let c_puct = s.c_puct.ok_or_else(|| PyValueError::new_err("c_puct is not set"))?;
        validate_search_inputs(visits, c_puct, 0.0)?;
        let divergences = s.divergences;
        if validate_positive_usize("active_root_limit", s.active_root_limit)? < 1 {
            return Err(PyValueError::new_err("one root is above the active root limit"));
        }
        let leaf_batch = validate_positive_u32("virtual_batch_size", s.virtual_batch_size.unwrap_or(visits))?;
        let root_policy_temperature = validate_positive_f32("root_policy_temperature", s.root_policy_temperature)?;
        let fpu_reduction = validate_nonnegative_f32("fpu_reduction", s.fpu_reduction)?;
        let virtual_loss = validate_nonnegative_f32("virtual_loss", s.virtual_loss)?;
        let root_fpu_reduction = match s.root_fpu_reduction {
            Some(value) => validate_nonnegative_f32("root_fpu_reduction", value)?,
            None => fpu_reduction,
        };
        let widening = build_widening(s.widening_policy_mass, s.widening_min_children, s.widening_max_children)?;
        let tss_enabled = s.tss_enabled;
        let root = replay(moves)?;
        if root.is_terminal() {
            return Err(PyValueError::new_err("the game has finished"));
        }
        let mut job = Job {
            game_key,
            seed,
            visits,
            leaf_batch,
            root_policy_temperature,
            root_fpu_reduction,
            fpu_reduction,
            virtual_loss,
            widening,
            request_ml: divergences.moves_left_utility,
            request_logits: divergences.gumbel_target || divergences.gumbel_root || divergences.gumbel_nonroot_select,
            root: None,
            search: None,
            baseline: HashMap::new(),
            pending: Vec::new(),
            request: None,
            result: None,
        };
        let root_hash = state_hash(&root);
        if let Some(mut search) = self.searches.remove(&game_key) {
            if search.root_hash == root_hash {
                search.set_additional_visits(visits);
                search.set_root_fpu_reduction(root_fpu_reduction);
                search.set_tss_enabled(tss_enabled);
                search.set_divergences(divergences);
                search.apply_root_policy_temperature(root_policy_temperature);
                if divergences.gumbel_root {
                    search.init_gumbel_root(mix_seed(seed, game_key ^ root_hash, 0, SEED_STREAM_GUMBEL), visits);
                } else {
                    search.clear_gumbel_root();
                }
                job.search = Some(search);
                start(&mut job, c_puct)?;
                self.job = Some(job);
                return Ok(());
            }
        }
        job.request = Some(prepare(&self.cache, &[(&root, root_hash)], job.request_ml, job.request_logits));
        job.root = Some(root);
        self.job = Some(job);
        Ok(())
    }

    fn job(&mut self) -> PyResult<&mut Job> {
        self.job.as_mut().ok_or_else(|| PyValueError::new_err("no search has begun"))
    }

    /// The rows the search waits for.
    pub fn request(&mut self) -> PyResult<&Request> {
        self.job()?
            .request
            .as_ref()
            .filter(|request| request.answer.is_none())
            .ok_or_else(|| PyValueError::new_err("the search is not waiting for evaluations"))
    }

    pub fn completed(&mut self) -> PyResult<u32> {
        Ok(self.job()?.search.as_ref().map_or(0, |search| search.completed_visits))
    }

    pub fn result(&mut self) -> PyResult<&SearchResult> {
        self.job()?.result.as_ref().ok_or_else(|| PyValueError::new_err("the search has not finished"))
    }

    /// Runs `run_searches_to_targets` until the next evaluation request (its row count) or the end (0).
    pub fn step(&mut self) -> PyResult<usize> {
        let cache = Arc::clone(&self.cache);
        let c_puct = self.settings.c_puct.unwrap_or(0.0);
        let tss_enabled = self.settings.tss_enabled;
        let divergences = self.settings.divergences;
        loop {
            let job = self.job()?;
            if job.result.is_some() {
                return Ok(0);
            }
            if let Some(request) = &job.request {
                if request.answer.is_none() {
                    return Ok(request.legal.len());
                }
            }
            if let Some(request) = job.request.take() {
                let evaluations = request.answer.expect("an answered request");
                match job.search.as_mut() {
                    None => {
                        let root = job.root.take().expect("a root waiting for its evaluation");
                        let root_hash = state_hash(&root);
                        let mut search = RustSearch::new(
                            root,
                            &evaluations[0],
                            job.visits,
                            job.fpu_reduction,
                            job.root_fpu_reduction,
                            job.root_policy_temperature,
                            job.widening,
                            tss_enabled,
                            divergences,
                        )?;
                        if divergences.gumbel_root {
                            search.init_gumbel_root(
                                mix_seed(job.seed, job.game_key ^ root_hash, 0, SEED_STREAM_GUMBEL),
                                job.visits,
                            );
                        }
                        job.search = Some(search);
                        start(job, c_puct)?;
                    }
                    Some(search) => {
                        let pending = std::mem::take(&mut job.pending);
                        let next = if search.needs_visits() {
                            select_leaf_batch(search, c_puct, job.leaf_batch, job.virtual_loss, &pending)?.0
                        } else {
                            Vec::new()
                        };
                        apply_eval_backups(search, pending, &evaluations, job.virtual_loss)?;
                        job.pending = next;
                    }
                }
                continue;
            }
            let search = job.search.as_mut().expect("a running search");
            early_stop_pass(search, &job.baseline);
            if job.pending.is_empty() {
                if !search.needs_visits() {
                    self.finish()?;
                    return Ok(0);
                }
                let (leaves, made_progress) = select_leaf_batch(search, c_puct, job.leaf_batch, job.virtual_loss, &[])?;
                if leaves.is_empty() {
                    if !made_progress {
                        self.finish()?;
                        return Ok(0);
                    }
                    continue;
                }
                job.pending = leaves;
            }
            let states: Vec<(&RustHexoState, StateHash)> =
                job.pending.iter().map(|leaf| (&leaf.state, leaf.state_hash)).collect();
            let request = prepare(&cache, &states, job.request_ml, job.request_logits);
            job.request = Some(request);
        }
    }

    /// The network's answers to the current request (see `sh_fulfill`), parsed as `parse_chunk_reply` and stored in
    /// the cache as `evaluate_state_refs_cached` does.
    pub fn fulfill(&mut self, values: &[f32], moves_left: &[f32], logits: &[f32]) -> PyResult<()> {
        let cache = Arc::clone(&self.cache);
        let max_states = self.cache_max_states;
        let job = self.job()?;
        let (request_ml, request_logits) = (job.request_ml, job.request_logits);
        let request = job
            .request
            .as_mut()
            .filter(|request| request.answer.is_none())
            .ok_or_else(|| PyValueError::new_err("the search is not waiting for evaluations"))?;
        let mut evaluations = Vec::with_capacity(request.legal_ids.len());
        let mut base = 0usize;
        for (row, legal_ids) in request.legal_ids.iter().enumerate() {
            let value = values[row];
            if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                return Err(PyValueError::new_err(format!(
                    "values_bytes row {row} must be finite and in [-1, 1], got {value}"
                )));
            }
            let row_logits = &logits[base..base + legal_ids.len()];
            base += legal_ids.len();
            if let Some(bad) = row_logits.iter().find(|l| !l.is_finite()) {
                return Err(PyValueError::new_err(format!("priors_logits_bytes row {row} must be finite, got {bad}")));
            }
            let mut priors: Vec<(PackedCoord, f32)> =
                legal_ids.iter().copied().zip(softmax(row_logits)).collect();
            finalize_priors(&mut priors, legal_ids.len(), row)?;
            let moves_left = if request_ml {
                let ml = moves_left[row];
                if !ml.is_finite() || !(0.0..=512.0).contains(&ml) {
                    return Err(PyValueError::new_err(format!(
                        "moves_left_bytes row {row} must be in [0, 512], got {ml}"
                    )));
                }
                Some(ml)
            } else {
                None
            };
            priors.shrink_to_fit();
            evaluations.push(Arc::new(RustEvaluation {
                value,
                legal_action_count: legal_ids.len(),
                priors,
                moves_left,
                logits: request_logits.then(|| legal_ids.iter().copied().zip(row_logits.iter().copied()).collect()),
            }));
        }
        {
            let mut cached = lock_cache(&cache);
            for (key, evaluation) in request.keys.iter().copied().zip(evaluations.iter()) {
                cached.insert_bounded(key, Arc::clone(evaluation), max_states);
            }
        }
        request.answer = Some(
            request
                .slots
                .iter()
                .map(|slot| match slot {
                    Slot::Ready(evaluation) => Arc::clone(evaluation),
                    Slot::Row(row) => Arc::clone(&evaluations[*row]),
                })
                .collect(),
        );
        Ok(())
    }

    /// The played stone as `build_search_result_payload_native` picks it at temperature 0, then the tree advanced
    /// to it and kept for the next stone of the turn.
    fn finish(&mut self) -> PyResult<()> {
        let job = self.job.as_mut().expect("a running job");
        let mut search = job.search.take().expect("a running search");
        let selected = select_search_action(&search, Some(&job.baseline), 0.0, job.seed)?;
        let root = search.root();
        let played = match selected {
            Some(action_id) => action_id,
            None => fallback_root_action(root).ok_or_else(|| {
                PyValueError::new_err("move selection found no legal root action (empty edges and priors)")
            })?,
        };
        let (ids, weights, q, total) = visit_policy(root, Some(&job.baseline));
        let mut order: Vec<usize> = (0..ids.len()).collect();
        order.sort_by(|&a, &b| {
            (ids[b] == played)
                .cmp(&(ids[a] == played))
                .then(weights[b].partial_cmp(&weights[a]).unwrap_or(std::cmp::Ordering::Equal))
                .then(ids[a].cmp(&ids[b]))
        });
        let mut moves = Vec::with_capacity(3 * (order.len() + 1));
        let mut stats = Vec::with_capacity(2 * (order.len() + 1));
        if !ids.contains(&played) {
            let coord = unpack_coord(played);
            moves.extend([coord.q as i32, coord.r as i32, 0]);
            stats.extend([root.edges.iter().find(|e| e.action_id == played).map_or(0.0, RustEdge::value), 0.0]);
        }
        for index in order {
            let coord = unpack_coord(ids[index]);
            let edge = root.edges.iter().find(|e| e.action_id == ids[index]).expect("a root edge");
            moves.extend([coord.q as i32, coord.r as i32, edge_delta_visits(edge, Some(&job.baseline)) as i32]);
            stats.extend([q[index], weights[index]]);
        }
        let root_value = root.value();
        if let Some(action_id) = selected {
            if search.advance_root(action_id)? {
                self.searches.insert(job.game_key, search);
            }
        }
        job.result = Some(SearchResult { moves, stats, root_value, visits: total });
        Ok(())
    }
}

fn coordinate(value: i32) -> PyResult<i16> {
    i16::try_from(value).map_err(|_| PyValueError::new_err(format!("coordinate {value} is out of range")))
}

fn replay(moves: &[i32]) -> PyResult<RustHexoState> {
    let mut state = RustHexoState::new();
    for pair in moves.chunks_exact(2) {
        let coord = HexCoord { q: coordinate(pair[0])?, r: coordinate(pair[1])? };
        apply_placement(&mut state, Placement { coord }).map_err(crate::state::move_error)?;
    }
    Ok(state)
}

/// See `sh_position`.
pub fn position(moves: &[i32]) -> PyResult<i32> {
    let state = replay(moves)?;
    Ok(match state.terminal() {
        Some(outcome) => 2 + 4 * outcome.winner.index() as i32,
        None => state.current_player().index() as i32,
    })
}

/// The root checks, the reuse baseline and the priming select of `run_searches_to_targets`.
fn start(job: &mut Job, c_puct: f32) -> PyResult<()> {
    let search = job.search.as_mut().expect("a search");
    if search.root_edges_empty() {
        return Err(PyValueError::new_err("MCTS root has no legal actions"));
    }
    job.baseline = search.root_edge_visits().into_iter().collect();
    early_stop_pass(search, &job.baseline);
    job.pending = select_leaf_batch(search, c_puct, job.leaf_batch, job.virtual_loss, &[])?.0;
    Ok(())
}

/// `evaluate_state_refs_cached` up to the evaluator call: cache hits and duplicates resolve here, every other state
/// becomes one featurized row.
fn prepare(
    cache: &SharedEvaluationCache,
    states: &[(&RustHexoState, StateHash)],
    request_ml: bool,
    request_logits: bool,
) -> Request {
    let mut request = Request {
        nodes: 0,
        offsets: vec![0],
        legal: Vec::new(),
        features: Vec::new(),
        coords: Vec::new(),
        neighbours: Vec::new(),
        legal_ids: Vec::new(),
        keys: Vec::new(),
        slots: Vec::with_capacity(states.len()),
        answer: None,
    };
    let mut rows: HashMap<StateHash, usize> = HashMap::new();
    let mut unique: Vec<&RustHexoState> = Vec::new();
    {
        let cached = lock_cache(cache);
        for &(state, key) in states {
            if let Some(evaluation) = cached.get(&key) {
                if (!request_ml || evaluation.moves_left.is_some()) && (!request_logits || evaluation.logits.is_some()) {
                    request.slots.push(Slot::Ready(evaluation));
                    continue;
                }
            }
            if let Some(&row) = rows.get(&key) {
                request.slots.push(Slot::Row(row));
                continue;
            }
            rows.insert(key, unique.len());
            request.keys.push(key);
            request.slots.push(Slot::Row(unique.len()));
            unique.push(state);
        }
    }
    for state in unique {
        let support = build_support(state);
        let features = build_features(state, &support);
        let mut half = vec![f16::ZERO; features.len()];
        half.convert_from_f32_slice(&features);
        debug_assert_eq!(features.len(), support.num_nodes() * NUM_FEATURES);
        request.features.extend(half.iter().map(|x| x.to_f32()));
        for c in &support.coords {
            request.coords.extend([c.q as i32, c.r as i32]);
        }
        for row in &support.nbr {
            request.neighbours.extend_from_slice(row);
        }
        request.nodes += support.num_nodes();
        request.offsets.push(request.nodes as i32);
        request.legal.push(support.legal_count as i32);
        request.legal_ids.push(support.coords[..support.legal_count].iter().map(|&c| pack_coord(c)).collect());
    }
    if request.legal_ids.is_empty() {
        request.answer = Some(
            request
                .slots
                .iter()
                .map(|slot| match slot {
                    Slot::Ready(evaluation) => Arc::clone(evaluation),
                    Slot::Row(_) => unreachable!("a row without a featurized state"),
                })
                .collect(),
        );
    }
    request
}

/// The evaluator's prior decode: softmax over a row's legal cells in f32, scaled by the reciprocal of the sum as
/// PyTorch's CPU softmax does.
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let scale = 1.0 / exps.iter().sum::<f32>();
    exps.into_iter().map(|e| e * scale).collect()
}

fn early_stop_pass(search: &mut RustSearch, baseline: &HashMap<PackedCoord, u32>) {
    if search.needs_visits() && early_stop_ready(search, Some(baseline), false, 0) {
        search.early_stopped = true;
        search.target_visits = search.completed_visits;
    }
}

fn finalize_priors(priors: &mut [(PackedCoord, f32)], legal_action_count: usize, row_index: usize) -> PyResult<()> {
    if legal_action_count == 0 {
        if priors.is_empty() {
            return Ok(());
        }
        return Err(PyValueError::new_err(format!(
            "evaluator returned {} priors for terminal row {row_index}",
            priors.len()
        )));
    }
    if priors.is_empty() {
        return Err(PyValueError::new_err(format!(
            "evaluator returned no priors for non-terminal row {row_index}"
        )));
    }
    let mut seen = HashSet::with_capacity(priors.len());
    let mut total = 0.0f32;
    for (action_id, prior) in priors.iter().copied() {
        if !seen.insert(action_id) {
            return Err(PyValueError::new_err(format!("duplicate action {action_id} in row {row_index}")));
        }
        if !prior.is_finite() || prior < 0.0 {
            return Err(PyValueError::new_err(format!(
                "invalid prior {prior} for action {action_id} in row {row_index}"
            )));
        }
        total += prior;
    }
    if total <= 0.0 {
        return Err(PyValueError::new_err(format!("zero total prior mass for row {row_index}")));
    }
    priors.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    for entry in priors.iter_mut() {
        entry.1 /= total;
    }
    Ok(())
}

fn select_leaf_batch(
    search: &mut RustSearch,
    c_puct: f32,
    leaf_batch_per_root: u32,
    virtual_loss: f32,
    in_flight: &[RustLeaf],
) -> PyResult<(Vec<RustLeaf>, bool)> {
    let mut leaves = Vec::new();
    let mut made_progress = false;
    if !search.needs_visits() {
        return Ok((leaves, made_progress));
    }
    let drained = in_flight.is_empty();
    if drained && search.has_gumbel_root() {
        while search.maybe_advance_gumbel_round() {}
    }
    let budget = leaf_batch_per_root.min(search.remaining_visits());
    for _ in 0..budget {
        let selected = search.select_pending_leaf(c_puct)?;
        let Some(selected) = selected else {
            break;
        };
        search.apply_virtual_visit(&selected.path, virtual_loss);
        made_progress = true;

        let ml_on = search.divergences.moves_left_utility;
        if let Some(outcome) = selected.terminal {
            let leaf_player = selected.state.current_player();
            let leaf_value = terminal_value(outcome, leaf_player);
            let leaf_ml = ml_on.then_some(0.0);
            search.backup_virtual(&selected.path, leaf_player, leaf_value, virtual_loss, leaf_ml);
        } else if let Some(node_id) = selected.existing_node {
            let node = &search.nodes[node_id];
            let player = node.player;
            let value = node.value();
            let leaf_ml = if ml_on { node.ml_mean() } else { None };
            search.backup_virtual(&selected.path, player, value, virtual_loss, leaf_ml);
        } else if let Some(verdict) = search
            .tss_enabled
            .then(|| threats::analyze(&selected.state).verdict())
            .flatten()
        {
            let leaf_player = selected.state.current_player();
            search.backup_virtual(&selected.path, leaf_player, verdict, virtual_loss, None);
        } else {
            search.mark_pending(selected.parent_node, selected.edge_index, 1);
            leaves.push(RustLeaf {
                root_index: 0,
                parent_node: selected.parent_node,
                edge_index: selected.edge_index,
                path: selected.path,
                state: selected.state,
                state_hash: selected.state_hash,
            });
        }
    }
    Ok((leaves, made_progress))
}

fn apply_eval_backups(
    search: &mut RustSearch,
    leaves: Vec<RustLeaf>,
    evaluations: &[Arc<RustEvaluation>],
    virtual_loss: f32,
) -> PyResult<()> {
    for (leaf, evaluation) in leaves.into_iter().zip(evaluations.iter()) {
        let child_id = search.add_node_from_eval(&leaf.state, leaf.state_hash, Arc::clone(evaluation))?;
        search.nodes[leaf.parent_node].edges[leaf.edge_index].child = Some(child_id);
        search.mark_pending(leaf.parent_node, leaf.edge_index, -1);
        let child_player = search.nodes[child_id].player;
        let child_value = search.nodes[child_id].value();
        let leaf_ml = if search.divergences.moves_left_utility {
            search.nodes[child_id].ml_mean()
        } else {
            None
        };
        search.backup_virtual(&leaf.path, child_player, child_value, virtual_loss, leaf_ml);
    }
    Ok(())
}

/// Early-stop test. Greedy unrecorded searches stop when the remaining budget cannot overtake the visit leader AND,
/// when LCB selection is active, the LCB winner currently equals the visit winner.
fn early_stop_ready(
    search: &RustSearch,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    recorded_full: bool,
    in_flight: u32,
) -> bool {
    let dv = search.divergences;
    if !dv.early_stop || in_flight > 0 {
        return false;
    }
    let remaining = search.remaining_visits();
    if remaining == 0 {
        return false;
    }
    if recorded_full {
        let floor = (search.target_visits as f32 * dv.full_visit_floor).ceil() as u32;
        if search.completed_visits < floor {
            return false;
        }
    }
    let root = search.root();
    let stats = lcb_stats(root, baseline);
    let mut best = 0u32;
    let mut second = 0u32;
    let mut best_id: Option<PackedCoord> = None;
    for &(action_id, delta, _visits, _value_sum, _value_sq_sum) in &stats {
        if delta > best {
            second = best;
            best = delta;
            best_id = Some(action_id as PackedCoord);
        } else if delta > second {
            second = delta;
        }
    }
    let Some(best_id) = best_id else {
        return false;
    };
    if best.saturating_sub(second) <= remaining {
        return false;
    }
    if dv.lcb_move_selection && !recorded_full {
        if let Some(lcb_id) =
            debug_lcb_from_stats(&stats, dv.lcb_z, dv.lcb_min_visits, dv.lcb_visit_fraction).map(|id| id as PackedCoord)
        {
            if lcb_id != best_id {
                return false;
            }
        }
    }
    true
}

fn lcb_stats(root: &RustNode, baseline: Option<&HashMap<PackedCoord, u32>>) -> Vec<(u64, u32, u32, f32, f32)> {
    root.edges
        .iter()
        .map(|edge| {
            (
                edge.action_id as u64,
                edge_delta_visits(edge, baseline),
                edge.visits,
                edge.value_sum,
                edge.value_sq_sum,
            )
        })
        .collect()
}

fn lcb_pick(root: &RustNode, baseline: Option<&HashMap<PackedCoord, u32>>, dv: &Divergences) -> Option<PackedCoord> {
    let stats = lcb_stats(root, baseline);
    debug_lcb_from_stats(&stats, dv.lcb_z, dv.lcb_min_visits, dv.lcb_visit_fraction).map(|id| id as PackedCoord)
}

/// Final-move decisiveness tie-break among guard-positive moves within `ml_final_pick_band` of the LCB leader.
fn ml_final_pick(
    root: &RustNode,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    dv: &Divergences,
    action_ids: &[PackedCoord],
    guarded_weights: &[f32],
) -> Option<PackedCoord> {
    let root_v = root.value();
    let dir: i32 = if root_v > dv.ml_q_gate {
        1
    } else if root_v < -dv.ml_q_gate {
        -1
    } else {
        return None;
    };
    let stats = lcb_stats(root, baseline);
    let max_delta = stats.iter().map(|s| s.1).max().unwrap_or(0);
    if max_delta == 0 {
        return None;
    }
    let threshold = (dv.lcb_min_visits as f32).max(dv.lcb_visit_fraction * max_delta as f32);
    let mut best_lcb = f32::NEG_INFINITY;
    let mut eligible: Vec<(PackedCoord, f32)> = Vec::new();
    for &(action_id, delta, visits, value_sum, value_sq_sum) in &stats {
        if (delta as f32) < threshold || visits == 0 {
            continue;
        }
        let n = visits as f32;
        let q = value_sum / n;
        let variance = (value_sq_sum / n - q * q).max(0.0);
        let lcb = q - dv.lcb_z * variance.sqrt() / n.sqrt();
        eligible.push((action_id as PackedCoord, lcb));
        if lcb > best_lcb {
            best_lcb = lcb;
        }
    }
    let mut pick: Option<(PackedCoord, f32)> = None;
    for &(id, lcb) in &eligible {
        if lcb < best_lcb - dv.ml_final_pick_band {
            continue;
        }
        let guard_positive = action_ids.iter().zip(guarded_weights.iter()).any(|(&aid, &w)| aid == id && w > 0.0);
        if !guard_positive {
            continue;
        }
        let Some(m) = root.edges.iter().find(|e| e.action_id == id).and_then(|e| e.ml_mean()) else {
            continue;
        };
        let better = match pick {
            None => true,
            Some((_, bm)) => {
                if dir == 1 {
                    m < bm
                } else {
                    m > bm
                }
            }
        };
        if better {
            pick = Some((id, m));
        }
    }
    pick.map(|(id, _)| id)
}

/// The most-visited root edge, else the highest-prior root action, when the delta-visit selection yields nothing.
fn fallback_root_action(root: &RustNode) -> Option<PackedCoord> {
    let by_visits = root
        .edges
        .iter()
        .max_by(|a, b| a.visits.cmp(&b.visits).then_with(|| b.action_id.cmp(&a.action_id)))
        .map(|edge| (edge.action_id, edge.visits));
    if let Some((action_id, visits)) = by_visits {
        if visits > 0 {
            return Some(action_id);
        }
    }
    let (prior_ids, prior_weights) = root_prior_policy(root);
    let best_prior = prior_ids
        .iter()
        .copied()
        .zip(prior_weights.iter().copied())
        .max_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.0.cmp(&a.0))
        })
        .map(|(action_id, _)| action_id);
    if best_prior.is_some() {
        return best_prior;
    }
    by_visits.map(|(action_id, _)| action_id)
}

fn root_prior_policy(root: &RustNode) -> (Vec<PackedCoord>, Vec<f32>) {
    let remaining = root.remaining_priors();
    let mut priors: HashMap<PackedCoord, f32> = HashMap::with_capacity(root.edges.len() + remaining.len());
    for edge in &root.edges {
        if edge.prior.is_finite() && edge.prior > 0.0 {
            priors.insert(edge.action_id, edge.prior);
        }
    }
    for (action_id, prior) in remaining {
        if prior.is_finite() && prior > 0.0 {
            priors.insert(action_id, prior);
        }
    }
    let mut pairs: Vec<(PackedCoord, f32)> = priors.into_iter().collect();
    pairs.sort_unstable_by_key(|(action_id, _prior)| *action_id);
    let action_ids: Vec<PackedCoord> = pairs.iter().map(|(action_id, _prior)| *action_id).collect();
    let mut weights: Vec<f32> = pairs.into_iter().map(|(_action_id, prior)| prior).collect();
    let total: f32 = weights.iter().copied().sum();
    if total > 0.0 {
        for weight in &mut weights {
            *weight /= total;
        }
    }
    (action_ids, weights)
}

fn validate_search_inputs(visits: u32, c_puct: f32, temperature: f32) -> PyResult<()> {
    if visits == 0 {
        return Err(PyValueError::new_err("visits must be > 0"));
    }
    if !c_puct.is_finite() || c_puct <= 0.0 {
        return Err(PyValueError::new_err("c_puct must be finite and > 0"));
    }
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(PyValueError::new_err("temperature must be finite and >= 0"));
    }
    Ok(())
}

fn validate_positive_u32(name: &str, value: u32) -> PyResult<u32> {
    if value == 0 {
        return Err(PyValueError::new_err(format!("{name} must be > 0")));
    }
    Ok(value)
}

fn validate_positive_usize(name: &str, value: usize) -> PyResult<usize> {
    if value == 0 {
        return Err(PyValueError::new_err(format!("{name} must be > 0")));
    }
    Ok(value)
}

fn validate_positive_f32(name: &str, value: f32) -> PyResult<f32> {
    if !value.is_finite() || value <= 0.0 {
        return Err(PyValueError::new_err(format!("{name} must be finite and > 0")));
    }
    Ok(value)
}

fn validate_nonnegative_f32(name: &str, value: f32) -> PyResult<f32> {
    if !value.is_finite() || value < 0.0 {
        return Err(PyValueError::new_err(format!("{name} must be finite and >= 0")));
    }
    Ok(value)
}

fn build_widening(mass: Option<f32>, min_children: Option<u32>, max_children: Option<u32>) -> PyResult<Widening> {
    let widening_mass = mass.unwrap_or(0.95);
    if !widening_mass.is_finite() || widening_mass <= 0.0 || widening_mass > 1.0 {
        return Err(PyValueError::new_err("widening_policy_mass must be in (0, 1]"));
    }
    let widening = Widening {
        mass: widening_mass,
        min_children: validate_positive_u32("widening_min_children", min_children.unwrap_or(2))? as usize,
        max_children: validate_positive_u32("widening_max_children", max_children.unwrap_or(32))? as usize,
    };
    if widening.min_children > widening.max_children {
        return Err(PyValueError::new_err("widening_min_children must be <= widening_max_children"));
    }
    Ok(widening)
}

/// The LCB formula of the search over per-edge (action_id, delta, visits, value_sum, value_sq_sum).
pub fn debug_lcb_from_stats(
    stats: &[(u64, u32, u32, f32, f32)],
    z: f32,
    min_visits: u32,
    visit_fraction: f32,
) -> Option<u64> {
    let max_delta = stats.iter().map(|s| s.1).max().unwrap_or(0);
    if max_delta == 0 {
        return None;
    }
    let threshold = (min_visits as f32).max(visit_fraction * max_delta as f32);
    let mut best: Option<(f32, u64)> = None;
    for &(action_id, delta, visits, value_sum, value_sq_sum) in stats {
        if (delta as f32) < threshold || visits == 0 {
            continue;
        }
        let n = visits as f32;
        let q = value_sum / n;
        let variance = (value_sq_sum / n - q * q).max(0.0);
        let lcb = q - z * variance.sqrt() / n.sqrt();
        let replace = match best {
            Some((current, current_id)) => lcb > current || (lcb == current && action_id < current_id),
            None => true,
        };
        if replace {
            best = Some((lcb, action_id));
        }
    }
    best.map(|(_, id)| id)
}

/// The moves-left utility bonus the tree adds to a child's score (bounded by `weight`).
pub fn debug_ml_bonus(q: f32, m_edge: f32, m_node: f32, weight: f32, scale: f32, gate: f32, two_sided: bool) -> f32 {
    let s = if q > gate {
        1.0
    } else if two_sided && q < -gate {
        -1.0
    } else {
        return 0.0;
    };
    -weight * s * ((m_edge - m_node) / scale).tanh()
}

pub fn mix_seed(base_seed: u64, game_key: u64, ply: u32, stream: u64) -> u64 {
    let mut value = base_seed ^ 0xA076_1D64_78BD_642F;
    value ^= game_key.wrapping_mul(0xE703_7ED1_A0B4_28DB);
    value ^= (ply as u64).wrapping_mul(0x8EBC_6AF0_9C88_C6E3);
    value ^= stream.wrapping_mul(0x5899_65CC_7537_4CC3);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn classify_root_move(root_state: &RustHexoState, action_id: PackedCoord) -> i8 {
    let me = root_state.current_player();
    let mut child = root_state.clone();
    let coord = unpack_coord(action_id);
    match apply_placement(&mut child, Placement { coord }) {
        Err(_) => 0,
        Ok(res) => {
            if let Some(outcome) = res.outcome {
                return if outcome.winner == me { 1 } else { -1 };
            }
            match threats::analyze(&child).verdict() {
                Some(v) => {
                    let ours = if child.current_player() == me { v } else { -v };
                    if ours > 0.5 {
                        1
                    } else if ours < -0.5 {
                        -1
                    } else {
                        0
                    }
                }
                None => 0,
            }
        }
    }
}

fn tactical_guard_weights(root_state: &RustHexoState, action_ids: &[PackedCoord], weights: &[f32]) -> Vec<f32> {
    let analysis = threats::analyze(root_state);
    if !analysis.own_win_now && analysis.opp_threat_count == 0 {
        return weights.to_vec();
    }
    let classes: Vec<i8> = action_ids.iter().map(|&id| classify_root_move(root_state, id)).collect();
    let mut guarded = weights.to_vec();
    if classes.iter().any(|&c| c == 1) {
        for (i, &c) in classes.iter().enumerate() {
            if c != 1 {
                guarded[i] = 0.0;
            }
        }
    } else if classes.iter().any(|&c| c != -1) {
        for (i, &c) in classes.iter().enumerate() {
            if c == -1 {
                guarded[i] = 0.0;
            }
        }
    }
    if guarded.iter().all(|&w| w <= 0.0) {
        return weights.to_vec();
    }
    guarded
}

fn select_search_action(
    search: &RustSearch,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    temperature: f32,
    seed: u64,
) -> PyResult<Option<PackedCoord>> {
    let (action_ids, weights, _q, _total) = visit_policy(search.root(), baseline);
    let guarded = if search.tss_enabled {
        tactical_guard_weights(&search.root_state, &action_ids, &weights)
    } else {
        weights.clone()
    };
    let (selected, _override) = select_action_with_lcb(search, baseline, &action_ids, &guarded, temperature, seed)?;
    Ok(selected)
}

/// Temperature sampling when temperature > 0; at temperature 0 with `lcb_move_selection`, LCB-of-Q among
/// guard-positive children (fallback max-visits).
fn select_action_with_lcb(
    search: &RustSearch,
    baseline: Option<&HashMap<PackedCoord, u32>>,
    action_ids: &[PackedCoord],
    guarded_weights: &[f32],
    temperature: f32,
    seed: u64,
) -> PyResult<(Option<PackedCoord>, bool)> {
    let dv = search.divergences;
    if temperature == 0.0 && dv.lcb_move_selection {
        let visit_pick = select_action_from_policy(action_ids, guarded_weights, 0.0, seed)?;
        let root = search.root();
        if let Some(lcb_id) = lcb_pick(root, baseline, &dv) {
            let allowed = action_ids.iter().zip(guarded_weights.iter()).any(|(&id, &w)| id == lcb_id && w > 0.0);
            if allowed {
                let final_id = if dv.ml_final_pick && dv.moves_left_utility {
                    ml_final_pick(root, baseline, &dv, action_ids, guarded_weights).unwrap_or(lcb_id)
                } else {
                    lcb_id
                };
                let overrode = visit_pick.map(|v| v != final_id).unwrap_or(false);
                return Ok((Some(final_id), overrode));
            }
        }
        return Ok((visit_pick, false));
    }
    Ok((select_action_from_policy(action_ids, guarded_weights, temperature, seed)?, false))
}

fn visit_policy(
    root: &RustNode,
    baseline: Option<&HashMap<PackedCoord, u32>>,
) -> (Vec<PackedCoord>, Vec<f32>, Vec<f32>, u32) {
    let deltas: Vec<u32> = root.edges.iter().map(|edge| edge_delta_visits(edge, baseline)).collect();
    let policy_total: u32 = deltas.iter().copied().sum();
    let mut policy_action_ids = Vec::with_capacity(root.edges.len());
    let mut policy_weights = Vec::with_capacity(root.edges.len());
    let mut policy_q = Vec::with_capacity(root.edges.len());
    for (edge, &visits) in root.edges.iter().zip(deltas.iter()) {
        if baseline.is_some() && visits == 0 {
            continue;
        }
        let weight = if policy_total > 0 { visits as f32 / policy_total as f32 } else { edge.prior };
        policy_action_ids.push(edge.action_id);
        policy_weights.push(weight);
        policy_q.push(edge.value());
    }
    (policy_action_ids, policy_weights, policy_q, policy_total)
}

fn edge_delta_visits(edge: &RustEdge, baseline: Option<&HashMap<PackedCoord, u32>>) -> u32 {
    let before = baseline.and_then(|visits| visits.get(&edge.action_id).copied()).unwrap_or(0);
    edge.visits.saturating_sub(before)
}

fn select_action_from_policy(
    action_ids: &[PackedCoord],
    weights: &[f32],
    temperature: f32,
    seed: u64,
) -> PyResult<Option<PackedCoord>> {
    if action_ids.is_empty() || weights.is_empty() {
        return Ok(None);
    }
    if action_ids.len() != weights.len() {
        return Err(PyValueError::new_err("visit policy action and weight lengths differ"));
    }
    let total_weight: f32 = weights.iter().copied().sum();
    for weight in weights {
        if !weight.is_finite() || *weight < 0.0 {
            return Err(PyValueError::new_err(format!(
                "visit policy weights must be finite and >= 0, got {weight}"
            )));
        }
    }
    if total_weight <= 0.0 {
        return Err(PyValueError::new_err("visit policy must contain positive weight mass"));
    }
    if temperature == 0.0 {
        return Ok(action_ids
            .iter()
            .copied()
            .zip(weights.iter().copied())
            .max_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| right.0.cmp(&left.0))
            })
            .map(|(action_id, _)| action_id));
    }
    let inv_temperature = 1.0 / temperature;
    let mut total = 0.0f64;
    let mut adjusted = Vec::with_capacity(weights.len());
    for weight in weights {
        let value = weight.powf(inv_temperature) as f64;
        total += value;
        adjusted.push(value);
    }
    if total <= 0.0 || !total.is_finite() {
        return Err(PyValueError::new_err(
            "temperature-adjusted visit policy must contain positive finite mass",
        ));
    }
    let mut threshold = random_unit(seed) * total;
    let mut last_positive: Option<PackedCoord> = None;
    for (action_id, weight) in action_ids.iter().copied().zip(adjusted) {
        if weight <= 0.0 {
            continue;
        }
        last_positive = Some(action_id);
        threshold -= weight;
        if threshold <= 0.0 {
            return Ok(Some(action_id));
        }
    }
    Ok(last_positive)
}
