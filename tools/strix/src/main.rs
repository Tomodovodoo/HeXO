use std::io::{self, BufRead, Write};
use hexo_engine::types::Player;
use hexo_solver::{SolverPosition, SolverEngine, solve_from_position_with_stats,
                  solve_wide_from_position_with_stats};
use hexo_solver::forcing::Outcome;
use serde::Deserialize;
use serde_json::{json, Value};

const REVISION: &str = "5a771e572553a8bd8e010112b2ce65f16e5afa1b";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    stones: Vec<(i32, i32, String)>,
    attacker: String,
    placements_remaining: u8,
    depth: u8,
    nodes: u64,
    wide: bool,
}
fn player(s: &str) -> Result<Player, String> {
    match s { "P1" => Ok(Player::P1), "P2" => Ok(Player::P2),
              _ => Err("player must be P1 or P2".into()) }
}
fn solve(request: Request) -> Result<Value, String> {
    if !(1..=2).contains(&request.placements_remaining) || request.depth == 0 {
        return Err("invalid phase/depth".into());
    }
    let mut seen = std::collections::HashSet::new();
    let mut stones = Vec::new();
    for (q,r,p) in request.stones {
        if !seen.insert((q,r)) { return Err("duplicate coordinate".into()); }
        stones.push(((q,r),player(&p)?));
    }
    let position = SolverPosition { win_length:6, placement_radius:8, max_moves:u32::MAX,
        to_move:player(&request.attacker)?, moves_remaining:request.placements_remaining, stones };
    let start = std::time::Instant::now();
    let result = if request.wide {
        solve_wide_from_position_with_stats(&position,SolverEngine::Idtt,request.depth,request.nodes)
    } else {
        solve_from_position_with_stats(&position,SolverEngine::Idtt,request.depth,request.nodes)
    };
    let (status, depth, first, pv) = match result.outcome {
        Outcome::Win(w) => ("REFERENCE_WIN_WITHIN_SCOPE",Some(w.depth),Some(w.first_move),w.pv),
        Outcome::No => ("NO_FORCING_WIN_WITHIN_SCOPE",None,None,vec![]),
        Outcome::BudgetExceeded => ("UNKNOWN",None,None,vec![]),
    };
    Ok(json!({"status":status,"depth":depth,"first_move":first,"pv":pv,
        "elapsed_s":start.elapsed().as_secs_f64(),"nodes":null,
        "revision":REVISION,"independently_verified_proof":false,
        "scope":{"driver":"idtt","generator":if request.wide {"wide"} else {"tight"},
            "attacker":request.attacker,"placements_remaining":request.placements_remaining,
            "depth_cap":request.depth,"node_budget":request.nodes,
            "depth_convention":"attacker turns including completing turn",
            "rules":{"win_length":6,"placement_radius":8,"match_move_cap":null},
            "domain":"fully forcing attacks consuming the defender's whole turn"}}))
}
fn main() {
    let stdin = io::stdin();
    let mut stdout = io::stdout().lock();
    for line in stdin.lock().lines() {
        let result = match line {
            Ok(line) => serde_json::from_str::<Request>(&line).map_err(|e|e.to_string()).and_then(solve),
            Err(e) => Err(e.to_string()),
        };
        let response = match result { Ok(value)=>value, Err(error)=>json!({"status":"UNKNOWN","error":error}) };
        if writeln!(stdout,"{}",response).and_then(|_|stdout.flush()).is_err() { break; }
    }
}
