use std::io::{self, BufRead, Write};
use hexo_engine::{GameConfig, GameState};
use hexo_engine::types::Player;
use hexo_infer::InferModel;
use hexo_rs::mcts::{MCTSConfig, gumbel_mcts::gumbel_mcts};
use rand::SeedableRng;
use serde::Deserialize;
use serde_json::{json, Value};

const REVISION: &str = "5a771e572553a8bd8e010112b2ce65f16e5afa1b";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    stones: Vec<(i32,i32,u8)>, player: u8, remaining: u8,
    simulations: u32, actions: usize, seed: u64,
}
fn player(p: u8) -> Result<Player,String> {
    match p {0=>Ok(Player::P1),1=>Ok(Player::P2),_=>Err("invalid player".into())}
}
fn search(model: &InferModel, req: Request) -> Result<Value,String> {
    if !(1..=2).contains(&req.remaining) || !(1..=100000).contains(&req.simulations)
        || !(1..=1024).contains(&req.actions) || req.stones.len()>800 {
        return Err("invalid phase/search budget/stone count".into());
    }
    let mut cells=Vec::new();
    let mut seen=std::collections::HashSet::new();
    for (q,r,p) in req.stones {
        if q.unsigned_abs()>1000000 || r.unsigned_abs()>1000000 || !seen.insert((q,r)) {
            return Err("invalid or duplicate coordinates".into());
        }
        cells.push(((q,r),player(p)?));
    }
    if !cells.contains(&((0,0),Player::P1)) {return Err("P1 origin required".into());}
    let side=player(req.player)?;
    let mut game=GameState::from_state(&cells,side,req.remaining,
        GameConfig{win_length:6,placement_radius:8,max_moves:u32::MAX});
    if game.has_winner().is_some() {return Err("terminal input".into());}
    let config=MCTSConfig{n_simulations:req.simulations,m_actions:req.actions,c_visit:50,
        c_scale:1.0,disable_gumbel_noise:true,..Default::default()};
    let mut rng=rand_chacha::ChaCha8Rng::seed_from_u64(req.seed);
    let start=std::time::Instant::now();
    let mut moves=Vec::new();
    let mut eval_calls=0u64;
    let mut eval_states=0u64;
    let mut finite=true;
    let mut root_visits=Vec::new();
    for _ in 0..req.remaining {
        let mut evaluator=|states:&[GameState]| {
            eval_calls+=1; eval_states+=states.len() as u64;
            // This path constructs the complete relational graph directly.
            let result=model.eval_states(states);
            finite &= result.1.iter().all(|v|v.is_finite())
                && result.0.iter().all(|row|row.values().all(|v|v.is_finite()));
            result
        };
        let result=gumbel_mcts(&game,&config,&mut rng,None,&mut evaluator).map_err(|e|e.to_string())?;
        if !finite {return Err("nonfinite network output".into());}
        root_visits.push(result.visit_counts.iter().map(|n|*n as u64).sum::<u64>());
        game.apply_move(result.action).map_err(|e|format!("invalid engine move: {e:?}"))?;
        moves.push(result.action);
        if game.is_terminal() {break;}
    }
    if !game.is_terminal() && game.current_player()==Some(side) {return Err("incomplete turn".into());}
    Ok(json!({"status":"OK","moves":moves,"search_ms":start.elapsed().as_secs_f64()*1000.0,
        "eval_calls":eval_calls,"eval_states":eval_states,"root_visits":root_visits,
        "simulations_per_placement":req.simulations,"revision":REVISION}))
}
fn main() {
    let mut model:Option<InferModel>=None;
    let stdin=io::stdin();
    let mut stdout=io::stdout().lock();
    for line in stdin.lock().lines() {
        let result:Result<Value,String>=(|| {
            let value:Value=serde_json::from_str(&line.map_err(|e|e.to_string())?).map_err(|e|e.to_string())?;
            if let Some(path)=value.get("load").and_then(Value::as_str) {
                if model.is_some() {return Err("model already loaded".into());}
                let bytes=std::fs::read(path).map_err(|e|e.to_string())?;
                let loaded=InferModel::from_safetensors(&bytes).map_err(|e|e.to_string())?;
                let response=json!({"status":"READY","revision":REVISION,
                    "source_checkpoint":loaded.source_checkpoint(),
                    "metadata":serde_json::from_str::<Value>(&loaded.config().metadata_json).map_err(|e|e.to_string())?});
                model=Some(loaded);
                return Ok(response);
            }
            search(model.as_ref().ok_or("load model first")?,serde_json::from_value(value).map_err(|e|e.to_string())?)
        })();
        let response=match result {Ok(value)=>value,Err(error)=>json!({"status":"UNKNOWN","error":error,"revision":REVISION})};
        if writeln!(stdout,"{}",response).and_then(|_|stdout.flush()).is_err() {break;}
    }
}
