//! Strix (SootyOwl/hexo-strix) for the browser: the learned network and Gumbel search of
//! `tools/strix_learned`, as a wasm32-wasip1 library driven by `web/engine/strix/strix.mjs`.
//!
//! Exports: `strix_alloc(len)` and `strix_free(ptr, len)` for request bytes; `strix_load(ptr, len)`
//! reads safetensors weights; `strix_turn(ptr, len)` reads the request JSON of `tools/strix_learned`
//! (`stones`, `player`, `remaining`, `simulations`, `actions`, `seed`, Strix's frame). Both return a
//! NUL-terminated JSON string that the caller releases with `strix_free_text`. A turn searches each
//! placement exactly as `tools/strix_learned/src/main.rs` does and adds, for the first placement,
//! `top` (up to five `[q, r, improved policy, mover win probability]`) and `value` (the mover's win
//! probability under the improved policy). `strix_value(ptr, len)` takes the same request and returns
//! `{"status":"OK","value"}`, the network's win probability for the mover without search.
//! `hexo_strix.progress(fraction)` is called after every network batch.
use std::ffi::{CString, c_char};
use std::sync::Mutex;
use hexo_engine::{GameConfig, GameState};
use hexo_engine::types::Player;
use hexo_infer::InferModel;
use hexo_rs::mcts::{MCTSConfig, gumbel_mcts::gumbel_mcts};
use rand::SeedableRng;
use serde::Deserialize;
use serde_json::{json, Value};

const REVISION: &str = "5a771e572553a8bd8e010112b2ce65f16e5afa1b";
static MODEL: Mutex<Option<InferModel>> = Mutex::new(None);

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "hexo_strix")]
unsafe extern "C" {
    #[link_name = "progress"]
    fn report(fraction: f64);
}
#[cfg(not(target_arch = "wasm32"))]
unsafe fn report(_fraction: f64) {}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    stones: Vec<(i32,i32,u8)>, player: u8, remaining: u8,
    simulations: u32, actions: usize, seed: u64,
}
fn player(p: u8) -> Result<Player,String> {
    match p {0=>Ok(Player::P1),1=>Ok(Player::P2),_=>Err("invalid player".into())}
}
/// The position of `req` after the checks of `tools/strix_learned`, and the side to move.
fn position(req: &Request) -> Result<(GameState,Player),String> {
    if !(1..=2).contains(&req.remaining) || !(1..=100000).contains(&req.simulations)
        || !(1..=1024).contains(&req.actions) || req.stones.len()>800 {
        return Err("invalid phase/search budget/stone count".into());
    }
    let mut cells=Vec::new();
    let mut seen=std::collections::HashSet::new();
    for &(q,r,p) in &req.stones {
        if q.unsigned_abs()>1000000 || r.unsigned_abs()>1000000 || !seen.insert((q,r)) {
            return Err("invalid or duplicate coordinates".into());
        }
        cells.push(((q,r),player(p)?));
    }
    if !cells.contains(&((0,0),Player::P1)) {return Err("P1 origin required".into());}
    let side=player(req.player)?;
    let game=GameState::from_state(&cells,side,req.remaining,
        GameConfig{win_length:6,placement_radius:8,max_moves:u32::MAX});
    if game.has_winner().is_some() {return Err("terminal input".into());}
    Ok((game,side))
}
fn value(model: &InferModel, req: Request) -> Result<Value,String> {
    let (game,_)=position(&req)?;
    let (_,values)=model.eval_states(std::slice::from_ref(&game));
    Ok(json!({"status":"OK","value":(values[0]+1.0)/2.0,"revision":REVISION}))
}
fn search(model: &InferModel, req: Request) -> Result<Value,String> {
    let (mut game,side)=position(&req)?;
    let config=MCTSConfig{n_simulations:req.simulations,m_actions:req.actions,c_visit:50,
        c_scale:1.0,disable_gumbel_noise:true,..Default::default()};
    let mut rng=rand_chacha::ChaCha8Rng::seed_from_u64(req.seed);
    let mut moves=Vec::new();
    let mut eval_calls=0u64;
    let mut eval_states=0u64;
    let mut finite=true;
    let mut root_visits=Vec::new();
    let mut top=Vec::new();
    let mut value=0.0;
    let placements=f64::from(req.remaining);
    for placement in 0..req.remaining {
        let mut searched=0u64;
        let mut evaluator=|states:&[GameState]| {
            eval_calls+=1; eval_states+=states.len() as u64; searched+=states.len() as u64;
            let result=model.eval_states(states);
            finite &= result.1.iter().all(|v|v.is_finite())
                && result.0.iter().all(|row|row.values().all(|v|v.is_finite()));
            let done=(searched as f64/f64::from(req.simulations+1)).min(1.0);
            unsafe {report((f64::from(placement)+done)/placements)};
            result
        };
        let result=gumbel_mcts(&game,&config,&mut rng,None,&mut evaluator).map_err(|e|e.to_string())?;
        if !finite {return Err("nonfinite network output".into());}
        root_visits.push(result.visit_counts.iter().map(|n|*n as u64).sum::<u64>());
        if placement==0 {
            let policy=&result.improved_policy;
            value=(policy.iter().zip(&result.per_child_q).map(|(p,q)|p*q).sum::<f64>()+1.0)/2.0;
            let mut order:Vec<usize>=(0..policy.len()).collect();
            order.sort_by(|&a,&b|policy[b].total_cmp(&policy[a]).then(result.coords[a].cmp(&result.coords[b])));
            top=order.into_iter().take(5).map(|i| {
                let (q,r)=result.coords[i];
                json!([q,r,policy[i],(result.per_child_q[i]+1.0)/2.0])
            }).collect();
        }
        game.apply_move(result.action).map_err(|e|format!("invalid engine move: {e:?}"))?;
        moves.push(result.action);
        if game.is_terminal() {break;}
    }
    if !game.is_terminal() && game.current_player()==Some(side) {return Err("incomplete turn".into());}
    Ok(json!({"status":"OK","moves":moves,"top":top,"value":value,
        "eval_calls":eval_calls,"eval_states":eval_states,"root_visits":root_visits,
        "simulations_per_placement":req.simulations,"revision":REVISION}))
}

fn text(value: Value) -> *mut c_char {
    CString::new(value.to_string()).expect("JSON has no NUL").into_raw()
}
fn failure(error: String) -> Value {
    json!({"status":"UNKNOWN","error":error,"revision":REVISION})
}

/// `len` writable bytes for a request; release them with `strix_free(ptr, len)`.
#[unsafe(no_mangle)]
pub extern "C" fn strix_alloc(len: usize) -> *mut u8 {
    let mut buffer=Vec::<u8>::with_capacity(len);
    let pointer=buffer.as_mut_ptr();
    std::mem::forget(buffer);
    pointer
}
/// Releases a buffer from `strix_alloc(len)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn strix_free(pointer: *mut u8, len: usize) {
    drop(unsafe {Vec::from_raw_parts(pointer,0,len)});
}
/// Releases a string returned by `strix_load` or `strix_turn`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn strix_free_text(pointer: *mut c_char) {
    drop(unsafe {CString::from_raw(pointer)});
}
/// Loads the safetensors weights in `bytes[..len]`, replacing any loaded model:
/// `{"status":"READY", source_checkpoint, metadata}` or `{"status":"UNKNOWN", error}`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn strix_load(bytes: *const u8, len: usize) -> *mut c_char {
    let bytes=unsafe {std::slice::from_raw_parts(bytes,len)};
    text(match InferModel::from_safetensors(bytes) {
        Ok(model) => {
            let metadata=serde_json::from_str::<Value>(&model.config().metadata_json).unwrap_or(Value::Null);
            let response=json!({"status":"READY","revision":REVISION,
                "source_checkpoint":model.source_checkpoint(),"metadata":metadata});
            *MODEL.lock().unwrap()=Some(model);
            response
        }
        Err(error) => failure(error.to_string()),
    })
}
fn answer(bytes: *const u8, len: usize, run: fn(&InferModel,Request)->Result<Value,String>) -> *mut c_char {
    let bytes=unsafe {std::slice::from_raw_parts(bytes,len)};
    let guard=MODEL.lock().unwrap();
    let result=match guard.as_ref() {
        None => Err("load model first".to_string()),
        Some(model) => serde_json::from_slice(bytes).map_err(|e|e.to_string()).and_then(|req|run(model,req)),
    };
    text(result.unwrap_or_else(failure))
}
/// One turn for the request JSON in `bytes[..len]` with the loaded model.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn strix_turn(bytes: *const u8, len: usize) -> *mut c_char {
    answer(bytes,len,search)
}
/// The network's value for the request JSON in `bytes[..len]`, without search.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn strix_value(bytes: *const u8, len: usize) -> *mut c_char {
    answer(bytes,len,value)
}
