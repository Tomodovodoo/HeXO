mod check;
use std::collections::BTreeMap;
use std::ffi::{CStr,CString,c_char};
use std::sync::{Arc,Mutex,OnceLock};
use std::sync::{mpsc,atomic::{AtomicBool,Ordering}};
use std::time::{Duration,Instant};
use hexo_engine::types::Player;
use hexo_solver::forcing::Meter;
use hexo_solver::prover::{self,Ctl,DriverKind,ProverConfig};
use hexo_solver::prover::io::{Position,PosConfig};
use hexo_solver::prover::certificate::{ProofCertificate,ProofNode,ProofResponse};
use serde::Deserialize;
use serde_json::{json,Value};

const REVISION:&str="5a771e572553a8bd8e010112b2ce65f16e5afa1b";
/// PDS-PN table size: one MiB per this many budgeted nodes, clamped to 1..=16 MiB.
const NODES_PER_TT_MB:u64=2048;
// Exact state keys, never Zobrist hashes. Rules/scope are fixed by this library version.
// The budgets (and the IDTT depth when IDTT runs) are part of the key: a search is a
// function of position and budgets. A hit replays the search's certificate, work and
// IDTT verdict, so it is indistinguishable from a fresh search except for `cache_hit`.
type Key=(Vec<((i32,i32),u8)>,u8,u8,u64,u64,u8);
type Solved=(ProofCertificate,u64,Option<String>);
static CACHE:OnceLock<Mutex<BTreeMap<Key,Solved>>>=OnceLock::new();
/// Whose forced win is asked: the side to move, or its opponent given a fresh
/// two-placement turn on the current stones (a flipped-turn threat query).
#[derive(Deserialize,Clone,Copy,PartialEq,Default)]
#[serde(rename_all="lowercase")]
enum Attacker {#[default] Mover, Opponent}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    history:Vec<(i32,i32)>, ms:u32, nodes:u64, idtt_nodes:u64, depth:u8,
    #[serde(default)] attacker:Attacker,
    #[serde(default)] certificate:Option<ProofCertificate>,
    #[serde(default)] root_moves:Option<Vec<(i32,i32)>>,
}
fn position(board:&check::Board,side:u8,remaining:u8)->Position {
    Position{stones:board.iter().map(|(&p,&s)|(p,if s==0{Player::P1}else{Player::P2})).collect(),
        attacker:if side==0{Player::P1}else{Player::P2},placements_remaining:remaining,
        config:PosConfig{win_length:6,placement_radius:8,max_moves:u32::MAX}}
}
fn complete_candidate(board:&check::Board,start:usize,moves:&[(i32,i32)],req:&Request,ctl:&Ctl,meter:&Meter)->Result<ProofCertificate,String> {
    let side=check::phase(start).0;
    let (post,ply,terminal)=check::apply(board,start,moves)?;
    if terminal {return Ok(ProofCertificate{version:1,width:"wide".into(),root:0,nodes:vec![ProofNode::ImmediateWin{action:moves.to_vec()}]});}
    let defenses=check::defenses(&post,side,ctl.deadline.ok_or("candidate needs a deadline")?)?;
    let mut cert=ProofCertificate{version:1,width:"wide".into(),root:0,nodes:vec![
        ProofNode::AttackerMove{action:moves.to_vec(),child:1,alternatives:vec![]},
        ProofNode::Unstoppable{threats:vec![]} ]};
    if defenses.is_empty(){return Ok(cert);}
    let mut responses=Vec::new();
    for action in defenses.into_values() {
        if ctl.expired() || meter.spent()>=req.nodes {return Err("candidate defense expansion budget".into());}
        let (next,n,won)=check::apply(&post,ply,&action)?;
        if won {return Err("defender counterwin".into());}
        let cfg=ProverConfig{driver:DriverKind::Pdspn,wide:true,node_budget:req.nodes,tt_mb:1,pn2_nodes:1000,..Default::default()};
        let child=prover::pdspn::solve(&position(&next,side,check::phase(n).1),&cfg,ctl);
        let mut child=child.certificate.ok_or("candidate has unproved defender continuation")?;
        if cert.nodes.len()+child.nodes.len()>50000 {return Err("candidate certificate size limit".into());}
        let offset=cert.nodes.len() as u32;
        responses.push(ProofResponse{action,child:offset+child.root});
        for node in &mut child.nodes {
            match node {
                ProofNode::AttackerMove{child,alternatives,..}=>{
                    *child+=offset;for alternative in alternatives {alternative.child+=offset;}
                }
                ProofNode::DefenderReplies{responses}=>{for response in responses {response.child+=offset;}}
                _=>{}
            }
        }
        cert.nodes.extend(child.nodes);
    }
    cert.nodes[1]=ProofNode::DefenderReplies{responses};Ok(cert)
}
/// One query. `nodes` bounds the total search work (IDTT nodes plus PDS-PN level-1
/// nodes and level-2 expansions), so the verdict and certificate are a function of
/// (position, attacker, nodes, idtt_nodes, build); `ms` is only a safety cap, and a
/// query that reaches it returns UNKNOWN.
fn run(req:Request, start:Instant) -> Result<Value,String> {
    if req.history.len()>800 || req.ms==0 || req.ms>60000 || req.nodes==0 || req.nodes>10_000_000
        || req.idtt_nodes>=req.nodes || req.depth==0 || req.depth>64 {return Err("invalid tactical limits".into());}
    let deadline=start+Duration::from_millis(req.ms as u64);
    let board=check::replay(&req.history)?;
    let ply=if req.attacker==Attacker::Opponent {check::flip(req.history.len())} else {req.history.len()};
    let (side,remaining)=check::phase(ply);
    let key=(board.iter().map(|(&p,&s)|(p,s)).collect(),side,remaining,req.nodes,req.idtt_nodes,
        if req.idtt_nodes>0 {req.depth} else {0});
    let scope=json!({"rules":{"win_length":6,"placement_radius":8,"match_move_cap":null},
        "defenses":"all legal two-stone covers including complete free-second frontier; quiet defender nodes unsupported",
        "attacks":"wide Strix proposals plus optional root candidate; selective negatives remain UNKNOWN","checker_version":3,
        "budget":{"nodes":req.nodes,"idtt_nodes":req.idtt_nodes,"idtt_depth_cap":req.depth,"safety_ms":req.ms,
            "work":"one shared meter over IDTT nodes, PDS-PN level-1 nodes and level-2 expansions; verifier path limit 128"}});
    let meter=Meter::new(req.nodes);
    let ctl=Ctl{deadline:Some(deadline),cancel:Arc::new(AtomicBool::new(false)),meter:Some(meter.clone())};
    let cache=CACHE.get_or_init(||Mutex::new(BTreeMap::new()));
    let mut cache_hit=false;
    let mut probe_verdict=None;
    let mut cached_nodes=None;
    let cert=if let Some(cert)=req.certificate.clone() {Some(cert)} else if let Some(moves)=&req.root_moves {
        Some(complete_candidate(&board,ply,moves,&req,&ctl,&meter)?)
    } else {
        let saved=cache.lock().map_err(|_|"cache lock")?.get(&key).cloned();
        if let Some((cert,used,verdict))=saved {
            cache_hit=true;cached_nodes=Some(used);probe_verdict=verdict;Some(cert)
        } else {
            let pos=position(&board,side,remaining);
            let cfg=ProverConfig{driver:DriverKind::Pdspn,wide:true,depth_cap:req.depth,
                node_budget:req.nodes,tt_mb:(req.nodes/NODES_PER_TT_MB).clamp(1,16) as usize,pn2_nodes:1000,..Default::default()};
            if req.idtt_nodes>0 {
                // A dedicated meter bounds the whole probe, PV reconstruction included.
                let share=Meter::new(req.idtt_nodes);
                let probe=prover::idtt(&pos,&ProverConfig{node_budget:req.idtt_nodes,..cfg.clone()},
                    &Ctl{meter:Some(share.clone()),..ctl.clone()});
                meter.add(share.spent().min(req.idtt_nodes));
                probe_verdict=Some(format!("{:?}",probe.verdict));
            }
            // Only PDS-PN emits an all-defense DAG. An IDTT PV is never enough.
            if ctl.expired() {None} else {prover::pdspn::solve(&pos,&cfg,&ctl).certificate}
        }
    };
    let mut response=json!({"status":"UNKNOWN","native_verified":false,"moves":[],"certificate":null,
        "revision":REVISION,"scope":scope,"cache_hit":cache_hit,"idtt_verdict":probe_verdict.clone(),
        "attacker":if req.attacker==Attacker::Opponent {"opponent"} else {"mover"},
        "reason":"no verified strategy","nodes_used":cached_nodes.unwrap_or(meter.spent().min(req.nodes)),"proof_turns":null,"elapsed_ms":0.0});
    if let Some(cert)=cert {
        match check::verify(&req.history,ply,&cert,deadline,50000) {
            Ok((moves,turns))=>{
                if !cache_hit && req.certificate.is_none() && req.root_moves.is_none() {
                    let mut guard=cache.lock().map_err(|_|"cache lock")?;
                    if guard.len()>=128 {guard.clear();}
                    guard.insert(key,(cert.clone(),meter.spent().min(req.nodes),probe_verdict));
                }
                response["status"]=json!("PROVEN_WIN");response["native_verified"]=json!(true);
                response["moves"]=json!(moves);response["proof_turns"]=json!(turns);
                response["certificate"]=serde_json::to_value(cert).map_err(|e|e.to_string())?;
                response["reason"]=json!("independent raw-board strategy verification");
            }
            Err(reason)=>response["reason"]=json!(reason),
        }
    }
    // Certificate reconstruction/serialization and verification share the safety cap.
    if Instant::now()>=deadline {
        response["status"]=json!("UNKNOWN");response["native_verified"]=json!(false);
        response["moves"]=json!([]);response["certificate"]=Value::Null;response["proof_turns"]=Value::Null;
        response["reason"]=json!("deadline");
    }
    response["elapsed_ms"]=json!(start.elapsed().as_secs_f64()*1000.0);
    Ok(response)
}

// Upstream certificate reconstruction has no cancellation hook. A single native
// worker contains that overrun; callers stop waiting at their absolute deadline.
// No queue of abandoned work and no per-request process/thread creation.
type Work=(Request,Instant,mpsc::Sender<Result<Value,String>>);
static WORKER:OnceLock<mpsc::SyncSender<Work>>=OnceLock::new();
static BUSY:AtomicBool=AtomicBool::new(false);
static LAST_WORK:OnceLock<Mutex<Value>>=OnceLock::new();
#[cfg(windows)]
fn thread_cpu_ms()->Option<f64> {
    #[repr(C)] struct FileTime {low:u32,high:u32}
    #[link(name="kernel32")]
    unsafe extern "system" {
        fn GetCurrentThread()->*mut std::ffi::c_void;
        fn GetThreadTimes(thread:*mut std::ffi::c_void,create:*mut FileTime,exit:*mut FileTime,kernel:*mut FileTime,user:*mut FileTime)->i32;
    }
    let mut c=FileTime{low:0,high:0};let mut e=FileTime{low:0,high:0};
    let mut k=FileTime{low:0,high:0};let mut u=FileTime{low:0,high:0};
    if unsafe{GetThreadTimes(GetCurrentThread(),&mut c,&mut e,&mut k,&mut u)}==0 {return None;}
    Some((((k.high as u64)<<32|k.low as u64)+((u.high as u64)<<32|u.low as u64)) as f64/10000.0)
}
#[cfg(not(windows))]
fn thread_cpu_ms()->Option<f64> {None}
fn dispatch(req:Request,start:Instant)->Result<Value,String> {
    if req.ms==0 || req.ms>60000 {return Err("invalid deadline".into());}
    let deadline=start+Duration::from_millis(req.ms as u64);
    let worker=WORKER.get_or_init(|| {
        let (sender,receiver)=mpsc::sync_channel::<Work>(1);
        std::thread::spawn(move || {
            while let Ok((req,start,reply))=receiver.recv() {
                let budget_ms=req.ms;let cpu_start=thread_cpu_ms();
                let result=std::panic::catch_unwind(||run(req,start))
                    .unwrap_or_else(|_|Err("native worker panic".into()));
                let elapsed=start.elapsed().as_secs_f64()*1000.0;
                let cpu=thread_cpu_ms().zip(cpu_start).map(|(a,b)|a-b);
                if let Ok(mut stats)=LAST_WORK.get_or_init(||Mutex::new(Value::Null)).lock() {
                    let late=elapsed>=budget_ms as f64;
                    let previous=stats.clone();
                    let row=json!({"elapsed_ms":elapsed,"thread_cpu_ms":cpu,
                        "requested_ms":budget_ms,"completed_after_deadline":late});
                    *stats=json!({"latest":row,"completed_queries":previous["completed_queries"].as_u64().unwrap_or(0)+1,
                        "completed_after_deadline_count":previous["completed_after_deadline_count"].as_u64().unwrap_or(0)+u64::from(late),
                        "total_worker_elapsed_ms":previous["total_worker_elapsed_ms"].as_f64().unwrap_or(0.0)+elapsed,
                        "total_thread_cpu_ms":cpu.map(|c|previous["total_thread_cpu_ms"].as_f64().unwrap_or(0.0)+c),
                        "last_after_deadline":if late {row}else{previous["last_after_deadline"].clone()}});
                }
                BUSY.store(false,Ordering::Release);
                let _=reply.send(result);
            }
        });
        sender
    });
    if BUSY.compare_exchange(false,true,Ordering::AcqRel,Ordering::Acquire).is_err() {
        return Err("native worker busy finishing bounded prior query".into());
    }
    let (send,recv)=mpsc::channel();
    if worker.send((req,start,send)).is_err() {BUSY.store(false,Ordering::Release);return Err("native worker stopped".into());}
    recv.recv_timeout(deadline.saturating_duration_since(Instant::now()))
        .map_err(|_|"deadline; prior native work may still be finishing".to_string())?
}

/// One serialized native query at a time. Python holds a lock around calls.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hexo_tactical_query(input:*const c_char)->*mut c_char {
    let start=Instant::now();
    let result=std::panic::catch_unwind(|| {
        if input.is_null(){return Err("null request".to_string());}
        let bytes=unsafe{CStr::from_ptr(input)}.to_bytes();
        if bytes.len()>8*1024*1024{return Err("request size limit".into());}
        serde_json::from_slice(bytes).map_err(|e|e.to_string()).and_then(|req|dispatch(req,start))
    });
    let mut value=match result {
        Ok(Ok(v))=>v,
        Ok(Err(reason))=>json!({"status":"UNKNOWN","native_verified":false,"reason":reason,"moves":[]}),
        Err(_)=>json!({"status":"UNKNOWN","native_verified":false,"reason":"native panic","moves":[]}),
    };
    value["background_worker_busy"]=json!(BUSY.load(Ordering::Acquire));
    value["last_worker_completion"]=LAST_WORK.get_or_init(||Mutex::new(Value::Null))
        .lock().map(|stats|stats.clone()).unwrap_or(Value::Null);
    value["equal_compute_clock"]=json!(false);
    CString::new(value.to_string()).unwrap().into_raw()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hexo_tactical_free(value:*mut c_char) {
    if !value.is_null(){drop(unsafe{CString::from_raw(value)});}
}

#[cfg(test)]
mod tests {
    use super::*;
    const OPEN_THREE:[(i32,i32);7]=[(0,0),(0,8),(2,8),(1,0),(2,0),(4,8),(6,8)];
    fn setup(limit:u64)->(Position,ProverConfig,Ctl,Meter) {
        let board=check::replay(&OPEN_THREE).unwrap();
        let (side,remaining)=check::phase(OPEN_THREE.len());
        let meter=Meter::new(limit);
        let ctl=Ctl{deadline:Some(Instant::now()+Duration::from_secs(60)),cancel:Arc::new(AtomicBool::new(false)),meter:Some(meter.clone())};
        let cfg=ProverConfig{driver:DriverKind::Pdspn,wide:true,depth_cap:8,node_budget:limit,tt_mb:1,pn2_nodes:1000,..Default::default()};
        (position(&board,side,remaining),cfg,ctl,meter)
    }
    #[test]
    fn meter_charges_level_two_and_idtt_work() {
        let (pos,cfg,ctl,meter)=setup(1_000_000);
        let solved=prover::pdspn::solve(&pos,&cfg,&ctl);
        assert!(solved.certificate.is_some());
        assert!(meter.spent()>solved.stats.nodes,"level-2 expansions are charged");
        let (pos,cfg,ctl,meter)=setup(1_000_000);
        prover::idtt(&pos,&cfg,&ctl);
        assert!(meter.spent()>0,"IDTT nodes are charged");
    }
    #[test]
    fn idtt_probe_spends_only_its_share() {
        let query=|idtt_nodes:u64| {
            let req=serde_json::from_value(json!({"history":OPEN_THREE,"ms":60000,"nodes":1_000_000,
                "idtt_nodes":idtt_nodes,"depth":8})).unwrap();
            run(req,Instant::now()).unwrap()
        };
        let (plain,probed)=(query(0),query(3));
        assert_eq!(plain["certificate"],probed["certificate"]);
        let extra=probed["nodes_used"].as_u64().unwrap()-plain["nodes_used"].as_u64().unwrap();
        assert!(extra<=3,"IDTT spent {extra} units of a 3-unit share");
    }
    #[test]
    fn exhausted_meter_stops_the_search() {
        for limit in [1,5,20] {
            let (pos,cfg,ctl,meter)=setup(limit);
            prover::pdspn::solve(&pos,&cfg,&ctl);
            assert!(meter.spent()<=limit+2,"{} units charged against {limit}",meter.spent());
        }
    }
}
