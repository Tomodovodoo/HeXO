mod check;
use std::collections::BTreeMap;
use std::ffi::{CStr,CString,c_char};
use std::sync::{Arc,Mutex,OnceLock};
use std::sync::{mpsc,atomic::{AtomicBool,Ordering}};
use std::time::{Duration,Instant};
use hexo_engine::types::Player;
use hexo_solver::prover::{self,Ctl,DriverKind,ProverConfig};
use hexo_solver::prover::io::{Position,PosConfig};
use hexo_solver::prover::certificate::{ProofCertificate,ProofNode,ProofResponse};
use serde::Deserialize;
use serde_json::{json,Value};

const REVISION:&str="5a771e572553a8bd8e010112b2ce65f16e5afa1b";
// Exact state keys, never Zobrist hashes. Rules/scope are fixed by this library version.
type Key=(Vec<((i32,i32),u8)>,u8,u8);
static CACHE:OnceLock<Mutex<BTreeMap<Key,ProofCertificate>>>=OnceLock::new();
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    history:Vec<(i32,i32)>, ms:u32, idtt_ms:u32, nodes:u64, depth:u8,
    #[serde(default)] certificate:Option<ProofCertificate>,
    #[serde(default)] root_moves:Option<Vec<(i32,i32)>>,
}
fn control(deadline:Instant)->Ctl {
    Ctl{deadline:Some(deadline),cancel:Arc::new(AtomicBool::new(false))}
}
fn complete_candidate(req:&Request,board:&check::Board,moves:&[(i32,i32)],deadline:Instant)->Result<ProofCertificate,String> {
    let side=check::phase(req.history.len()).0;
    let (post,ply,terminal)=check::apply(board,req.history.len(),moves)?;
    if terminal {return Ok(ProofCertificate{version:1,width:"wide".into(),root:0,nodes:vec![ProofNode::ImmediateWin{action:moves.to_vec()}]});}
    let defenses=check::defenses(&post,side,deadline)?;
    let mut cert=ProofCertificate{version:1,width:"wide".into(),root:0,nodes:vec![
        ProofNode::AttackerMove{action:moves.to_vec(),child:1,alternatives:vec![]},
        ProofNode::Unstoppable{threats:vec![]} ]};
    if defenses.is_empty(){return Ok(cert);}
    let mut responses=Vec::new();let mut left=req.nodes;
    for action in defenses.into_values() {
        if Instant::now()>=deadline || left==0 {return Err("candidate defense expansion budget".into());}
        let (next,n,won)=check::apply(&post,ply,&action)?;
        if won {return Err("defender counterwin".into());}
        let pos=Position{stones:next.iter().map(|(&p,&s)|(p,if s==0{Player::P1}else{Player::P2})).collect(),
            attacker:if side==0{Player::P1}else{Player::P2},placements_remaining:check::phase(n).1,
            config:PosConfig{win_length:6,placement_radius:8,max_moves:u32::MAX}};
        let cfg=ProverConfig{driver:DriverKind::Pdspn,wide:true,node_budget:left,tt_mb:1,pn2_nodes:1000,..Default::default()};
        let ctl=control(deadline);
        let child=prover::pdspn::solve(&pos,&cfg,&ctl);
        left=left.saturating_sub(child.stats.nodes.max(1));
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
fn run(req:Request, start:Instant) -> Result<Value,String> {
    if req.history.len()>800 || req.ms==0 || req.ms>60000 || req.idtt_ms>=req.ms
        || req.nodes==0 || req.nodes>10_000_000 || req.depth==0 || req.depth>64 {return Err("invalid tactical limits".into());}
    let deadline=start+Duration::from_millis(req.ms as u64);
    let board=check::replay(&req.history)?;
    let (side,remaining)=check::phase(req.history.len());
    let key=(board.iter().map(|(&p,&s)|(p,s)).collect(),side,remaining);
    let scope=json!({"rules":{"win_length":6,"placement_radius":8,"match_move_cap":null},
        "defenses":"all legal two-stone covers including complete free-second frontier; quiet defender nodes unsupported",
        "attacks":"wide Strix proposals plus optional root candidate; selective negatives remain UNKNOWN","checker_version":2,
        "budget":{"requested_ms":req.ms,"idtt_ms":req.idtt_ms,"idtt_depth_cap":req.depth,
            "pds_pn":"unbounded forcing driver with node/time limits; verifier path limit128",
            "nodes_per_driver":req.nodes,"candidate_all_children_total_nodes":req.nodes}});
    let cache=CACHE.get_or_init(||Mutex::new(BTreeMap::new()));
    let mut cache_hit=false;
    let mut probe_verdict=None;
    let cert=if let Some(cert)=req.certificate.clone() {Some(cert)} else if let Some(moves)=&req.root_moves {
        Some(complete_candidate(&req,&board,moves,deadline)?)
    } else {
        let saved=cache.lock().map_err(|_|"cache lock")?.get(&key).cloned();
        if saved.is_some() {cache_hit=true;saved} else {
            let pos=Position{stones:board.iter().map(|(&p,&s)|(p,if s==0{Player::P1}else{Player::P2})).collect(),
                attacker:if side==0{Player::P1}else{Player::P2},placements_remaining:remaining,
                config:PosConfig{win_length:6,placement_radius:8,max_moves:u32::MAX}};
            let cfg=ProverConfig{driver:DriverKind::Pdspn,wide:true,depth_cap:req.depth,
                node_budget:req.nodes,tt_mb:16,pn2_nodes:1000,..Default::default()};
            if req.idtt_ms>0 {
                let ctl=control((Instant::now()+Duration::from_millis(req.idtt_ms as u64)).min(deadline));
                let probe=prover::idtt(&pos,&cfg,&ctl);
                probe_verdict=Some(format!("{:?}",probe.verdict));
            }
            if Instant::now()>=deadline {None} else {
                let ctl=control(deadline);
                // Only PDS-PN emits an all-defense DAG. An IDTT PV is never enough.
                prover::pdspn::solve(&pos,&cfg,&ctl).certificate
            }
        }
    };
    let mut response=json!({"status":"UNKNOWN","native_verified":false,"moves":[],"certificate":null,
        "revision":REVISION,"scope":scope,"cache_hit":cache_hit,"idtt_verdict":probe_verdict,
        "reason":"no verified strategy","elapsed_ms":0.0});
    if let Some(cert)=cert {
        match check::verify(&req.history,&cert,deadline,50000) {
            Ok(moves)=>{
                let mut guard=cache.lock().map_err(|_|"cache lock")?;
                if guard.len()>=128 {guard.clear();}
                guard.insert(key,cert.clone());
                response["status"]=json!("PROVEN_WIN");response["native_verified"]=json!(true);
                response["moves"]=json!(moves);response["certificate"]=serde_json::to_value(cert).map_err(|e|e.to_string())?;
                response["reason"]=json!("independent raw-board strategy verification");
            }
            Err(reason)=>response["reason"]=json!(reason),
        }
    }
    // Certificate reconstruction/serialization and verification share the total budget.
    if Instant::now()>=deadline {
        response["status"]=json!("UNKNOWN");response["native_verified"]=json!(false);
        response["moves"]=json!([]);response["certificate"]=Value::Null;
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
