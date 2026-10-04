mod check;
use std::collections::BTreeMap;
use std::ffi::{CStr,CString,c_char};
use std::sync::{Arc,Mutex,OnceLock};
use std::sync::atomic::{AtomicBool,Ordering};
#[cfg(not(target_family="wasm"))]
use std::sync::mpsc;
use std::time::{Duration,Instant};
use hexo_engine::types::Player;
use hexo_solver::forcing::Meter;
use hexo_solver::prover::{self,Ctl,DriverKind,ProverConfig};
use hexo_solver::prover::io::{Position,PosConfig,Verdict};
use hexo_solver::prover::certificate::{ProofCertificate,ProofNode,ProofResponse,ExactFact,ExactScope,exact_at};
use serde::Deserialize;
use serde_json::{json,Value};

const REVISION:&str="5a771e572553a8bd8e010112b2ce65f16e5afa1b";
/// PDS-PN table size: one MiB per this many budgeted nodes, clamped to 1..=16 MiB.
const NODES_PER_TT_MB:u64=2048;
/// Certificate size and checker visits scale with search work, up to 200,000.
fn check_nodes(nodes:u64)->usize {(nodes.saturating_mul(8)).clamp(50_000,200_000) as usize}
// Exact state keys, never Zobrist hashes. Rules/scope are fixed by this library version.
// The budgets (and the IDTT depth when IDTT runs) and the shortest flag are part of the key: a search is a
// function of position, budgets and that flag. Searches with resident state are keyed apart, so a
// cold (table_mb 0) query never replays a result that depended on earlier queries. A hit replays the search's certificate, work and
// IDTT verdict, so it is indistinguishable from a fresh search except for `cache_hit`.
type Key=(Vec<((i32,i32),u8)>,u8,u8,u64,u64,u8,bool,bool);
type Solved=(ProofCertificate,u64,Option<String>,bool);
static CACHE:OnceLock<Mutex<BTreeMap<Key,Solved>>>=OnceLock::new();
/// Whose forced win is asked: the side to move, or its opponent given a fresh
/// two-placement turn on the current stones (a flipped-turn threat query).
#[derive(Deserialize,Clone,Copy,PartialEq,Default)]
#[serde(rename_all="lowercase")]
enum Attacker {#[default] Mover, Opponent, Defender}
/// Trusted exact game outcomes from the caller's graph, supplied separately
/// from certificates. A certificate cannot manufacture its own premises.
#[derive(Deserialize,serde::Serialize)]
#[serde(deny_unknown_fields)]
struct Known {history:Vec<(i32,i32)>,winner:u8,plies:u32}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    history:Vec<(i32,i32)>, ms:u32, nodes:u64, idtt_nodes:u64, depth:u8,
    /// Prepared cancellation token; zero keeps the legacy query ABI.
    #[serde(default)] request_id:u64,
    #[serde(default)] attacker:Attacker,
    #[serde(default)] certificate:Option<ProofCertificate>,
    #[serde(default)] root_moves:Option<Vec<(i32,i32)>>,
    /// Resident search state of this worker thread in megabytes (dfpn::set_resident); 0 = none.
    #[serde(default)] table_mb:u64,
    /// After a found proof, spend the rest of `nodes` tightening it to the fewest attacker turns
    /// (guided PDS-PN threshold probes); `shortest` in the response says whether that minimum is exact.
    #[serde(default)] shortest:bool,
    /// Return scoped forcing-search numbers, including on UNKNOWN.
    #[serde(default)] bounds:bool,
    /// Carry resident entries and proven witnesses through table resizes.
    #[serde(default)] resume:bool,
    #[serde(default)] known:Vec<Known>,
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
    let defenses=check::defenses(&post,side,ctl)?;
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
        if cert.nodes.len()+child.nodes.len()>check_nodes(req.nodes) {return Err("candidate certificate size limit".into());}
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

/// Start at the real defender turn, including a one-stone half turn. All moves
/// outside the complete cover set lose immediately; every cover needs a proof.
fn defend(board:&check::Board,ply:usize,req:&Request,ctl:&Ctl,meter:&Meter)->Result<ProofCertificate,String> {
    let (mover,remaining)=check::phase(ply);
    let attacker=1-mover;
    let root=position(board,mover,remaining);
    if let Some((fact,known))=exact_at(&root.stones,root.attacker,remaining) {
        if known.winner==root.attacker {return Err("defender root is exact won".into());}
        return Ok(ProofCertificate{version:1,width:"wide".into(),root:0,nodes:vec![ProofNode::Exact{fact,after:vec![]}]});
    }
    let defenses=check::defenses_at(board,attacker,remaining,ctl)?;
    let mut cert=ProofCertificate{version:1,width:"wide".into(),root:0,nodes:vec![ProofNode::Unstoppable{threats:vec![]}]};
    if defenses.is_empty() {return Ok(cert);}
    let mut responses=Vec::new();
    for action in defenses.into_values() {
        if ctl.expired() {return Err("defender query deadline".into());}
        let (next,n,won)=check::apply(board,ply,&action)?;
        if won {return Err("defender counterwin".into());}
        let pos=position(&next,attacker,check::phase(n).1);
        let mut terminal=exact_at(&pos.stones,pos.attacker,pos.placements_remaining).map(|(id,f)|(id,f,vec![]));
        // The graph can already have refuted a first stone without allocating
        // its second-stone children. Every continuation of that losing turn is
        // still lost, and the certificate records the actual remaining move.
        if terminal.is_none() && action.len()==2 {
            for first in 0..2 {
                let mut partial=board.clone();partial.insert(action[first],mover);
                if !check::legal(board,action[first]) {continue;}
                let prefix=position(&partial,mover,1);
                if let Some((id,fact))=exact_at(&prefix.stones,prefix.attacker,1) {
                    terminal=Some((id,fact,vec![action[1-first]]));break;
                }
            }
        }
        let mut child=if let Some((fact,known,after))=terminal {
            if known.winner!=pos.attacker {return Err("defender has an exact winning cover".into());}
            ProofCertificate{version:1,width:"wide".into(),root:0,nodes:vec![ProofNode::Exact{fact,after}]}
        } else {
            if meter.spent()>=req.nodes {return Err("defender query node budget".into());}
            let cfg=ProverConfig{driver:DriverKind::Pdspn,wide:true,node_budget:req.nodes,tt_mb:1,pn2_nodes:1000,..Default::default()};
            prover::pdspn::solve(&pos,&cfg,ctl).certificate.ok_or("unproved defender continuation")?
        };
        if cert.nodes.len()+child.nodes.len()>check_nodes(req.nodes) {return Err("defender certificate size limit".into());}
        let offset=cert.nodes.len() as u32;
        responses.push(ProofResponse{action,child:offset+child.root});
        for node in &mut child.nodes {match node {
            ProofNode::AttackerMove{child,alternatives,..}=>{*child+=offset;for a in alternatives {a.child+=offset;}}
            ProofNode::DefenderReplies{responses}=>{for r in responses {r.child+=offset;}}
            _=>{}
        }}
        cert.nodes.extend(child.nodes);
    }
    cert.nodes[0]=ProofNode::DefenderReplies{responses};Ok(cert)
}
/// The certificate re-proved at the fewest attacker turns the remaining nodes and half the remaining time can
/// establish, and whether that count is the exact minimum over the solver's forcing width; None when the
/// probes returned no certificate. The caller verifies it.
fn shorten(pos:&Position,cert:&ProofCertificate,req:&Request,ctl:&Ctl,meter:&Meter)->Option<(ProofCertificate,bool)> {
    let left=req.nodes.saturating_sub(meter.spent());
    let deadline=ctl.deadline.map(|d|{let now=Instant::now();now+d.saturating_duration_since(now)/2});
    let ctl=Ctl{deadline,..ctl.clone()};
    let cfg=ProverConfig{driver:DriverKind::PdspnShortest,wide:true,node_budget:left,
        tt_mb:(left/NODES_PER_TT_MB).clamp(1,16) as usize,pn2_nodes:1000,..Default::default()};
    let found=prover::guided_pdspn_shortest(pos,cert,&cfg,&ctl).ok()?;
    Some((found.certificate?,found.verdict==Verdict::Win))
}
/// One query. `nodes` bounds the total search work (IDTT nodes plus PDS-PN level-1
/// nodes and level-2 expansions), so with `table_mb` 0 the verdict and certificate are a
/// function of (position, attacker, nodes, idtt_nodes, build); a resident table
/// (`table_mb` > 0) carries search state across queries, so they then also depend on the
/// earlier queries of the worker. The checker accepts at most
/// clamp(8 * nodes, 50,000, 200,000) certificate nodes and visits. `ms` is only a
/// safety cap, and a query that reaches it returns UNKNOWN.
fn run_controlled(req:Request, start:Instant, cancel:Arc<AtomicBool>) -> Result<Value,String> {
    if req.history.len()>800 || req.ms==0 || req.ms>60000 || req.nodes==0 || req.nodes>10_000_000
        || req.idtt_nodes>=req.nodes || req.depth==0 || req.depth>64 || req.table_mb>256
        || (req.resume && req.table_mb==0) || req.known.len()>4096
        || req.known.iter().map(|k|k.history.len()).sum::<usize>()>200_000
        || (req.attacker==Attacker::Defender && req.root_moves.is_some()) {return Err("invalid tactical limits".into());}
    let deadline=start+Duration::from_millis(req.ms as u64);
    let meter=Meter::new(req.nodes);
    let ctl=Ctl{deadline:Some(deadline),cancel,meter:Some(meter.clone())};
    if ctl.expired() {return Err("cancelled or deadline".into());}
    let mut facts=Vec::new();
    let mut identities=BTreeMap::new();
    for known in &req.known {
        if known.history.len()>800 || known.winner>1 || known.plies==0 || known.plies>10_000 {return Err("invalid exact premise".into());}
        let board=check::replay_controlled(&known.history,&ctl)?;
        let (side,remaining)=check::phase(known.history.len());
        if side!=known.winner && known.plies<=remaining as u32 {return Err("exact premise distance precedes winner's turn".into());}
        let pos=position(&board,side,remaining);
        if let Some(winner)=identities.insert((board,side,remaining),known.winner) {
            if winner!=known.winner {return Err("contradictory exact premises".into());}
            return Err("duplicate exact premise".into());
        }
        let turns=(known.plies+if side==known.winner {4-remaining as u32} else {2-remaining as u32}+3)/4;
        facts.push(ExactFact{stones:pos.stones,player:pos.attacker,remaining,
            winner:if known.winner==0 {Player::P1}else{Player::P2},turns});
    }
    let _facts=ExactScope::new(facts);
    // Premise-dependent search state must never serve another snapshot.
    if !req.known.is_empty() {prover::dfpn::set_resident(0);}
    else if req.resume {prover::dfpn::set_resident_resume(req.table_mb as usize);}
    else {prover::dfpn::set_resident(req.table_mb as usize);}
    let board=check::replay_controlled(&req.history,&ctl)?;
    let ply=if req.attacker==Attacker::Opponent {check::flip(req.history.len())} else {req.history.len()};
    let (mover,remaining)=check::phase(ply);
    let side=if req.attacker==Attacker::Defender {1-mover} else {mover};
    let cacheable=req.known.is_empty() && req.attacker!=Attacker::Defender;
    let fresh=req.certificate.is_none() && req.root_moves.is_none();
    let key=(board.iter().map(|(&p,&s)|(p,s)).collect(),side,remaining,req.nodes,req.idtt_nodes,
        if req.idtt_nodes>0 {req.depth} else {0},req.table_mb>0,req.shortest && fresh);
    let scope=json!({"rules":{"win_length":6,"placement_radius":8,"match_move_cap":null},
        "defenses":"all legal covers for the remaining stones including complete free-second frontier; quiet defender nodes unsupported",
        "attacks":"wide Strix proposals plus optional root candidate; selective negatives remain UNKNOWN","checker_version":4,
        "exact_premises":req.known.len(),
        "budget":{"nodes":req.nodes,"idtt_nodes":req.idtt_nodes,"idtt_depth_cap":req.depth,"safety_ms":req.ms,
            "work":"one shared meter over IDTT nodes, PDS-PN level-1 nodes and level-2 expansions; verifier path limit 128",
            "check_nodes":check_nodes(req.nodes)}});
    let cache=CACHE.get_or_init(||Mutex::new(BTreeMap::new()));
    let mut cache_hit=false;
    let mut probe_verdict=None;
    let mut cached_nodes=None;
    let mut exact=false;
    let mut proof_numbers=None;
    let mut resident_reused=false;
    let cert=if let Some(cert)=req.certificate.clone() {Some(cert)} else if req.attacker==Attacker::Defender {
        defend(&board,ply,&req,&ctl,&meter).ok()
    } else if let Some(moves)=&req.root_moves {
        Some(complete_candidate(&board,ply,moves,&req,&ctl,&meter)?)
    } else {
        let saved=if cacheable {cache.lock().map_err(|_|"cache lock")?.get(&key).cloned()} else {None};
        if let Some((cert,used,verdict,minimal))=saved {
            proof_numbers=Some((0,prover::PROOF_NUMBER_INFINITY));
            cache_hit=true;cached_nodes=Some(used);probe_verdict=verdict;exact=minimal;Some(cert)
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
            let found=if ctl.expired() {None} else {
                let found=prover::pdspn::solve(&pos,&cfg,&ctl);
                proof_numbers=found.proof_numbers;resident_reused=found.resident_reused;
                found.certificate
            };
            match found {
                Some(cert) if req.shortest && req.known.is_empty() =>
                    match shorten(&pos,&cert,&req,&ctl,&meter) {
                        Some((tight,minimal)) if check::verify(&req.history,ply,&tight,&ctl,check_nodes(req.nodes)).is_ok() =>
                            {exact=minimal;Some(tight)}
                        _=>Some(cert),
                    },
                found=>found,
            }
        }
    };
    let mut response=json!({"status":"UNKNOWN","native_verified":false,"moves":[],"certificate":null,
        "revision":REVISION,"scope":scope,"cache_hit":cache_hit,"idtt_verdict":probe_verdict.clone(),
        "attacker":match req.attacker {Attacker::Opponent=>"opponent",Attacker::Defender=>"defender",Attacker::Mover=>"mover"},
        "reason":"no verified strategy","nodes_used":cached_nodes.unwrap_or(meter.spent().min(req.nodes)),"proof_turns":null,"elapsed_ms":0.0,
        "nodes_fresh":meter.spent().min(req.nodes),
        "shortest":false});
    if req.bounds {
        response["proof_numbers"]=proof_numbers.map(|(pn,dn)|json!({"pn":pn,"dn":dn,
            "infinity":prover::PROOF_NUMBER_INFINITY,"scope":"wide-forcing","game_exact":false})).unwrap_or(Value::Null);
    }
    if req.resume {response["resident_reused"]=json!(resident_reused);}
    if let Some(mut cert)=cert {
        match check::verify_for(&req.history,ply,side,&cert,&ctl,check_nodes(req.nodes)) {
            Ok((moves,turns,visited))=>{
                // Only the primary verified strategy may declare dependencies.
                // Drop unused nodes/alternatives in certificates containing exact
                // leaves, so an unchecked fact cannot leak into saved evidence.
                if cert.nodes.iter().any(|n|matches!(n,ProofNode::Exact{..})) {
                    let indices:BTreeMap<_,_>=visited.iter().enumerate().map(|(i,&id)|(id,i as u32)).collect();
                    cert.nodes=visited.iter().map(|&id| {
                        let mut node=cert.nodes[id as usize].clone();
                        match &mut node {
                            ProofNode::AttackerMove{child,alternatives,..}=>{*child=indices[child];alternatives.clear();}
                            ProofNode::DefenderReplies{responses}=>for reply in responses {reply.child=indices[&reply.child];},
                            _=>{},
                        }
                        node
                    }).collect();
                    cert.root=indices[&cert.root];
                }
                // A shortening cut short by the time cap is not a function of the key, so it is not kept.
                if cacheable && !cache_hit && fresh && (exact || !req.shortest) {
                    let mut guard=cache.lock().map_err(|_|"cache lock")?;
                    if guard.len()>=128 {guard.clear();}
                    guard.insert(key,(cert.clone(),meter.spent().min(req.nodes),probe_verdict,exact));
                }
                response["status"]=json!(if req.attacker==Attacker::Defender {"PROVEN_LOSS"} else {"PROVEN_WIN"});response["native_verified"]=json!(true);
                response["winner"]=json!(side);
                response["exact_hits"]=json!(cert.nodes.iter().filter(|n|matches!(n,ProofNode::Exact{..})).count());
                let used:std::collections::BTreeSet<_>=cert.nodes.iter().filter_map(|n|if let ProofNode::Exact{fact,..}=n {Some(*fact)} else {None}).collect();
                response["dependencies"]=json!(used.into_iter().map(|id|json!({"fact":id,"outcome":req.known[id as usize]})).collect::<Vec<_>>());
                response["moves"]=json!(moves);response["proof_turns"]=json!(turns);response["shortest"]=json!(exact);
                response["certificate"]=serde_json::to_value(cert).map_err(|e|e.to_string())?;
                response["reason"]=json!(if req.known.is_empty() {"independent raw-board strategy verification"} else {"raw-board strategy verified against supplied exact graph premises"});
            }
            Err(reason)=>response["reason"]=json!(reason),
        }
    }
    // Certificate reconstruction/serialization and verification share the safety cap.
    if ctl.expired() {
        response["status"]=json!("UNKNOWN");response["native_verified"]=json!(false);
        response["moves"]=json!([]);response["certificate"]=Value::Null;response["proof_turns"]=Value::Null;
        response["shortest"]=json!(false);
        response["reason"]=json!(if ctl.cancel.load(Ordering::Acquire) {"cancelled"} else {"deadline"});
    }
    response["elapsed_ms"]=json!(start.elapsed().as_secs_f64()*1000.0);
    Ok(response)
}

#[cfg(test)]
fn run(req:Request,start:Instant)->Result<Value,String> {
    run_controlled(req,start,Arc::new(AtomicBool::new(false)))
}

// Tokens exist before admission, so cancellation also covers the handoff to the
// worker. IDs are never reused; releasing a caller's token does not invalidate
// the Arc retained by a timed-out worker or cancel its next query.
#[derive(Default)]
struct Controls {next:u64, pending:BTreeMap<u64,Arc<AtomicBool>>}
static CONTROLS:OnceLock<Mutex<Controls>>=OnceLock::new();
fn controls()->&'static Mutex<Controls> {CONTROLS.get_or_init(||Mutex::new(Controls::default()))}
#[unsafe(no_mangle)]
pub extern "C" fn hexo_tactical_prepare()->u64 {
    let Ok(mut controls)=controls().lock() else {return 0;};
    if controls.pending.len()>=64 {return 0;}
    let Some(id)=controls.next.checked_add(1) else {return 0;};
    controls.next=id;
    controls.pending.insert(id,Arc::new(AtomicBool::new(false)));
    id
}
#[unsafe(no_mangle)]
pub extern "C" fn hexo_tactical_cancel(request_id:u64)->bool {
    let Ok(controls)=controls().lock() else {return false;};
    let Some(cancel)=controls.pending.get(&request_id) else {return false;};
    cancel.store(true,Ordering::Release);
    true
}
#[unsafe(no_mangle)]
pub extern "C" fn hexo_tactical_release(request_id:u64) {
    if let Ok(mut controls)=controls().lock() {controls.pending.remove(&request_id);}
}
fn query_control(request_id:u64)->Result<Arc<AtomicBool>,String> {
    if request_id==0 {return Ok(Arc::new(AtomicBool::new(false)));}
    controls().lock().map_err(|_|"cancellation token lock")?
        .pending.get(&request_id).cloned().ok_or("unknown cancellation token".into())
}

// One resident native worker. Search, certificate work and the raw-board checker
// share a cancellation token; the isolated wrapper remains the hard-stop fallback.
#[cfg(not(target_family="wasm"))]
type Work=(Request,Instant,Arc<AtomicBool>,mpsc::Sender<Result<Value,String>>);
#[cfg(not(target_family="wasm"))]
static WORKER:OnceLock<mpsc::SyncSender<Work>>=OnceLock::new();
static BUSY:AtomicBool=AtomicBool::new(false);
static LAST_WORK:OnceLock<Mutex<Value>>=OnceLock::new();
#[cfg(target_family="wasm")]
fn dispatch(req:Request,start:Instant,dispatched:&mut bool)->Result<Value,String> {
    if req.ms==0 || req.ms>60000 {return Err("invalid deadline".into());}
    let cancel=query_control(req.request_id)?;
    if cancel.load(Ordering::Acquire) {return Err("cancelled".into());}
    *dispatched=true;
    run_controlled(req,start,cancel)
}
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
#[cfg(not(any(windows,target_family="wasm")))]
fn thread_cpu_ms()->Option<f64> {None}
#[cfg(not(target_family="wasm"))]
fn dispatch(req:Request,start:Instant,dispatched:&mut bool)->Result<Value,String> {
    if req.ms==0 || req.ms>60000 {return Err("invalid deadline".into());}
    let deadline=start+Duration::from_millis(req.ms as u64);
    let worker=WORKER.get_or_init(|| {
        let (sender,receiver)=mpsc::sync_channel::<Work>(1);
        std::thread::spawn(move || {
            while let Ok((req,start,cancel,reply))=receiver.recv() {
                let budget_ms=req.ms;let cpu_start=thread_cpu_ms();
                let result=std::panic::catch_unwind(||run_controlled(req,start,Arc::clone(&cancel)))
                    .unwrap_or_else(|_|Err("native worker panic".into()));
                let elapsed=start.elapsed().as_secs_f64()*1000.0;
                let cpu=thread_cpu_ms().zip(cpu_start).map(|(a,b)|a-b);
                if let Ok(mut stats)=LAST_WORK.get_or_init(||Mutex::new(Value::Null)).lock() {
                    let late=elapsed>=budget_ms as f64;
                    let previous=stats.clone();
                    let row=json!({"elapsed_ms":elapsed,"thread_cpu_ms":cpu,
                        "requested_ms":budget_ms,"completed_after_deadline":late,
                        "cancelled":cancel.load(Ordering::Acquire)});
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
    let cancel=query_control(req.request_id)?;
    if cancel.load(Ordering::Acquire) {return Err("cancelled".into());}
    if BUSY.compare_exchange(false,true,Ordering::AcqRel,Ordering::Acquire).is_err() {
        return Err("native worker busy finishing bounded prior query".into());
    }
    let (send,recv)=mpsc::channel();
    if worker.send((req,start,Arc::clone(&cancel),send)).is_err() {BUSY.store(false,Ordering::Release);return Err("native worker stopped".into());}
    *dispatched=true;
    match recv.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
        Ok(result)=>result,
        Err(_)=>{
            cancel.store(true,Ordering::Release);
            Err("deadline; prior native work cancelling".into())
        }
    }
}

/// One serialized native query at a time. Python holds a lock around calls.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hexo_tactical_query(input:*const c_char)->*mut c_char {
    let start=Instant::now();
    let mut dispatched=false;
    let result=std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if input.is_null(){return Err("null request".to_string());}
        let bytes=unsafe{CStr::from_ptr(input)}.to_bytes();
        if bytes.len()>64*1024*1024{return Err("request size limit".into());}
        serde_json::from_slice(bytes).map_err(|e|e.to_string()).and_then(|req|dispatch(req,start,&mut dispatched))
    }));
    let fresh=if dispatched {Value::Null}else{json!(0)};
    let mut value=match result {
        Ok(Ok(v))=>v,
        Ok(Err(reason))=>json!({"status":"UNKNOWN","native_verified":false,"reason":reason,"moves":[],"nodes_fresh":fresh}),
        Err(_)=>json!({"status":"UNKNOWN","native_verified":false,"reason":"native panic","moves":[],"nodes_fresh":fresh}),
    };
    value["background_worker_busy"]=json!(BUSY.load(Ordering::Acquire));
    value["last_worker_completion"]=LAST_WORK.get_or_init(||Mutex::new(Value::Null))
        .lock().map(|stats|stats.clone()).unwrap_or(Value::Null);
    value["equal_compute_clock"]=json!(false);
    CString::new(value.to_string()).unwrap().into_raw()
}
/// A request buffer of `len` spaces plus a NUL: the caller writes exactly `len` non-NUL bytes and
/// releases it with hexo_tactical_free.
#[unsafe(no_mangle)]
pub extern "C" fn hexo_tactical_alloc(len:usize)->*mut c_char {
    CString::new(vec![b' ';len]).unwrap().into_raw()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hexo_tactical_free(value:*mut c_char) {
    if !value.is_null(){drop(unsafe{CString::from_raw(value)});}
}

#[cfg(test)]
mod tests {
    use super::*;
    const OPEN_THREE:[(i32,i32);7]=[(0,0),(0,8),(2,8),(1,0),(2,0),(4,8),(6,8)];
    const IMMEDIATE:[(i32,i32);11]=[(0,0),(0,3),(1,3),(1,0),(2,0),(2,3),(3,3),(3,0),(4,0),(4,3),(5,4)];
    #[test]
    fn cancellation_tokens_survive_handoff_and_never_alias_successors() {
        let old=hexo_tactical_prepare();
        assert_ne!(old,0);
        let retained=query_control(old).unwrap();
        assert!(hexo_tactical_cancel(old));
        hexo_tactical_release(old);
        let next=hexo_tactical_prepare();
        assert_ne!(old,next);
        assert!(!hexo_tactical_cancel(old));
        assert!(retained.load(Ordering::Acquire));
        assert!(!query_control(next).unwrap().load(Ordering::Acquire));
        let req=serde_json::from_value(json!({"history":IMMEDIATE,"ms":1000,"nodes":1000,
            "idtt_nodes":0,"depth":8,"request_id":next})).unwrap();
        assert!(hexo_tactical_cancel(next));
        let mut dispatched=false;
        assert!(dispatch(req,Instant::now(),&mut dispatched).unwrap_err().contains("cancelled"));
        assert!(!dispatched);
        hexo_tactical_release(next);
        assert!(query_control(next).is_err());
    }
    #[test]
    fn rejected_native_requests_have_confirmed_zero_fresh_work() {
        let query=|input:&str| {
            let input=CString::new(input).unwrap();
            let raw=unsafe{hexo_tactical_query(input.as_ptr())};
            let result:Value=unsafe{serde_json::from_slice(CStr::from_ptr(raw).to_bytes()).unwrap()};
            unsafe{hexo_tactical_free(raw)};
            assert_eq!(result["status"],"UNKNOWN");
            assert_eq!(result["nodes_fresh"],0);
            result
        };
        query("{");
        query(r#"{"history":[],"ms":0,"nodes":1,"idtt_nodes":0,"depth":8}"#);
        let token=hexo_tactical_prepare();
        hexo_tactical_cancel(token);
        let result=query(&json!({"history":[],"ms":1000,"nodes":1,"idtt_nodes":0,"depth":8,"request_id":token}).to_string());
        assert_eq!(result["reason"],"cancelled");
        hexo_tactical_release(token);
        #[cfg(not(target_family="wasm"))]
        {
            assert!(!BUSY.swap(true,Ordering::AcqRel));
            let result=query(r#"{"history":[],"ms":1000,"nodes":1,"idtt_nodes":0,"depth":8}"#);
            BUSY.store(false,Ordering::Release);
            assert!(result["reason"].as_str().unwrap().contains("busy"));
        }
    }
    #[test]
    fn cancelled_certificate_checks_return_no_strategy() {
        let (pos,_,ctl,_)=setup(100000);
        let certificate=prover::pdspn::solve(&pos,&ProverConfig{wide:true,node_budget:100000,
            tt_mb:1,pn2_nodes:1000,..Default::default()},&ctl).certificate.unwrap();
        ctl.cancel.store(true,Ordering::Release);
        assert!(check::verify(&OPEN_THREE,OPEN_THREE.len(),&certificate,&ctl,200000).is_err());
        let req=serde_json::from_value(json!({"history":OPEN_THREE,"ms":1000,"nodes":100000,
            "idtt_nodes":0,"depth":8,"certificate":certificate})).unwrap();
        assert!(run_controlled(req,Instant::now(),Arc::clone(&ctl.cancel)).is_err());
    }
    #[test]
    fn certificate_limit_follows_node_budget() {
        assert_eq!((check_nodes(1),check_nodes(8192),check_nodes(27000)),(50000,65536,200000));
        let cert=ProofCertificate{version:1,width:"wide".into(),root:0,
            nodes:vec![ProofNode::ImmediateWin{action:vec![(5,0)]};50001]};
        let query=|nodes| {
            let req=serde_json::from_value(json!({"history":IMMEDIATE,"ms":60000,"nodes":nodes,
                "idtt_nodes":0,"depth":8,"certificate":cert})).unwrap();
            run(req,Instant::now()).unwrap()
        };
        assert_eq!(query(6250)["reason"],"certificate format/size");
        assert_eq!(query(8192)["status"],"PROVEN_WIN");
    }
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
    fn resident_state_proves_again_with_less_work() {
        let query=|table_mb:usize| {
            let (pos,cfg,ctl,meter)=setup(1_000_000);
            prover::dfpn::set_resident(table_mb);
            let solved=prover::pdspn::solve(&pos,&cfg,&ctl);
            (solved.certificate.is_some(),meter.spent())
        };
        let (fresh,cold)=query(0);
        let (first,_)=query(4);
        let (again,warm)=query(4);
        assert!(fresh && first && again,"every search proves the win");
        assert!(warm<cold,"the resident table saves work: {warm} of {cold}");
        prover::dfpn::set_resident(0);
    }
    #[test]
    fn resumed_proofs_survive_table_growth_and_shrink() {
        let query=|table_mb:usize,resume:bool| {
            let (pos,cfg,ctl,meter)=setup(1_000_000);
            if resume {prover::dfpn::set_resident_resume(table_mb);}
            else {prover::dfpn::set_resident(table_mb);}
            let solved=prover::pdspn::solve(&pos,&cfg,&ctl);
            let certificate=solved.certificate.unwrap();
            assert!(check::verify(&OPEN_THREE,OPEN_THREE.len(),&certificate,&ctl,200000).is_ok());
            (meter.spent(),solved.resident_reused)
        };
        let (cold,_)=query(0,false);
        let (_,first_reused)=query(4,true);
        assert!(!first_reused);
        for table_mb in [8,1] {
            let (spent,reused)=query(table_mb,true);
            assert!(reused,"a resize retains usable search entries");
            assert!(spent<cold,"verified warm proof uses less fresh work: {spent} vs {cold}");
        }
        prover::dfpn::set_resident(0);
    }
    #[test]
    fn proof_cache_reports_no_fresh_work() {
        let query=|| {
            let req=serde_json::from_value(json!({"history":OPEN_THREE,"ms":60000,"nodes":987_654,
                "idtt_nodes":0,"depth":8,"bounds":true})).unwrap();
            run(req,Instant::now()).unwrap()
        };
        let found=query();
        assert_eq!(found["status"],"PROVEN_WIN");
        assert!(found["nodes_fresh"].as_u64().unwrap()>0);
        let hit=query();
        assert_eq!(hit["status"],"PROVEN_WIN");
        assert_eq!(hit["cache_hit"],true);
        assert_eq!(hit["nodes_fresh"],0);
        assert!(hit["nodes_used"].as_u64().unwrap()>0);
        assert_eq!(hit["proof_numbers"]["pn"],0);
        assert_eq!(hit["proof_numbers"]["game_exact"],false);
    }
    #[test]
    fn resident_results_never_serve_cold_queries() {
        let query=|table_mb:u64| {
            let req=serde_json::from_value(json!({"history":OPEN_THREE,"ms":60000,"nodes":777_777,
                "idtt_nodes":0,"depth":8,"table_mb":table_mb})).unwrap();
            run(req,Instant::now()).unwrap()["cache_hit"].as_bool().unwrap()
        };
        assert!(!query(4),"first resident search");
        assert!(!query(0),"a cold query does not reuse the resident result");
        assert!(query(0),"cold results are cached for cold queries");
        prover::dfpn::set_resident(0);
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
