//! Typed, owned completions for the native graph scheduler. Only this module
//! constructs these handles, from the same checked query path as the JSON ABI.
use super::*;

struct Answer {value:Value,info:[u64;13],moves:Vec<(i32,i32)>,frontier:Vec<i64>}

// The native pool already has one stable dispatcher thread per handle. Bind on
// first execution, not construction: the pool creates handles on its owner.
struct DirectState {worker:WorkerState,thread:Mutex<Option<std::thread::ThreadId>>}
struct DirectWorker {state:Arc<DirectState>}
thread_local! {
    static DIRECT_OWNER:std::cell::RefCell<std::sync::Weak<DirectState>>=const{std::cell::RefCell::new(std::sync::Weak::new())};
}
impl DirectWorker {
    fn bind(&self)->Result<(),String> {
        let current=std::thread::current().id();
        let mut thread=self.state.thread.lock().map_err(|_|"direct worker affinity lock")?;
        if thread.is_some_and(|owner|owner!=current) {return Err("direct worker thread changed".into());}
        DIRECT_OWNER.with(|slot| {
            let mut owner=slot.borrow_mut();
            if let Some(other)=owner.upgrade() {
                if !Arc::ptr_eq(&other,&self.state) {return Err("direct worker thread already owned".into());}
            } else {
                // Replacing a retired handle must not inherit its cache,
                // retained search frontier, learned stamps or seeding progress.
                CACHE.with(|cache|cache.lock().map_err(|_|"result cache lock").map(|mut c|c.clear()))?;
                prover::dfpn::set_resident(0);
                stamps::reset_worker();
                *owner=Arc::downgrade(&self.state);
            }
            *thread=Some(current);Ok(())
        })
    }
}
struct Active<'a>(&'a WorkerState);
impl Drop for Active<'_> {
    fn drop(&mut self) {
        if let Ok(mut active)=self.0.active.lock() {*active=None;}
        self.0.busy.store(false,Ordering::Release);
    }
}
fn finish(value:&mut Value,cancel:&AtomicBool,deadline:Instant) {
    if cancel.load(Ordering::Acquire) || Instant::now()>=deadline {
        value["status"]=json!("UNKNOWN");value["native_verified"]=json!(false);
        value["moves"]=json!([]);value["certificate"]=Value::Null;
        value["proof_turns"]=Value::Null;value["shortest"]=json!(false);
        value["reason"]=json!("cancelled or deadline; completion collected");
    }
}
fn complete_direct(worker:&DirectWorker,req:Request,start:Instant,dispatched:&mut bool)->Result<Value,String> {
    if req.ms==0 || req.ms>60000 {return Err("invalid deadline".into());}
    let deadline=start+Duration::from_millis(req.ms as u64);
    let cancel=query_control(req.request_id)?;
    if cancel.load(Ordering::Acquire) {return Err("cancelled".into());}
    let state=&worker.state.worker;
    if state.busy.compare_exchange(false,true,Ordering::AcqRel,Ordering::Acquire).is_err() {
        return Err("direct worker busy".into());
    }
    let _active=Active(state);
    worker.bind()?;
    *state.active.lock().map_err(|_|"worker cancellation lock")?=Some(Arc::clone(&cancel));
    let budget_ms=req.ms;let cpu_start=thread_cpu_ms();
    *dispatched=true;
    let result=std::panic::catch_unwind(||run_controlled(req,start,Arc::clone(&cancel)))
        .unwrap_or_else(|_|Err("native worker panic".into()));
    state.record(start,budget_ms,&cancel,cpu_start);
    let mut value=result?;finish(&mut value,&cancel,deadline);Ok(value)
}

// A native scheduler already owns a background dispatcher. After the slice ends
// it must collect the cancelled worker's meter, rather than abandon its reply.
// Legacy query callers keep their bounded-wait dispatch_on contract.
fn complete(worker:&Worker,req:Request,start:Instant,dispatched:&mut bool)->Result<Value,String> {
    if req.ms==0 || req.ms>60000 {return Err("invalid deadline".into());}
    let deadline=start+Duration::from_millis(req.ms as u64);
    let cancel=query_control(req.request_id)?;
    if cancel.load(Ordering::Acquire) {return Err("cancelled".into());}
    if worker.state.busy.compare_exchange(false,true,Ordering::AcqRel,Ordering::Acquire).is_err() {
        return Err("native worker busy finishing bounded prior query".into());
    }
    *worker.state.active.lock().map_err(|_|"worker cancellation lock")?=Some(Arc::clone(&cancel));
    let (send,recv)=mpsc::channel();
    if worker.sender.as_ref().unwrap().send((req,start,Arc::clone(&cancel),send)).is_err() {
        worker.state.busy.store(false,Ordering::Release);return Err("native worker stopped".into());
    }
    *dispatched=true;
    let mut value=match recv.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
        Ok(result)=>result?,
        Err(mpsc::RecvTimeoutError::Disconnected)=>return Err("native worker stopped".into()),
        Err(mpsc::RecvTimeoutError::Timeout)=>{
            cancel.store(true,Ordering::Release);
            recv.recv().map_err(|_|"native worker stopped")??
        }
    };
    // A queued reply can also be received after the clock. Neither receive
    // route may expose an exact answer after cancellation or the deadline.
    finish(&mut value,&cancel,deadline);
    Ok(value)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hexo_tactical_worker_answer(worker:*mut std::ffi::c_void,input:*const c_char)->*mut std::ffi::c_void {
    if worker.is_null() {return std::ptr::null_mut();}
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let worker=unsafe{&*worker.cast::<Worker>()};let mut scope=None;
        let mut value=unsafe{query_value(input,|req,start,dispatched| {
            scope=Some((req.history.len(),req.attacker));
            complete(worker,req,start,dispatched)
        })};
        worker.stats(&mut value);
        make_answer(value,scope)
    })).unwrap_or(std::ptr::null_mut())
}

/// Direct handles own no background thread. Do not free during an ABI call;
/// the native pool joins its dispatcher before releasing the handle.
#[unsafe(no_mangle)]
pub extern "C" fn hexo_tactical_worker_new_direct()->*mut std::ffi::c_void {
    Box::into_raw(Box::new(DirectWorker{state:Arc::new(DirectState{
        worker:WorkerState::default(),thread:Mutex::new(None)})})).cast()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hexo_tactical_worker_answer_direct(worker:*mut std::ffi::c_void,input:*const c_char)->*mut std::ffi::c_void {
    if worker.is_null() {return std::ptr::null_mut();}
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let worker=unsafe{&*worker.cast::<DirectWorker>()};let mut scope=None;
        let mut value=unsafe{query_value(input,|req,start,dispatched| {
            scope=Some((req.history.len(),req.attacker));
            complete_direct(worker,req,start,dispatched)
        })};
        worker.state.worker.stats(&mut value);
        make_answer(value,scope)
    })).unwrap_or(std::ptr::null_mut())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hexo_tactical_worker_busy_direct(worker:*mut std::ffi::c_void)->bool {
    !worker.is_null() && unsafe{&*worker.cast::<DirectWorker>()}.state.worker.busy.load(Ordering::Acquire)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hexo_tactical_worker_free_direct(worker:*mut std::ffi::c_void) {
    if !worker.is_null() {drop(unsafe{Box::from_raw(worker.cast::<DirectWorker>())});}
}

fn make_answer(value:Value,scope:Option<(usize,Attacker)>)->*mut std::ffi::c_void {
        let mut info=[0;13];let mut moves=vec![];
        if let Some((stones,side))=scope {
            let (player,remaining)=check::phase(stones);
            info[9]=player as u64;info[10]=remaining as u64;
            info[11]=match side {Attacker::Mover=>0,Attacker::Defender=>1,Attacker::Opponent=>2};
            info[12]=1;
            let verified=value["native_verified"].as_bool()==Some(true);
            let winner=value["winner"].as_u64();let turns=value["proof_turns"].as_u64();
            let verdict=match (side,value["status"].as_str()) {
                (Attacker::Mover,Some("PROVEN_WIN"))=>1,
                (Attacker::Defender,Some("PROVEN_LOSS"))=>3,
                _=>0,
            };
            if verified && verdict!=0 && winner==Some(if verdict==1 {player as u64}else{1-player as u64}) && turns.is_some_and(|t|t>0) {
                moves=serde_json::from_value(value["moves"].clone()).unwrap_or_default();
                if verdict==3 || (!moves.is_empty() && moves.len()<=remaining as usize) {
                    info[0]=verdict;info[1]=winner.unwrap()+1;info[2]=turns.unwrap();
                }
            }
        }
        if let Some(fresh)=value["nodes_fresh"].as_u64() {info[3]=fresh;info[4]=1;}
        info[5]=value["nodes_used"].as_u64().unwrap_or(0);
        let numbers=&value["proof_numbers"];
        if numbers["scope"].as_str()==Some("wide-forcing") && numbers["game_exact"].as_bool()==Some(false) {
            if let (Some(pn),Some(dn))=(numbers["pn"].as_u64(),numbers["dn"].as_u64()) {info[6]=pn;info[7]=dn;info[8]=1;}
        }
        let mut frontier=vec![];
        if let Some(paths)=value["neural_frontier"].as_array() {
            for endpoint in paths {
                let path:Vec<(i32,i32)>=serde_json::from_value(endpoint["path"].clone()).unwrap();
                frontier.extend([path.len() as i64,endpoint["reason"].as_u64().unwrap() as i64]);
                for (q,r) in path {frontier.extend([q as i64,r as i64]);}
            }
        }
        Box::into_raw(Box::new(Answer{value,info,moves,frontier})).cast()
}

/// out: verdict(0/1/3), winner+1, proof turns, fresh nodes, fresh known,
/// historical nodes, pn, dn, bounds known, mover, remaining, side, scope known.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hexo_tactical_answer_info(answer:*const std::ffi::c_void,out:*mut u64)->bool {
    if answer.is_null() || out.is_null() {return false;}
    let answer=unsafe{&*answer.cast::<Answer>()};
    unsafe{std::ptr::copy_nonoverlapping(answer.info.as_ptr(),out,13)};true
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hexo_tactical_answer_moves(answer:*const std::ffi::c_void,out:*mut i64,capacity:usize)->i32 {
    if answer.is_null() {return -1;}
    let moves=&unsafe{&*answer.cast::<Answer>()}.moves;
    if out.is_null() {return moves.len() as i32;}
    if capacity<moves.len() {return -1;}
    for (i,&(q,r)) in moves.iter().enumerate() {unsafe{*out.add(2*i)=q as i64;*out.add(2*i+1)=r as i64;}}
    moves.len() as i32
}
/// Packed legal paths: stone count, reason, then q/r pairs. At most 8 paths of 64 stones.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hexo_tactical_answer_frontier(answer:*const std::ffi::c_void,out:*mut i64,capacity:usize)->i32 {
    if answer.is_null() {return -1;}
    let values=&unsafe{&*answer.cast::<Answer>()}.frontier;
    if out.is_null() {return values.len() as i32;}
    if capacity<values.len() {return -1;}
    unsafe{std::ptr::copy_nonoverlapping(values.as_ptr(),out,values.len())};values.len() as i32
}
/// Optional evidence serialization; release the returned buffer with
/// hexo_tactical_free. It is not needed to install a graph completion.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hexo_tactical_answer_json(answer:*const std::ffi::c_void)->*mut c_char {
    if answer.is_null() {return std::ptr::null_mut();}
    CString::new(unsafe{&*answer.cast::<Answer>()}.value.to_string()).unwrap().into_raw()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn hexo_tactical_answer_free(answer:*mut std::ffi::c_void) {
    if !answer.is_null() {drop(unsafe{Box::from_raw(answer.cast::<Answer>())});}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn direct_active_state_clears_when_execution_unwinds() {
        let state=WorkerState::default();
        state.busy.store(true,Ordering::Release);
        *state.active.lock().unwrap()=Some(Arc::new(AtomicBool::new(false)));
        let failed=std::panic::catch_unwind(|| {let _active=Active(&state);panic!("interrupted execution");});
        assert!(failed.is_err());
        assert!(!state.busy.load(Ordering::Acquire));
        assert!(state.active.lock().unwrap().is_none());
    }
}
