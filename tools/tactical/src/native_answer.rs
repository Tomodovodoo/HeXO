//! Typed, owned completions for the native graph scheduler. Only this module
//! constructs these handles, from the same checked query path as the JSON ABI.
use super::*;

struct Answer {value:Value,info:[u64;13],moves:Vec<(i32,i32)>}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hexo_tactical_worker_answer(worker:*mut std::ffi::c_void,input:*const c_char)->*mut std::ffi::c_void {
    if worker.is_null() {return std::ptr::null_mut();}
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let worker=unsafe{&*worker.cast::<Worker>()};let mut scope=None;
        let mut value=unsafe{query_value(input,|req,start,dispatched| {
            scope=Some((req.history.len(),req.attacker));
            dispatch_on(worker,req,start,dispatched)
        })};
        worker.stats(&mut value);
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
        Box::into_raw(Box::new(Answer{value,info,moves})).cast()
    })).unwrap_or(std::ptr::null_mut())
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
