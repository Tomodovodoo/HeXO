//! Raw-coordinate strategy checker. No upstream rules, completion or cover helpers.
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;
use hexo_solver::prover::certificate::{ProofCertificate, ProofNode};
type Point = (i32,i32);
pub type Board = BTreeMap<Point,u8>;
const AXES: [Point;3] = [(1,0),(0,1),(1,-1)];
pub const LIMIT: i32 = 1_000_000;

pub fn phase(n: usize) -> (u8,u8) {
    if n == 0 {(0,1)} else {(((n+1)/2%2) as u8, if n%2==0 {1} else {2})}
}
pub fn won(board: &Board, at: Point, side: u8) -> bool {
    AXES.iter().any(|&(dq,dr)| {
        let mut count=1;
        for sign in [-1,1] {
            let mut k=1;
            while board.get(&(at.0+sign*k*dq,at.1+sign*k*dr))==Some(&side) {count+=1;k+=1;}
        }
        count>=6
    })
}
pub fn legal(board: &Board, p: Point) -> bool {
    !board.contains_key(&p)
        && if board.is_empty() {p==(0,0)} else {board.keys().any(|&(q,r)| {
            let (a,b)=(i64::from(p.0)-i64::from(q),i64::from(p.1)-i64::from(r));
            a.abs().max(b.abs()).max((a+b).abs())<=8
        })}
}
pub fn replay(history: &[Point]) -> Result<Board,String> {
    let mut b=Board::new();
    for (n,&p) in history.iter().enumerate() {
        if p.0.unsigned_abs()>LIMIT as u32 || p.1.unsigned_abs()>LIMIT as u32 || !legal(&b,p) {return Err("illegal history".into());}
        let side=phase(n).0;b.insert(p,side);
        if won(&b,p,side) {return Err("terminal input history".into());}
    }
    Ok(b)
}
fn check(deadline: Instant) -> Result<(),String> {
    if Instant::now()>=deadline {Err("verification deadline".into())} else {Ok(())}
}
fn completions(b:&Board, side:u8, allowance:u8, deadline:Instant) -> Result<BTreeSet<Vec<Point>>,String> {
    let mut segments=BTreeSet::new();
    for (&(q,r),&owner) in b {
        check(deadline)?;
        if owner!=side {continue;}
        for (dq,dr) in AXES {for offset in 0..6 {segments.insert((q-offset*dq,r-offset*dr,dq,dr));}}
    }
    let mut out=BTreeSet::new();
    for (q,r,dq,dr) in segments {
        check(deadline)?;
        let points:Vec<_>=(0..6).map(|k|(q+k*dq,r+k*dr)).collect();
        if points.iter().any(|p| b.get(p)==Some(&(1-side))) {continue;}
        let mut empty:Vec<_>=points.into_iter().filter(|p|!b.contains_key(p)).collect();
        // Every completion gap is within five of an existing same-color stone.
        if !empty.is_empty() && empty.len()<=allowance as usize && empty.iter().all(|&p|legal(b,p)) {
            empty.sort();out.insert(empty);
        }
    }
    Ok(out)
}
fn covers(threats:&BTreeSet<Vec<Point>>, deadline:Instant) -> Result<BTreeSet<Vec<Point>>,String> {
    let endpoints:Vec<_>=threats.iter().flatten().copied().collect::<BTreeSet<_>>().into_iter().collect();
    let hits=|p:&[Point]| threats.iter().all(|t|p.iter().any(|x|t.contains(x)));
    let mut result=BTreeSet::new();
    for (i,&a) in endpoints.iter().enumerate() {
        check(deadline)?;
        if hits(&[a]) {result.insert(vec![a]);}
        for &b in &endpoints[i+1..] {if hits(&[a,b]) {result.insert(vec![a,b]);}}
    }
    Ok(result)
}
pub fn apply(b:&Board, n:usize, moves:&[Point]) -> Result<(Board,usize,bool),String> {
    let (side,remaining)=phase(n);
    if moves.is_empty() || moves.len()>remaining as usize {return Err("invalid turn length".into());}
    let mut out=b.clone();let mut win=false;
    for &p in moves {
        if win || !legal(&out,p) {return Err("illegal or post-terminal move".into());}
        out.insert(p,side);win=won(&out,p,side);
    }
    if !win && phase(n+moves.len()).0==side {return Err("incomplete turn".into());}
    Ok((out,n+moves.len(),win))
}

pub fn defenses(b:&Board, attacker:u8, deadline:Instant) -> Result<BTreeMap<Vec<Point>,Vec<Point>>,String> {
    if !completions(b,1-attacker,2,deadline)?.is_empty() {return Err("defender counterwin".into());}
    let threats=completions(b,attacker,2,deadline)?;
    if threats.is_empty() {return Err("quiet defender unsupported".into());}
    let small=covers(&threats,deadline)?;
    let mut result=BTreeMap::new();
    for cover in small {
        check(deadline)?;
        if cover.len()==2 {result.insert(cover.clone(),cover);continue;}
        let fixed=cover[0];let mut post=b.clone();post.insert(fixed,1-attacker);
        // Full finite legal frontier AFTER the mandatory block. This includes
        // fillers made legal by that first placement, with that order retained.
        let mut frontier=BTreeSet::new();
        for &(q,r) in post.keys() {
            check(deadline)?;
            for dq in -8i32..=8 {for dr in -8i32..=8 {
                if dq.abs().max(dr.abs()).max((dq+dr).abs())<=8 && !post.contains_key(&(q+dq,r+dr)) {
                    frontier.insert((q+dq,r+dr));
                }
            }}
        }
        for filler in frontier {
            check(deadline)?;
            let mut key=vec![fixed,filler];key.sort();
            result.entry(key).or_insert(vec![fixed,filler]);
            if result.len()>50000 {return Err("free-second coverage work limit".into());}
        }
    }
    Ok(result)
}

pub fn verify(history:&[Point], cert:&ProofCertificate, deadline:Instant, max_nodes:usize) -> Result<Vec<Point>,String> {
    if cert.version!=1 || cert.width!="wide" || cert.nodes.len()>max_nodes {return Err("certificate format/size".into());}
    let board=replay(history)?;
    let attacker=phase(history.len()).0;
    struct Checker<'a> {cert:&'a ProofCertificate, attacker:u8, deadline:Instant, left:usize, stack:BTreeSet<u32>}
    impl Checker<'_> {
        fn walk(&mut self,id:u32,b:&Board,n:usize) -> Result<(),String> {
            check(self.deadline)?;
            if self.left==0 || self.stack.len()>=128 || !self.stack.insert(id) {return Err("certificate work limit/cycle/depth".into());}
            self.left-=1;
            let node=self.cert.nodes.get(id as usize).ok_or("invalid certificate edge")?;
            let (side,remaining)=phase(n);
            match node {
                ProofNode::ImmediateWin{action} => {
                    if side!=self.attacker || !apply(b,n,action)?.2 {return Err("false immediate win".into());}
                }
                ProofNode::AttackerMove{action,child,..} => {
                    if side!=self.attacker {return Err("attacker phase mismatch".into());}
                    let (next,ply,terminal)=apply(b,n,action)?;
                    if terminal {return Err("terminal move must use immediate-win leaf".into());}
                    self.walk(*child,&next,ply)?;
                }
                ProofNode::DefenderReplies{responses} => {
                    if side==self.attacker || remaining!=2 {return Err("defender phase mismatch".into());}
                    if !completions(b,side,remaining,self.deadline)?.is_empty() {return Err("defender counterwin".into());}
                    let required=defenses(b,self.attacker,self.deadline)?;
                    if required.is_empty() {return Err("defenses supplied for unstoppable position".into());}
                    if responses.len()!=required.len() {return Err("missing defense branch including free-second coverage".into());}
                    let mut seen=BTreeSet::new();
                    for reply in responses {
                        let mut key=reply.action.clone();key.sort();
                        if !required.contains_key(&key) || !seen.insert(key) {return Err("invalid/duplicate defense".into());}
                        let (next,ply,terminal)=apply(b,n,&reply.action)?;
                        if terminal {return Err("defender wins".into());}
                        self.walk(reply.child,&next,ply)?;
                    }
                    if seen!=required.keys().cloned().collect() {return Err("missing defense branch including free-second coverage".into());}
                }
                ProofNode::Unstoppable{..} => {
                    if side==self.attacker || remaining!=2 {return Err("unstoppable phase mismatch".into());}
                    if !completions(b,side,remaining,self.deadline)?.is_empty() {return Err("defender counterwin".into());}
                    let threats=completions(b,self.attacker,2,self.deadline)?;
                    if threats.is_empty() || !covers(&threats,self.deadline)?.is_empty() {return Err("false unstoppable".into());}
                }
            }
            self.stack.remove(&id);Ok(())
        }
    }
    let mut checker=Checker{cert,attacker,deadline,left:max_nodes,stack:BTreeSet::new()};
    checker.walk(cert.root,&board,history.len())?;
    check(deadline)?;
    match &cert.nodes[cert.root as usize] {
        ProofNode::ImmediateWin{action}|ProofNode::AttackerMove{action,..}=>Ok(action.clone()),
        _=>Err("root must be attacker action".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn free_filler_frontier_expands_after_mandatory_block() {
        for sign in [-1,1] {
            let mut b=Board::new();
            for q in 0..5 {b.insert((sign*q,0),0);}
            b.insert((-sign,0),1);
            let fixed=(sign*5,0);let filler=(sign*13,0);
            assert!(!legal(&b,filler));
            let replies=defenses(&b,0,Instant::now()+std::time::Duration::from_secs(1)).unwrap();
            let mut key=vec![fixed,filler];key.sort();
            assert_eq!(replies.get(&key),Some(&vec![fixed,filler]));
            b.insert(fixed,1);assert!(legal(&b,filler));
        }
    }
}
