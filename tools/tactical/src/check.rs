//! Raw-coordinate strategy checker. No upstream rules, completion or cover helpers.
use std::collections::{BTreeMap, BTreeSet};
use hexo_solver::prover::Ctl;
use hexo_solver::prover::certificate::{ProofCertificate, ProofNode};
type Point = (i32,i32);
pub type Board = BTreeMap<Point,u8>;
const AXES: [Point;3] = [(1,0),(0,1),(1,-1)];
pub const LIMIT: i32 = 1_000_000;

pub fn phase(n: usize) -> (u8,u8) {
    if n == 0 {(0,1)} else {(((n+1)/2%2) as u8, if n%2==0 {1} else {2})}
}
/// The first ply after `n` at which the side not to move at `n` starts a fresh
/// two-placement turn: the phase of a flipped-turn query on the same stones.
pub fn flip(n: usize) -> usize {
    n+1+n%2
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
#[cfg(test)]
pub fn replay(history: &[Point]) -> Result<Board,String> {
    replay_controlled(history,&Ctl::new(0.0))
}
pub fn replay_controlled(history: &[Point], ctl:&Ctl) -> Result<Board,String> {
    let mut b=Board::new();
    for (n,&p) in history.iter().enumerate() {
        check(ctl)?;
        if p.0.unsigned_abs()>LIMIT as u32 || p.1.unsigned_abs()>LIMIT as u32 || !legal(&b,p) {return Err("illegal history".into());}
        let side=phase(n).0;b.insert(p,side);
        if won(&b,p,side) {return Err("terminal input history".into());}
    }
    Ok(b)
}
fn check(ctl: &Ctl) -> Result<(),String> {
    if ctl.cancel.load(std::sync::atomic::Ordering::Acquire) {Err("verification cancelled".into())}
    else if ctl.expired() {Err("verification deadline".into())} else {Ok(())}
}
pub(crate) fn completions(b:&Board, side:u8, allowance:u8, ctl:&Ctl) -> Result<BTreeSet<Vec<Point>>,String> {
    let mut segments=BTreeSet::new();
    for (&(q,r),&owner) in b {
        check(ctl)?;
        if owner!=side {continue;}
        for (dq,dr) in AXES {for offset in 0..6 {segments.insert((q-offset*dq,r-offset*dr,dq,dr));}}
    }
    let mut out=BTreeSet::new();
    for (q,r,dq,dr) in segments {
        check(ctl)?;
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
pub(crate) fn covers(threats:&BTreeSet<Vec<Point>>, ctl:&Ctl) -> Result<BTreeSet<Vec<Point>>,String> {
    let endpoints:Vec<_>=threats.iter().flatten().copied().collect::<BTreeSet<_>>().into_iter().collect();
    let hits=|p:&[Point]| threats.iter().all(|t|p.iter().any(|x|t.contains(x)));
    let mut result=BTreeSet::new();
    for (i,&a) in endpoints.iter().enumerate() {
        check(ctl)?;
        if hits(&[a]) {result.insert(vec![a]);}
        for &b in &endpoints[i+1..] {check(ctl)?;if hits(&[a,b]) {result.insert(vec![a,b]);}}
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

pub fn defenses(b:&Board, attacker:u8, ctl:&Ctl) -> Result<BTreeMap<Vec<Point>,Vec<Point>>,String> {
    defenses_at(b,attacker,2,ctl)
}
pub fn defenses_at(b:&Board, attacker:u8, remaining:u8, ctl:&Ctl) -> Result<BTreeMap<Vec<Point>,Vec<Point>>,String> {
    if !completions(b,1-attacker,remaining,ctl)?.is_empty() {return Err("defender counterwin".into());}
    let threats=completions(b,attacker,2,ctl)?;
    if threats.is_empty() {return Err("quiet defender unsupported".into());}
    let small=covers(&threats,ctl)?;
    let mut result=BTreeMap::new();
    for cover in small {
        check(ctl)?;
        if cover.len()>remaining as usize {continue;}
        if remaining==1 {result.insert(cover.clone(),cover);continue;}
        if cover.len()==2 {result.insert(cover.clone(),cover);continue;}
        let fixed=cover[0];let mut post=b.clone();post.insert(fixed,1-attacker);
        // Full finite legal frontier AFTER the mandatory block. This includes
        // fillers made legal by that first placement, with that order retained.
        let mut frontier=BTreeSet::new();
        for &(q,r) in post.keys() {
            check(ctl)?;
            for dq in -8i32..=8 {for dr in -8i32..=8 {
                if dq.abs().max(dr.abs()).max((dq+dr).abs())<=8 && !post.contains_key(&(q+dq,r+dr)) {
                    frontier.insert((q+dq,r+dr));
                }
            }}
        }
        for filler in frontier {
            check(ctl)?;
            let mut key=vec![fixed,filler];key.sort();
            result.entry(key).or_insert(vec![fixed,filler]);
            if result.len()>50000 {return Err("free-second coverage work limit".into());}
        }
    }
    Ok(result)
}

/// Check `cert` as a strategy for the side to move at ply `start` on the stones of
/// `history` (`start` is `history.len()`, or `flip(history.len())` for a flipped-turn
/// query). Returns the root action and the most attacker turns on any certificate
/// path, counting the completing turn (an immediate win is 1 turn).
pub fn verify(history:&[Point], start:usize, cert:&ProofCertificate, ctl:&Ctl, max_nodes:usize) -> Result<(Vec<Point>,u32),String> {
    verify_for(history,start,phase(start).0,cert,ctl,max_nodes).map(|(moves,turns,_)|(moves,turns))
}
pub fn verify_for(history:&[Point], start:usize, attacker:u8, cert:&ProofCertificate, ctl:&Ctl, max_nodes:usize) -> Result<(Vec<Point>,u32,BTreeSet<u32>),String> {
    check(ctl)?;
    if start!=history.len() && start!=flip(history.len()) {return Err("invalid certificate root phase".into());}
    let board=replay_controlled(history,ctl)?;
    verify_board(&board,start,attacker,cert,ctl,max_nodes)
}
pub(crate) fn verify_board(board:&Board, start:usize, attacker:u8, cert:&ProofCertificate, ctl:&Ctl, max_nodes:usize) -> Result<(Vec<Point>,u32,BTreeSet<u32>),String> {
    check(ctl)?;
    if cert.version!=1 || cert.width!="wide" || cert.nodes.len()>max_nodes {return Err("certificate format/size".into());}
    struct Checker<'a> {cert:&'a ProofCertificate, attacker:u8, ctl:&'a Ctl, left:usize, stack:BTreeSet<u32>, used:BTreeSet<u32>}
    impl Checker<'_> {
        fn walk(&mut self,id:u32,b:&Board,n:usize) -> Result<u32,String> {
            check(self.ctl)?;
            if self.left==0 || self.stack.len()>=128 || !self.stack.insert(id) {return Err("certificate work limit/cycle/depth".into());}
            self.left-=1;
            let node=self.cert.nodes.get(id as usize).ok_or("invalid certificate edge")?;
            let (side,remaining)=phase(n);
            let turns=match node {
                ProofNode::StampLink{source}=>{
                    if !matches!(self.cert.nodes.get(*source as usize),Some(ProofNode::Stamp{..})) {
                        return Err("stamp link does not name a strategy".into());
                    }
                    self.walk(*source,b,n)?
                }
                ProofNode::ZoneReplies{zone,fallback,responses}=>{
                    if side==self.attacker || !completions(b,side,remaining,self.ctl)?.is_empty() {
                        return Err("zone defender phase or counterwin".into());
                    }
                    let entry=match self.cert.nodes.get(*fallback as usize) {
                        Some(ProofNode::StampLink{source})=>self.cert.nodes.get(*source as usize),
                        entry=>entry,
                    };
                    let source=match entry {
                        Some(ProofNode::Stamp{source})=>source,
                        _=>return Err("zone fallback must be a checked stamp".into()),
                    };
                    let stamp=crate::stamps::remember((**source).clone(),self.ctl)?;
                    let required=stamp.danger(b,remaining,self.ctl)?;
                    let region:BTreeSet<_>=zone.iter().copied().collect();
                    if region.len()!=zone.len() || region.len()>256 || !required.is_subset(&region)
                        || region.iter().any(|p|b.contains_key(p)||p.0.unsigned_abs()>LIMIT as u32||p.1.unsigned_abs()>LIMIT as u32)
                        || responses.len()!=region.len() {return Err("invalid or incomplete proof zone".into());}
                    let mut deepest=self.walk(*fallback,b,crate::stamps::ply(self.attacker,2))?;
                    let mut seen=BTreeSet::new();
                    for r in responses {
                        if r.action.len()!=1 || !region.contains(&r.action[0]) || !seen.insert(r.action[0]) {
                            return Err("missing/duplicate relevant defense".into());
                        }
                        // Overapproximate the defender's reach. If a legal pair
                        // has its relevant stone second, commute it to first.
                        let mut next=b.clone();next.insert(r.action[0],side);
                        if won(&next,r.action[0],side) {return Err("defender wins in zone".into());}
                        deepest=deepest.max(self.walk(r.child,&next,n+1)?);
                    }
                    deepest
                }
                ProofNode::Stamp{source}=>crate::stamps::verify(source,b,n,self.attacker,self.ctl)?,
                ProofNode::Exact{fact,after} => {
                    use hexo_solver::prover::certificate::exact_fact;
                    use hexo_engine::types::Player;
                    let fact=exact_fact(*fact).ok_or("missing exact premise")?;
                    let player=|p| if p==Player::P1 {0} else {1};
                    let mut stones:Board=fact.stones.iter().map(|&(p,s)|(p,player(s))).collect();
                    let mut turn=(player(fact.player),fact.remaining);
                    if !after.is_empty() {
                        if turn.0==self.attacker || after.len()!=turn.1 as usize || n<after.len() || phase(n-after.len())!=turn {
                            return Err("exact premise can continue only the losing defender's turn".into());
                        }
                        let (next,ply,won)=apply(&stones,n-after.len(),after)?;
                        if won {return Err("defender counterwin contradicts premise".into());}
                        stones=next;turn=phase(ply);
                    }
                    if stones!=*b || turn!=(side,remaining) || player(fact.winner)!=self.attacker {
                        return Err("exact premise position/turn/winner mismatch".into());
                    }
                    fact.turns
                }
                ProofNode::ImmediateWin{action} => {
                    if side!=self.attacker || !apply(b,n,action)?.2 {return Err("false immediate win".into());}
                    1
                }
                ProofNode::AttackerMove{action,child,..} => {
                    if side!=self.attacker {return Err("attacker phase mismatch".into());}
                    let (next,ply,terminal)=apply(b,n,action)?;
                    if terminal {return Err("terminal move must use immediate-win leaf".into());}
                    1+self.walk(*child,&next,ply)?
                }
                ProofNode::DefenderReplies{responses} => {
                    if side==self.attacker {return Err("defender phase mismatch".into());}
                    if !completions(b,side,remaining,self.ctl)?.is_empty() {return Err("defender counterwin".into());}
                    let required=defenses_at(b,self.attacker,remaining,self.ctl)?;
                    if required.is_empty() {return Err("defenses supplied for unstoppable position".into());}
                    if responses.len()!=required.len() {return Err("missing defense branch including free-second coverage".into());}
                    let mut seen=BTreeSet::new();let mut deepest=0;
                    for reply in responses {
                        let mut key=reply.action.clone();key.sort();
                        if !required.contains_key(&key) || !seen.insert(key) {return Err("invalid/duplicate defense".into());}
                        let (next,ply,terminal)=apply(b,n,&reply.action)?;
                        if terminal {return Err("defender wins".into());}
                        deepest=deepest.max(self.walk(reply.child,&next,ply)?);
                    }
                    if seen!=required.keys().cloned().collect() {return Err("missing defense branch including free-second coverage".into());}
                    deepest
                }
                ProofNode::Unstoppable{..} => {
                    if side==self.attacker {return Err("unstoppable phase mismatch".into());}
                    if !completions(b,side,remaining,self.ctl)?.is_empty() {return Err("defender counterwin".into());}
                    let threats=completions(b,self.attacker,2,self.ctl)?;
                    if threats.is_empty() || covers(&threats,self.ctl)?.iter().any(|c|c.len()<=remaining as usize) {return Err("false unstoppable".into());}
                    1
                }
            };
            self.stack.remove(&id);self.used.insert(id);Ok(turns)
        }
    }
    let mut checker=Checker{cert,attacker,ctl,left:max_nodes,stack:BTreeSet::new(),used:BTreeSet::new()};
    let turns=checker.walk(cert.root,board,start)?;
    check(ctl)?;
    match &cert.nodes[cert.root as usize] {
        ProofNode::ImmediateWin{action}|ProofNode::AttackerMove{action,..}=>Ok((action.clone(),turns,checker.used)),
        ProofNode::Stamp{source}=>Ok((crate::stamps::moves(source,board,start)?,turns,checker.used)),
        _ if phase(start).0!=attacker=>Ok((vec![],turns,checker.used)),
        ProofNode::Exact{..}=>Ok((vec![],turns,checker.used)),
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
            let replies=defenses(&b,0,&Ctl::new(1.0)).unwrap();
            let mut key=vec![fixed,filler];key.sort();
            assert_eq!(replies.get(&key),Some(&vec![fixed,filler]));
            b.insert(fixed,1);assert!(legal(&b,filler));
        }
    }
}
