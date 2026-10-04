//! Local consequences of an ordinary, all-defence proof. A stamp preserves the
//! attack's windows and every played cell. Guards cover all six-cell windows a
//! future defender stone can affect; a global check covers the remaining ones.
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::rc::Rc;
use hexo_engine::types::{Coord, Player};
use hexo_solver::prover::Ctl;
use hexo_solver::prover::certificate::{ProofCertificate, ProofNode, StampOracle, StampSource, StoneAt};
use crate::check::{self, Board};

const AXES:[Coord;3]=[(1,0),(0,1),(1,-1)];
const MAX_BYTES:usize=4*1024*1024;
const MAX_STAMPS:usize=32;
thread_local! {static TIMINGS:RefCell<(bool,BTreeMap<&'static str,(u64,f64)>)>=RefCell::new((false,BTreeMap::new()));}
pub struct Timing(&'static str,Option<std::time::Instant>);
impl Drop for Timing {fn drop(&mut self){if let Some(start)=self.1 {TIMINGS.with(|t|{
    let mut t=t.borrow_mut();let row=t.1.entry(self.0).or_default();row.0+=1;row.1+=start.elapsed().as_secs_f64()*1000.;
});}}}
pub fn measure(name:&'static str)->Timing {Timing(name,TIMINGS.with(|t|t.borrow().0.then(std::time::Instant::now)))}
pub fn reset_timings(enabled:bool) {TIMINGS.with(|t|*t.borrow_mut()=(enabled,BTreeMap::new()));}
pub fn timings()->serde_json::Value {TIMINGS.with(|t|serde_json::to_value(&t.borrow().1).unwrap())}
type Window=(i32,i32,u8);
fn cells(w:Window)->[Coord;6] { let (q,r,axis)=w;let (dq,dr)=AXES[axis as usize];std::array::from_fn(|i|(q+i as i32*dq,r+i as i32*dr)) }
fn windows(p:Coord)->impl Iterator<Item=Window> {
    AXES.into_iter().enumerate().flat_map(move |(axis,(dq,dr))|(0..6).map(move |i|(p.0-i*dq,p.1-i*dr,axis as u8)))
}
fn side(p:Player)->u8 {if p==Player::P1 {0}else{1}}
fn player(s:u8)->Player {if s==0 {Player::P1}else{Player::P2}}
fn rotate(mut p:Coord,sym:u8)->Coord {
    if sym>=6 {p=(p.1,p.0);}
    for _ in 0..sym%6 {p=(-p.1,p.0+p.1);} p
}
fn transform(p:Coord,sym:u8,offset:Coord)->Coord {let p=rotate(p,sym);(p.0+offset.0,p.1+offset.1)}
fn inverse(p:Coord,sym:u8,offset:Coord)->Coord {
    let p=(p.0-offset.0,p.1-offset.1);let p=rotate(p,(6-sym%6)%6);
    if sym>=6 {(p.1,p.0)}else{p}
}
fn transformed_source(source:&StampSource,sym:u8,offset:Coord,swap:bool)->StampSource {
    let mut out=source.clone();
    let point=|p|transform(p,sym,offset);
    for (p,s) in &mut out.stones {*p=point(*p);*s^=u8::from(swap);}
    out.stones.sort();out.player^=u8::from(swap);out.winner^=u8::from(swap);
    for node in &mut out.certificate.nodes {match node {
        ProofNode::ImmediateWin{action}|ProofNode::AttackerMove{action,..}=>{
            for p in action {*p=point(*p);}
            if let ProofNode::AttackerMove{alternatives,..}=node {for r in alternatives {for p in &mut r.action {*p=point(*p);}}}
        }
        ProofNode::DefenderReplies{responses}=>for r in responses {for p in &mut r.action {*p=point(*p);}},
        ProofNode::Unstoppable{threats}=>for t in threats {for p in t {*p=point(*p);}},
        ProofNode::Stamp{source}=>**source=transformed_source(source,sym,offset,swap),
        _=>{},
    }} out
}
pub fn ply(p:u8,remaining:u8)->usize {if p==0 {5-remaining as usize}else{3-remaining as usize}}
fn control(ctl:&Ctl)->Result<(),String> {if ctl.expired(){Err("stamp deadline or cancellation".into())}else{Ok(())}}

#[derive(Clone)]
pub struct Stamp {
    pub portable:Cell<bool>,
    pub source:StampSource,
    pub turns:u32,
    pub required:BTreeSet<Coord>,
    pub empty:BTreeSet<Coord>,
    guards:BTreeMap<Window,i8>,
    before:BTreeSet<Coord>,
    allowance:u8,
    bytes:usize,
}

impl Stamp {
    pub fn compile(mut source:StampSource,ctl:&Ctl)->Result<Self,String> {
        let _time=measure("compile");
        control(ctl)?;
        if source.player>1 || source.winner>1 || !(1..=2).contains(&source.remaining)
            || source.stones.is_empty() || source.stones.len()>800 || source.certificate.nodes.len()>50000
            || source.certificate.nodes.iter().any(|n|matches!(n,ProofNode::Exact{..}|ProofNode::ZoneReplies{..})) {
            return Err("stamp requires an ordinary complete strategy".into());
        }
        let root:Board=source.stones.iter().copied().collect();
        if root.len()!=source.stones.len() || root.iter().any(|(&p,&s)|s>1 || p.0.unsigned_abs()>check::LIMIT as u32
            || p.1.unsigned_abs()>check::LIMIT as u32 || check::won(&root,p,s)) {return Err("invalid stamp position".into());}
        let start=ply(source.player,source.remaining);
        check::verify_board(&root,start,source.winner,&source.certificate,ctl,50000)?;
        if source.certificate.nodes.iter().any(|n|matches!(n,ProofNode::Stamp{..}|ProofNode::StampLink{..})) {
            source.certificate=materialize(&source.certificate,&root,start,source.winner,ctl)?;
        }
        let (_,turns,used)=check::verify_board(&root,start,source.winner,&source.certificate,ctl,50000)?;
        if used.len()>4096 {return Err("stamp strategy size limit".into());}
        let indices:BTreeMap<_,_>=used.iter().enumerate().map(|(i,&id)|(id,i as u32)).collect();
        source.certificate.nodes=used.iter().map(|&id| {
            let mut node=source.certificate.nodes[id as usize].clone();
            match &mut node {
                ProofNode::AttackerMove{child,alternatives,..}=>{*child=indices[child];alternatives.clear();}
                ProofNode::DefenderReplies{responses}=>for r in responses {r.child=indices[&r.child];},
                _=>{},
            } node
        }).collect();
        source.certificate.root=indices[&source.certificate.root];
        let bytes=serde_json::to_vec(&source).map_err(|e|e.to_string())?.len();
        if bytes>MAX_BYTES/2 {return Err("stamp size limit".into());}
        let mut stamp=Self{portable:Cell::new(false),source,turns,required:BTreeSet::new(),empty:BTreeSet::new(),guards:BTreeMap::new(),
            before:BTreeSet::new(),allowance:0,bytes};
        let cert=stamp.source.certificate.clone();
        let start=ply(stamp.source.player,stamp.source.remaining);
        stamp.collect(&root,&root,start,cert.root,&cert,&mut 50000,ctl)?;
        if stamp.required.is_empty() || stamp.empty.len()>4096 || stamp.guards.len()>8192 {return Err("stamp footprint limit".into());}
        // Discard the rest of the original game. Replaying the strategy with
        // only its supporting stones also removes now-obsolete defense branches.
        if stamp.required.len()<stamp.source.stones.len() {
            let bare:Board=stamp.required.iter().map(|&p|(p,stamp.source.winner)).collect();
            if let Ok(certificate)=materialize(&stamp.source.certificate,&bare,start,stamp.source.winner,ctl) {
                let source=StampSource{stones:bare.into_iter().collect(),certificate,..stamp.source.clone()};
                if let Ok(reduced)=remember(source,ctl) {return Ok((*reduced).clone());}
            }
        }
        stamp.bytes=2*stamp.bytes+64*(stamp.required.len()+stamp.empty.len()+stamp.before.len()+stamp.guards.len())
            +std::mem::size_of::<Stamp>();
        if stamp.bytes>MAX_BYTES/2 {return Err("compiled stamp size limit".into());}
        Ok(stamp)
    }

    fn preserve(&mut self,root:&Board,points:impl IntoIterator<Item=Coord>) {
        for p in points {match root.get(&p) {
            Some(&s) if s==self.source.winner=>{self.required.insert(p);},
            None=>{self.empty.insert(p);},
            _=>{},
        }}
    }
    fn collect(&mut self,root:&Board,b:&Board,n:usize,id:u32,cert:&ProofCertificate,left:&mut usize,ctl:&Ctl)->Result<(),String> {
        control(ctl)?;
        if *left==0 {return Err("stamp work limit".into());} *left-=1;
        let (mover,remaining)=check::phase(n);let winner=self.source.winner;
        match &cert.nodes[id as usize] {
            ProofNode::ImmediateWin{action}|ProofNode::AttackerMove{action,..}=>{
                let mut post=b.clone();
                for &p in action {
                    // Legality must remain true without relying on irrelevant
                    // enemy stones. A friendly anchor is at most eight away.
                    let anchor=post.iter().find(|&(&a,&s)|s==winner && distance(a,p)<=8).map(|(&a,_)|a)
                        .ok_or("stamp move needs an enemy reach anchor")?;
                    self.preserve(root,[anchor,p]);post.insert(p,winner);
                }
                if let ProofNode::AttackerMove{child,..}=&cert.nodes[id as usize] {
                    self.collect(root,&post,n+action.len(),*child,cert,left,ctl)?;
                } else {
                    let line=post.iter().filter(|&(_, &s)|s==winner).flat_map(|(&p,_)|windows(p))
                        .find(|&w|cells(w).iter().all(|p|post.get(p)==Some(&winner))).ok_or("stamp win has no six")?;
                    self.preserve(root,cells(line));
                }
            }
            ProofNode::DefenderReplies{..}|ProofNode::Unstoppable{..}=>{
                if mover==winner {return Err("stamp defender phase".into());}
                let mut threats:BTreeMap<Vec<Coord>,Window>=BTreeMap::new();
                for (&p,&s) in b {if s==winner {for w in windows(p) {
                    let points=cells(w);
                    if points.iter().any(|p|b.get(p)==Some(&(1-winner))) {continue;}
                    let mut empty:Vec<_>=points.into_iter().filter(|p|!b.contains_key(p)).collect();empty.sort();
                    if !empty.is_empty() && empty.len()<=2 {threats.entry(empty).or_insert(w);}
                }}}
                let sets=threats.keys().cloned().collect();
                if check::covers(&sets,ctl)?.iter().any(|c|c.len()<remaining as usize) {
                    return Err("stamp strategy gives a defender a free stone".into());
                }
                for w in threats.into_values() {self.preserve(root,cells(w));}
                let added:BTreeSet<_>=b.iter().filter(|&(p,&s)|s==winner && !root.contains_key(p)).map(|(&p,_)|p).collect();
                if self.allowance==0 {self.before=added.clone();} else {self.before=self.before.intersection(&added).copied().collect();}
                self.allowance=self.allowance.max(remaining);
                let extra:BTreeSet<_>=b.iter().filter(|&(p,&s)|s!=winner && !root.contains_key(p)).map(|(&p,_)|p).collect();
                for w in extra.iter().copied().flat_map(windows) {
                    let points=cells(w);if points.iter().any(|p|added.contains(p)) {continue;}
                    let count=points.iter().filter(|p|extra.contains(p)).count() as i8;
                    let limit=5-remaining as i8-count;
                    self.guards.entry(w).and_modify(|n|*n=(*n).min(limit)).or_insert(limit);
                }
                if let ProofNode::DefenderReplies{responses}=&cert.nodes[id as usize] {
                    for reply in responses {
                        self.preserve(root,reply.action.iter().copied());
                        let (post,ply,_)=check::apply(b,n,&reply.action)?;
                        self.collect(root,&post,ply,reply.child,cert,left,ctl)?;
                    }
                }
            }
            _=>return Err("stamp source is not independent".into()),
        }
        Ok(())
    }

    pub fn matches(&self,get:StoneAt<'_>,stones:&[(Coord,Player)],mover:Player,remaining:u8)->bool {
        let _time=measure("match");
        if side(mover)!=self.source.player || remaining!=self.source.remaining {return false;}
        let winner=player(self.source.winner);
        if self.required.iter().any(|&p|get(p)!=Some(winner)) || self.empty.iter().any(|&p|get(p).is_some()) {return false;}
        if self.allowance>0 {
            // Windows unaffected by any later defensive placement can only
            // become safer. Check them once, after the common attacker prefix.
            for &(p,s) in stones {if s!=winner {for w in windows(p) {
                let points=cells(w);
                if points.iter().any(|p|self.before.contains(p)||get(*p)==Some(winner)) {continue;}
                if points.iter().filter(|&&p|get(p)==Some(s)).count()+self.allowance as usize>=6 {return false;}
            }}}
            for (&w,&limit) in &self.guards {
                let points=cells(w);
                if !points.iter().any(|&p|get(p)==Some(winner))
                    && points.iter().filter(|&&p|get(p)==Some(winner.opponent())).count() as i8>limit {return false;}
            }
        }
        true
    }

    pub fn key(&self)->String {
        // Coordinates of the supporting stones, checked empty cells and every
        // counter-threat guard belong to the key, together with the real phase.
        (0..12).map(|sym| {
            let anchor=self.required.iter().map(|&p|rotate(p,sym)).min().unwrap();
            let point=|p|transform(p,sym,(-anchor.0,-anchor.1));
            let ordered=|s:&BTreeSet<Coord>|s.iter().copied().map(point).collect::<BTreeSet<_>>();
            let guards:BTreeSet<_>=self.guards.iter().map(|(&w,&limit)|{
                let set:BTreeSet<_>=cells(w).into_iter().map(point).collect();(set,limit)
            }).collect();
            serde_json::to_string(&(ordered(&self.required),ordered(&self.empty),guards,ordered(&self.before),
                self.allowance,self.source.player==self.source.winner,self.source.remaining)).unwrap()
        }).min().unwrap()
    }

    /// Every cell where up to `spares` enemy stones can invalidate this strategy.
    /// No radius assumption: this includes remote counter-threat windows.
    pub fn danger(&self,b:&Board,spares:u8,ctl:&Ctl)->Result<BTreeSet<Coord>,String> {
        control(ctl)?;
        if !(1..=2).contains(&spares) {return Err("invalid defender tempo".into());}
        let winner=self.source.winner;
        let stones:Vec<_>=b.iter().map(|(&p,&s)|(p,player(s))).collect();
        if !self.matches(&|p|b.get(&p).copied().map(player),&stones,player(winner),2) {
            return Err("fallback stamp does not prove a fresh attacker turn".into());
        }
        let mut danger=self.empty.clone();
        let mut guard=|w:Window,limit:i8,before:&BTreeSet<Coord>| {
            let points=cells(w);
            if points.iter().any(|p|b.get(p)==Some(&winner)||before.contains(p)) {return;}
            let count=points.iter().filter(|p|b.get(p)==Some(&(1-winner))).count() as i8;
            if count+spares as i8>limit {danger.extend(points.into_iter().filter(|p|!b.contains_key(p)));}
        };
        if self.allowance>0 {
            for (&p,&s) in b {control(ctl)?;if s!=winner {for w in windows(p) {guard(w,5-self.allowance as i8,&self.before);}}}
            for (&w,&limit) in &self.guards {guard(w,limit,&BTreeSet::new());}
        }
        // Two premoves and a two-stone winning turn need at least two existing
        // enemy stones, so all possible remote wins appear in these windows.
        Ok(danger)
    }
}
fn distance(a:Coord,b:Coord)->i64 {let (q,r)=(i64::from(a.0)-i64::from(b.0),i64::from(a.1)-i64::from(b.1));q.abs().max(r.abs()).max((q+r).abs())}

/// Save the actual primary strategy, not ever-growing chains of earlier sources.
/// Recompute the response set: extra friendly stones may end a branch earlier.
fn materialize(cert:&ProofCertificate,b:&Board,n:usize,winner:u8,ctl:&Ctl)->Result<ProofCertificate,String> {
    fn walk(out:&mut ProofCertificate,cert:&ProofCertificate,id:u32,b:&Board,n:usize,winner:u8,ctl:&Ctl,depth:usize)->Result<u32,String> {
        control(ctl)?;
        if depth>128 || out.nodes.len()>=4096 {return Err("expanded stamp size/depth limit".into());}
        let node=match &cert.nodes[id as usize] {
            ProofNode::StampLink{source}=>return walk(out,cert,*source,b,n,winner,ctl,depth+1),
            ProofNode::Stamp{source}=>return walk(out,&source.certificate,source.certificate.root,b,n,winner,ctl,depth+1),
            ProofNode::ImmediateWin{action}|ProofNode::AttackerMove{action,..}=>{
                let mut post=b.clone();let mut moves=vec![];let mut terminal=false;
                for &p in action {
                    if !check::legal(&post,p) {return Err("illegal reused attack".into());}
                    post.insert(p,winner);moves.push(p);
                    if check::won(&post,p,winner) {terminal=true;break;}
                }
                if terminal {ProofNode::ImmediateWin{action:moves}} else if let ProofNode::AttackerMove{child,..}=&cert.nodes[id as usize] {
                    let child=walk(out,cert,*child,&post,n+moves.len(),winner,ctl,depth+1)?;
                    ProofNode::AttackerMove{action:moves,child,alternatives:vec![]}
                } else {return Err("reused win disappeared".into());}
            }
            ProofNode::DefenderReplies{responses}=>{
                let required=check::defenses_at(b,winner,check::phase(n).1,ctl)?;
                let mut kept=vec![];
                for (key,action) in required {
                    let reply=responses.iter().find(|r|{let mut k=r.action.clone();k.sort();k==key}).ok_or("reused defense not covered")?;
                    let (post,ply,terminal)=check::apply(b,n,&action)?;
                    if terminal {return Err("reused defense wins".into());}
                    kept.push(hexo_solver::prover::certificate::ProofResponse{action,child:walk(out,cert,reply.child,&post,ply,winner,ctl,depth+1)?});
                }
                if kept.is_empty() {ProofNode::Unstoppable{threats:check::completions(b,winner,2,ctl)?.into_iter().collect()}} else {ProofNode::DefenderReplies{responses:kept}}
            }
            ProofNode::Unstoppable{..}=>ProofNode::Unstoppable{threats:check::completions(b,winner,2,ctl)?.into_iter().collect()},
            _=>return Err("stamp cannot import conditional premises".into()),
        };
        let id=out.nodes.len() as u32;out.nodes.push(node);Ok(id)
    }
    let mut out=ProofCertificate{version:1,width:"wide".into(),root:0,nodes:vec![]};
    out.root=walk(&mut out,cert,cert.root,b,n,winner,ctl,0)?;Ok(out)
}

thread_local! {
    static LIBRARY:RefCell<Vec<Rc<Stamp>>>=const{RefCell::new(Vec::new())};
    static DEPTH:Cell<u8>=const{Cell::new(0)};
    static SEEDED:Cell<bool>=const{Cell::new(false)};
}
pub fn seed(ctl:&Ctl)->Result<(),String> {
    if SEEDED.with(Cell::get) {return Ok(());}
    #[derive(serde::Deserialize)] struct Primitive {source:StampSource}
    let entries:Vec<Primitive>=serde_json::from_str(include_str!("../stamps.json")).map_err(|e|e.to_string())?;
    for entry in entries {remember(entry.source,ctl)?.portable.set(true);}
    SEEDED.with(|s|s.set(true));Ok(())
}
struct CompileDepth;
impl Drop for CompileDepth {fn drop(&mut self){DEPTH.with(|n|n.set(n.get()-1));}}
pub fn remember(source:StampSource,ctl:&Ctl)->Result<Rc<Stamp>,String> {
    let _time=measure("remember");
    control(ctl)?;
    if let Some(stamp)=LIBRARY.with(|l|l.borrow().iter().find(|s|s.source==source).cloned()) {return Ok(stamp);}
    DEPTH.with(|n|if n.get()>=8 {Err("nested stamp limit")} else {n.set(n.get()+1);Ok(())})?;
    let _depth=CompileDepth;
    let stamp=Rc::new(Stamp::compile(source,ctl)?);
    if let Some(prior)=LIBRARY.with(|l|l.borrow().iter().find(|s|s.source==stamp.source).cloned()) {return Ok(prior);}
    LIBRARY.with(|l| {
        let mut list=l.borrow_mut();
        while !list.is_empty() && (list.len()>=MAX_STAMPS || list.iter().map(|s|s.bytes).sum::<usize>()+stamp.bytes>MAX_BYTES) {
            let evict=list.iter().position(|s|!s.portable.get()).unwrap_or(0);list.remove(evict);
        }
        list.push(stamp.clone());
    });
    Ok(stamp)
}
pub fn prune(board:&Board) {
    // Small primitives remain relocatable. Larger, game-local strategies whose
    // fixed cells are occupied incorrectly cannot revive on this branch.
    LIBRARY.with(|l|l.borrow_mut().retain(|s|s.portable.get() || s.required.len()<=4 ||
        !(s.empty.iter().any(|p|board.contains_key(p)) || s.required.iter().any(|p|board.get(p).is_some_and(|&side|side!=s.source.winner)))));
}
pub fn stats()->(usize,usize) {LIBRARY.with(|l|{let l=l.borrow();(l.len(),l.iter().map(|s|s.bytes).sum())})}
pub fn verify(source:&StampSource,b:&Board,n:usize,winner:u8,ctl:&Ctl)->Result<u32,String> {
    let _time=measure("verify");
    if source.winner!=winner || check::phase(n)!=(source.player,source.remaining) {return Err("stamp winner/tempo mismatch".into());}
    let stamp=remember(source.clone(),ctl)?;let (mover,remaining)=check::phase(n);
    let stones:Vec<_>=b.iter().map(|(&p,&s)|(p,player(s))).collect();
    if !stamp.matches(&|p|b.get(&p).copied().map(player),&stones,player(mover),remaining) {
        // A rotated source can choose a different witness window when its mask
        // is re-derived. Conservative mask failure is not a refutation: replay
        // and check the actual strategy, never accept the source's verdict alone.
        let strategy=materialize(&source.certificate,b,n,winner,ctl)?;
        return check::verify_board(b,n,winner,&strategy,ctl,50000).map(|(_,turns,_)|turns);
    }
    Ok(stamp.turns)
}
pub fn moves(source:&StampSource,b:&Board,n:usize)->Result<Vec<Coord>,String> {
    if check::phase(n).0!=source.winner {return Ok(vec![]);}
    let action=match &source.certificate.nodes[source.certificate.root as usize] {
        ProofNode::ImmediateWin{action}|ProofNode::AttackerMove{action,..}=>action,
        _=>return Err("stamp has no winning root action".into()),
    };
    let mut board=b.clone();let mut moves=vec![];
    for &p in action {
        if !check::legal(&board,p) {return Err("stamp root action is illegal".into());}
        board.insert(p,source.winner);moves.push(p);
        if check::won(&board,p,source.winner) {break;}
    }
    Ok(moves)
}

type Pattern=(Vec<Coord>,Vec<(usize,u8,Coord)>);
fn locate(patterns:&[Pattern],entries:&[Rc<Stamp>],stones:&[(Coord,Player)],get:StoneAt<'_>,
    candidates:&mut [BTreeSet<(u8,Coord,bool)>],ctl:&Ctl) {
    for &(at,owner) in stones {
        if ctl.expired() {return;}
        for (offsets,uses) in patterns {
            if offsets.iter().any(|&(q,r)|get((at.0+q,at.1+r))!=Some(owner)) {continue;}
            for &(id,sym,anchor) in uses {if candidates[id].len()<512 {
                candidates[id].insert((sym,(at.0-anchor.0,at.1-anchor.1),side(owner)!=entries[id].source.winner));
            }}
        }
    }
}

pub struct Oracle {entries:Vec<Rc<Stamp>>, instances:RefCell<Vec<(usize,u8,Coord,bool)>>,
    base_len:usize, candidates:RefCell<Vec<BTreeSet<(u8,Coord,bool)>>>, patterns:Vec<Pattern>,
    misses:RefCell<HashSet<(u64,u8,u8)>>, pub hits:Cell<u64>,ctl:Ctl}
impl Oracle {
    pub fn new(ctl:&Ctl,root:&Board)->Rc<Self> {
        let _time=measure("index");
        let mut entries=LIBRARY.with(|l|l.borrow().clone());
        entries.sort_by_key(|s|(s.empty.len(),s.turns));
        // Many proofs have the same supporting shape but different strategies.
        // Match their geometry once; only successful shapes visit their masks.
        let mut patterns:BTreeMap<Vec<Coord>,Vec<(usize,u8,Coord)>>=BTreeMap::new();
        for (id,stamp) in entries.iter().enumerate() {if stamp.required.len()<=4 {for sym in 0..12 {
            let rotated:Vec<_>=stamp.required.iter().map(|&p|rotate(p,sym)).collect();
            for &anchor in &rotated {
                let mut offsets:Vec<_>=rotated.iter().filter(|&&p|p!=anchor).map(|p|(p.0-anchor.0,p.1-anchor.1)).collect();offsets.sort();
                patterns.entry(offsets).or_default().push((id,sym,anchor));
            }
        }}}
        let patterns:Vec<_>=patterns.into_iter().collect();
        let mut candidates=vec![BTreeSet::new();entries.len()];
        let stones:Vec<_>=root.iter().map(|(&p,&s)|(p,player(s))).collect();
        locate(&patterns,&entries,&stones,&|p|root.get(&p).copied().map(player),&mut candidates,ctl);
        for (id,found) in candidates.iter_mut().enumerate() {
            found.retain(|&(sym,offset,_)|entries[id].empty.iter().all(|&p|!root.contains_key(&transform(p,sym,offset))));
        }
        let patterns=patterns.into_iter().filter_map(|(offsets,uses)| {
            let uses:Vec<_>=uses.into_iter().filter(|&(id,_,_)|entries[id].portable.get()).collect();
            (!uses.is_empty()).then_some((offsets,uses))
        }).collect();
        Rc::new(Self{entries,candidates:RefCell::new(candidates),patterns,base_len:root.len(),instances:RefCell::new(vec![]),
            misses:RefCell::new(HashSet::new()),hits:Cell::new(0),ctl:ctl.clone()})
    }
}
impl StampOracle for Oracle {
    fn lookup(&self,hash:u64,stones:&[(Coord,Player)],get:StoneAt<'_>,mover:Player,remaining:u8)->Option<(usize,Player,u32)> {
        let _time=measure("lookup");
        if self.ctl.expired() || self.misses.borrow().contains(&(hash,side(mover),remaining)) {return None;}
        for (id,stamp) in self.entries.iter().enumerate() {if stamp.matches(get,stones,mover,remaining) {
            self.hits.set(self.hits.get()+1);return Some((id,player(stamp.source.winner),stamp.turns));
        }}
        // Small primitive shapes can occur anywhere and in either colour.
        // A cheap supporting-stone/mask match precedes the global threat guards.
        // SolverBoard preserves the query's root stones as a prefix. Every
        // searched edge adds at most two stones; an occurrence created there
        // contains one of them. Retain candidate geometry across branches and
        // recheck its full mask on every use. An unusual jump can only miss an
        // optimization, never establish a proof without that check.
        let added=&stones[self.base_len.min(stones.len()).max(stones.len().saturating_sub(2))..];
        let mut candidates=self.candidates.borrow_mut();
        locate(&self.patterns,&self.entries,added,get,&mut candidates,&self.ctl);
        for (id,stamp) in self.entries.iter().enumerate() {
            if stamp.required.len()>4 || stamp.source.remaining!=remaining {continue;}
            let swap=side(mover)!=stamp.source.player;
            let winner=player(stamp.source.winner^u8::from(swap));
            // Every new occurrence contains a newly played stone; `locate`
            // tried each supporting cell as its anchor without rescanning root stones.
            for &(sym,offset,colour) in &candidates[id] {
                if colour!=swap {continue;}
                let mapped=|p|get(transform(p,sym,offset)).map(|s|if swap{s.opponent()}else{s});
                if stamp.required.iter().any(|&p|mapped(p)!=Some(player(stamp.source.winner)))
                    || stamp.empty.iter().any(|&p|mapped(p).is_some()) {continue;}
                let local:Vec<_>=stones.iter().map(|&(p,s)|(inverse(p,sym,offset),if swap{s.opponent()}else{s})).collect();
                if !stamp.matches(&mapped,&local,player(stamp.source.player),remaining) {continue;}
                let key=(id,sym,offset,swap);let mut list=self.instances.borrow_mut();
                let index=if let Some(i)=list.iter().position(|k|*k==key) {i} else {
                    if list.len()>=64 {continue;}let i=list.len();list.push(key);i
                };
                self.hits.set(self.hits.get()+1);return Some((self.entries.len()+index,winner,stamp.turns));
            }
        }
        let mut misses=self.misses.borrow_mut();if misses.len()>=16384 {misses.clear();}misses.insert((hash,side(mover),remaining));None
    }
    fn source(&self,id:usize)->StampSource {
        if id<self.entries.len() {return self.entries[id].source.clone();}
        let (index,sym,offset,swap)=self.instances.borrow()[id-self.entries.len()];
        transformed_source(&self.entries[index].source,sym,offset,swap)
    }
    fn verify(&self,source:&StampSource,get:StoneAt<'_>,stones:&[(Coord,Player)],mover:Player,remaining:u8)->Option<u32> {
        let stamp=remember(source.clone(),&self.ctl).ok()?;
        if stamp.matches(get,stones,mover,remaining) {return Some(stamp.turns);}
        let board=stones.iter().map(|&(p,s)|(p,side(s))).collect();
        verify(source,&board,ply(side(mover),remaining),source.winner,&self.ctl).ok()
    }
}
