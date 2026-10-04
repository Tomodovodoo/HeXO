//! Local consequences of an ordinary, all-defence proof. A stamp preserves the
//! attack's windows and every played cell. Guards cover all six-cell windows a
//! future defender stone can affect; a global check covers the remaining ones.
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::rc::{Rc,Weak};
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

/// Share identical suffixes of a checked primary strategy. Every defense stays
/// present; the checker still visits a shared node in each board context.
fn compact(cert:&ProofCertificate,ctl:&Ctl)->Result<ProofCertificate,String> {
    struct Build<'a> {cert:&'a ProofCertificate,ctl:&'a Ctl,nodes:Vec<ProofNode>,old:BTreeMap<u32,u32>,same:BTreeMap<Vec<u8>,u32>}
    impl Build<'_> {
        fn visit(&mut self,id:u32)->Result<u32,String> {
            control(self.ctl)?;
            if let Some(&new)=self.old.get(&id) {return Ok(new);}
            let mut node=self.cert.nodes[id as usize].clone();
            match &mut node {
                ProofNode::AttackerMove{child,alternatives,..}=>{alternatives.clear();*child=self.visit(*child)?;},
                ProofNode::DefenderReplies{responses}=>for reply in responses {reply.child=self.visit(reply.child)?;},
                ProofNode::ImmediateWin{..}|ProofNode::Unstoppable{..}=>{},
                _=>return Err("stamp source is not independent".into()),
            }
            let key=serde_json::to_vec(&node).map_err(|e|e.to_string())?;
            let new=if let Some(&same)=self.same.get(&key) {same} else {
                let new=self.nodes.len() as u32;self.nodes.push(node);self.same.insert(key,new);new
            };
            self.old.insert(id,new);Ok(new)
        }
    }
    let mut build=Build{cert,ctl,nodes:vec![],old:BTreeMap::new(),same:BTreeMap::new()};
    let root=build.visit(cert.root)?;
    Ok(ProofCertificate{root,nodes:build.nodes,..cert.clone()})
}

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
        source.certificate=compact(&source.certificate,ctl)?;
        let (_,turns,used)=check::verify_board(&root,start,source.winner,&source.certificate,ctl,50000)?;
        if used.len()>4096 {return Err("stamp strategy size limit".into());}
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
                if let Ok(reduced)=remember(source,ctl) {
                    let stones:Vec<_>=root.iter().map(|(&p,&s)|(p,player(s))).collect();
                    if reduced.matches(&|p|root.get(&p).copied().map(player),&stones,player(stamp.source.player),stamp.source.remaining) {
                        return Ok((*reduced).clone());
                    }
                }
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

    /// A sufficient implication between the predicates in `matches`, including
    /// the win bound and bounded discovery. Distinct fixed frames remain: the
    /// geometry/instance caps can prevent rediscovery of a discarded frame.
    fn dominates(&self,other:&Self,ctl:&Ctl)->Result<bool,String> {
        control(ctl)?;
        if self.source.remaining!=other.source.remaining || self.source.player!=other.source.player
            || self.source.winner!=other.source.winner
            || self.turns>other.turns || self.required.len()>other.required.len() || self.empty.len()>other.empty.len()
            || (self.allowance>0 && (other.allowance==0 || self.allowance>other.allowance)) {return Ok(false);}
        // A relocatable source also had capped candidates outside its fixed
        // frame. Preserve their geometry and, for incremental discovery, the
        // root mask filtering that leaves capacity for later occurrences.
        if other.required.len()<=4 && (self.required!=other.required || (other.portable.get() && self.empty!=other.empty)) {
            return Ok(false);
        }
        Ok(self.dominates_here(other))
    }
    fn dominates_here(&self,other:&Self)->bool {
        if !self.required.is_subset(&other.required) || !self.empty.is_subset(&other.empty) {return false;}
        if self.allowance==0 {return true;}
        // Every window exempted from the other's global counter-threat test
        // must also be exempted from ours. Its lower allowance is no stricter.
        if !other.before.is_subset(&self.before) {return false;}
        self.guards.iter().all(|(&w,&limit)| {
            let points=cells(w);
            points.iter().any(|p|other.required.contains(p))
                || other.guards.get(&w).is_some_and(|&bound|bound<=limit)
                || (!points.iter().any(|p|other.before.contains(p)) && 5-other.allowance as i8<=limit)
        })
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
    static IMPORTS:RefCell<Vec<(Vec<u8>,Weak<Stamp>)>>=const{RefCell::new(Vec::new())};
    static DEPTH:Cell<u8>=const{Cell::new(0)};
    static SEEDED:Cell<usize>=const{Cell::new(0)};
}
pub fn seed(ctl:&Ctl)->Result<(),String> {
    if SEEDED.with(Cell::get)==usize::MAX {return Ok(());}
    #[derive(serde::Deserialize)] struct Primitive {source:StampSource}
    let entries:Vec<Primitive>=serde_json::from_str(include_str!("../stamps.json")).map_err(|e|e.to_string())?;
    // A leaf query may expire between entries. Keep completed imports rather
    // than recompiling their original, unshared certificates next slice.
    for (i,entry) in entries.into_iter().enumerate().skip(SEEDED.with(Cell::get)) {
        remember_as(entry.source,ctl,true)?;
        SEEDED.with(|s|s.set(i+1));
    }
    SEEDED.with(|s|s.set(usize::MAX));Ok(())
}
struct CompileDepth;
impl Drop for CompileDepth {fn drop(&mut self){DEPTH.with(|n|n.set(n.get()-1));}}
pub fn remember(source:StampSource,ctl:&Ctl)->Result<Rc<Stamp>,String> {
    remember_as(source,ctl,false)
}
fn import_bytes()->usize {IMPORTS.with(|i|i.borrow().iter().map(|(key,_)|key.len()+64).sum())}
fn trim_imports(room:usize) {
    IMPORTS.with(|i| {
        let mut list=i.borrow_mut();list.retain(|(_,s)|s.strong_count()>0);
        while !list.is_empty() && (list.len()>=MAX_STAMPS || list.iter().map(|(k,_)|k.len()+64).sum::<usize>()>room) {list.remove(0);}
    });
}
pub fn import(source:StampSource,ctl:&Ctl)->Result<(),String> {
    control(ctl)?;
    if let Some(stamp)=LIBRARY.with(|l|l.borrow().iter().find(|s|s.source==source).cloned()) {
        stamp.portable.set(true);return Ok(());
    }
    let mut key=serde_json::to_vec(&source).map_err(|e|e.to_string())?;key.shrink_to_fit();
    let hit=IMPORTS.with(|i|i.borrow().iter().find_map(|(k,s)|if *k==key {s.upgrade()}else{None}));
    if hit.is_some_and(|s|LIBRARY.with(|l|l.borrow().iter().any(|p|Rc::ptr_eq(p,&s)))) {return Ok(());}
    let stamp=remember_as(source,ctl,true)?;
    let entries=LIBRARY.with(|l|l.borrow().clone());
    for retained in &entries {
        if Rc::ptr_eq(retained,&stamp) || retained.dominates(&stamp,ctl)? {
            let used=entries.iter().map(|s|s.bytes).sum::<usize>();
            let bytes=key.len()+64;
            trim_imports(MAX_BYTES.saturating_sub(used+bytes));
            if used+import_bytes()+bytes<=MAX_BYTES {IMPORTS.with(|i|i.borrow_mut().push((key,Rc::downgrade(retained))));}
            break;
        }
    }
    Ok(())
}
fn remember_as(source:StampSource,ctl:&Ctl,portable:bool)->Result<Rc<Stamp>,String> {
    let _time=measure("remember");
    control(ctl)?;
    if let Some(stamp)=LIBRARY.with(|l|l.borrow().iter().find(|s|s.source==source).cloned()) {
        if portable {stamp.portable.set(true);}return Ok(stamp);
    }
    DEPTH.with(|n|if n.get()>=8 {Err("nested stamp limit")} else {n.set(n.get()+1);Ok(())})?;
    let _depth=CompileDepth;
    let stamp=Rc::new(Stamp::compile(source,ctl)?);
    if let Some(prior)=LIBRARY.with(|l|l.borrow().iter().find(|s|s.source==stamp.source).cloned()) {
        if portable {prior.portable.set(true);}return Ok(prior);
    }
    if portable {stamp.portable.set(true);}
    LIBRARY.with(|l|->Result<(),String> {
        let mut list=l.borrow_mut();
        let mut replaced=Vec::new();
        for (i,prior) in list.iter().enumerate() {
            let older=prior.dominates(&stamp,ctl)?;
            let newer=stamp.dominates(prior,ctl)?;
            if older && (!newer || prior.bytes<=stamp.bytes) {
                if stamp.portable.get() {prior.portable.set(true);}
                return Ok(());
            }
            if newer {replaced.push(i);}
        }
        for &i in replaced.iter().rev() {
            if list[i].portable.get() {stamp.portable.set(true);}
            list.remove(i);
        }
        trim_imports(MAX_BYTES.saturating_sub(list.iter().map(|s|s.bytes).sum::<usize>()+stamp.bytes));
        while !list.is_empty() && (list.len()>=MAX_STAMPS || list.iter().map(|s|s.bytes).sum::<usize>()+stamp.bytes+import_bytes()>MAX_BYTES) {
            let evict=list.iter().position(|s|!s.portable.get()).unwrap_or(0);list.remove(evict);
        }
        list.push(stamp.clone());
        Ok(())
    })?;
    // A caller may be checking this exact certificate or using its root move.
    // Library dominance must not substitute another strategy or coordinate frame.
    Ok(stamp)
}
pub fn prune(board:&Board) {
    // Small primitives remain relocatable. Larger, game-local strategies whose
    // fixed cells are occupied incorrectly cannot revive on this branch.
    LIBRARY.with(|l|l.borrow_mut().retain(|s|s.portable.get() || s.required.len()<=4 ||
        !(s.empty.iter().any(|p|board.contains_key(p)) || s.required.iter().any(|p|board.get(p).is_some_and(|&side|side!=s.source.winner)))));
}
pub fn stats()->(usize,usize) {LIBRARY.with(|l|{let l=l.borrow();(l.len(),l.iter().map(|s|s.bytes).sum::<usize>()+import_bytes())})}
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dominance_preserves_matches_and_speed_tradeoffs() {
        let ctl=Ctl::new(0.0);
        let source=StampSource{stones:(0..4).map(|q|((q,0),0)).collect(),player:0,remaining:2,winner:0,
            certificate:ProofCertificate{version:1,width:"wide".into(),root:0,
                nodes:vec![ProofNode::ImmediateWin{action:vec![(4,0),(5,0)]}]}};
        let mut broad=Stamp::compile(source,&ctl).unwrap();
        // Restrict a real proof with counter-threat predicates to test their
        // implication independently of the strategy that generated them.
        broad.allowance=2;broad.before.insert((2,9));broad.guards.insert((0,10,0),1);
        let mut narrow=broad.clone();narrow.required.insert((9,9));narrow.empty.insert((8,8));
        narrow.before.clear();narrow.guards.insert((0,10,0),0);narrow.turns+=1;
        assert!(broad.dominates(&narrow,&ctl).unwrap());
        assert!(!narrow.dominates(&broad,&ctl).unwrap());
        let varied=[(0,10),(1,10),(2,10),(3,10),(4,10),(5,10),(2,9),(8,8)];
        let mut accepted=0;
        for mut code in 0..3usize.pow(varied.len() as u32) {
            let mut board:Board=narrow.required.iter().map(|&p|(p,0)).collect();
            for &p in &varied {let s=code%3;code/=3;if s>0 {board.insert(p,(s-1) as u8);}}
            let stones:Vec<_>=board.iter().map(|(&p,&s)|(p,player(s))).collect();
            let get=|p|board.get(&p).copied().map(player);
            if narrow.matches(&get,&stones,Player::P1,2) {
                accepted+=1;assert!(broad.matches(&get,&stones,Player::P1,2));
            }
        }
        assert!(accepted>0);
        let mut changed=broad.clone();changed.turns=narrow.turns+1;
        assert!(!changed.dominates(&narrow,&ctl).unwrap());
        changed=broad.clone();changed.source.remaining=1;
        assert!(!changed.dominates(&narrow,&ctl).unwrap());
        changed=broad.clone();changed.source.player=1;
        assert!(!changed.dominates(&narrow,&ctl).unwrap());
        changed=broad.clone();changed.guards.insert((0,10,0),-1);
        assert!(!changed.dominates(&narrow,&ctl).unwrap());
        changed=narrow.clone();changed.before.insert((20,20));
        assert!(!broad.dominates(&changed,&ctl).unwrap());
        changed=narrow.clone();changed.allowance=1;
        assert!(!broad.dominates(&changed,&ctl).unwrap());
    }

    #[test]
    fn transformed_imports_return_the_requested_strategy_and_keep_fixed_frames() {
        LIBRARY.with(|l|l.borrow_mut().clear());
        let ctl=Ctl::new(0.0);
        let entries:serde_json::Value=serde_json::from_str(include_str!("../stamps.json")).unwrap();
        let source:StampSource=serde_json::from_value(entries[0]["source"].clone()).unwrap();
        let first=remember(source,&ctl).unwrap();
        let original=first.source.clone();let count=stats().0;
        assert_eq!(count,1);assert!(!first.portable.get());
        for sym in 0..12 {
            let source=transformed_source(&original,sym,(20,-10),sym%2==1);
            import(source.clone(),&ctl).unwrap();
            let returned=remember(source.clone(),&ctl).unwrap();
            assert_eq!(returned.source,source);
            assert!(LIBRARY.with(|l|l.borrow().iter().any(|s|s.dominates(&returned,&ctl).unwrap())));
            reset_timings(true);
            import(source,&ctl).unwrap();
            assert!(timings().get("compile").is_none());
            reset_timings(false);
        }
        assert!(stats().0>1);
        let mut duplicate=original.clone();
        duplicate.certificate.nodes.extend(std::iter::repeat_n(original.certificate.nodes[0].clone(),512));
        import(duplicate.clone(),&ctl).unwrap();
        reset_timings(true);
        import(duplicate.clone(),&ctl).unwrap();
        assert!(timings().get("compile").is_none());
        assert!(stats().1<=MAX_BYTES);
        LIBRARY.with(|l|l.borrow_mut().clear());
        reset_timings(true);
        import(duplicate,&ctl).unwrap();
        assert!(timings().get("compile").is_some());
        reset_timings(false);
        LIBRARY.with(|l|l.borrow_mut().clear());
    }

    #[test]
    fn replacement_keeps_portability_and_active_oracle_sources() {
        LIBRARY.with(|l|l.borrow_mut().clear());
        let ctl=Ctl::new(0.0);
        let source=StampSource{stones:(0..4).map(|q|((q,0),0)).collect(),player:0,remaining:2,winner:0,
            certificate:ProofCertificate{version:1,width:"wide".into(),root:0,
                nodes:vec![ProofNode::ImmediateWin{action:vec![(4,0),(5,0)]}]}};
        let mut narrow=Stamp::compile(source.clone(),&ctl).unwrap();
        narrow.source.stones.reverse();narrow.allowance=2;narrow.guards.insert((8,8,0),0);narrow.portable.set(true);
        let old_source=narrow.source.clone();
        LIBRARY.with(|l|l.borrow_mut().push(Rc::new(narrow)));
        let oracle=Oracle::new(&ctl,&source.stones.iter().copied().collect());
        let broad=remember(source,&ctl).unwrap();
        assert_eq!(stats().0,1);assert!(broad.portable.get());
        assert!(!broad.guards.contains_key(&(8,8,0)));
        assert_eq!(oracle.source(0),old_source);
        assert!(LIBRARY.with(|l|Rc::ptr_eq(&l.borrow()[0],&broad)));
        LIBRARY.with(|l|l.borrow_mut().clear());
    }

    #[test]
    fn fixed_frames_survive_saturated_geometry_discovery() {
        let ctl=Ctl::new(0.0);
        let source=StampSource{stones:(0..4).map(|q|((q,0),0)).collect(),player:0,remaining:2,winner:0,
            certificate:ProofCertificate{version:1,width:"wide".into(),root:0,
                nodes:vec![ProofNode::ImmediateWin{action:vec![(4,0),(5,0)]}]}};
        let translated=transformed_source(&source,0,(20,-10),false);
        // Neither frame relies on the other's capped geometry candidates.
        for sources in [[source.clone(),translated.clone()],[translated.clone(),source.clone()]] {
            LIBRARY.with(|l|l.borrow_mut().clear());
            for source in sources {remember(source,&ctl).unwrap();}
            assert_eq!(stats().0,2);
            assert!(LIBRARY.with(|l|l.borrow().iter().all(|s|!s.portable.get())));
            let mut board:Board=[((20,-10),0),((21,-10),0)].into_iter().collect();
            let oracle=Oracle::new(&ctl,&board);
            for candidates in oracle.candidates.borrow_mut().iter_mut() {
                for i in 0..512 {candidates.insert((0,(1000+i,1000),false));}
            }
            board.insert((22,-10),0);board.insert((23,-10),0);
            let stones:Vec<_>=board.iter().map(|(&p,&s)|(p,player(s))).collect();
            let (id,winner,turns)=oracle.lookup(1,&stones,&|p|board.get(&p).copied().map(player),Player::P1,2).unwrap();
            assert_eq!((winner,turns),(Player::P1,1));
            assert_eq!(verify(&oracle.source(id),&board,ply(0,2),0,&ctl).unwrap(),1);
        }
        LIBRARY.with(|l|l.borrow_mut().clear());
    }
}
