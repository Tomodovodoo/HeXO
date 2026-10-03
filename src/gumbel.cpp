// Native placement-tree scheduling. Algorithm reference: DeepMind mctx.
#include "hexo.cpp"
#include <memory>
#include <random>
#include <map>
#include <numeric>
#include <array>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
namespace gumbel {
struct Node;
// One proven edge of a position, sorted by action like the legal moves.
struct EdgeProof { Cell action;int winner=-1,distance=-1;bool bound=false; };
// A proven position: its mover, winner, distance and bound, the stones it holds and the edge proofs gathered from its
// expanded nodes (for a win its winning moves, for a loss each move's resistance).
struct Outcome {
 int player=0,winner=-1,distance=-1,stones=0;bool bound=false;std::vector<EdgeProof> edges;
 bool witnessed()const {return winner==player && std::any_of(edges.begin(),edges.end(),[&](const EdgeProof& e){return e.winner==winner;});}
};
// Tightens a proof (winner, distance, bound) to a new one when that is new, contradicts it, or is tighter; true when
// it changed.
inline bool tighten(int winner,int distance,bool bound,int& w,int& d,bool& b) {
 if(w>=0 && w==winner && (distance>d || (distance==d && (bound || !b))))return false;
 w=winner;d=distance;b=bound;return true;
}
// 128-bit order-independent keys: `position` (stones by colour, mover, remaining placements) decides the game value;
// `context` adds the network's turn inputs (the stone placed earlier in this turn, the opponent's previous turn), the
// same identity as dense_selfplay.position_key.
struct Key { uint64_t a=0,b=0;bool operator==(const Key&)const=default; };
struct KeyHash { size_t operator()(const Key& k)const {return size_t(k.a^mix(k.b));} };
std::pair<Key,Key> keys(const Board& board) {
 Key position{mix(board.player*3+board.remaining+17),mix(board.player*3+board.remaining+71)};
 for(auto [cell,p]:board.cells){auto h=CellHash{}(cell);position.a+=mix(h^mix(p+1));position.b+=mix(h+mix(p+911));}
 Key context=position;const size_t n=board.history.size(),start=n%2 || n==0?n:n-1;
 if(start<n){auto h=CellHash{}(board.history[start].c);context.a^=mix(h+0x51);context.b^=mix(h+0x93);}
 for(size_t i=start>=2?start-2:0;i<start;++i){auto h=CellHash{}(board.history[i].c);context.a+=mix(h+0x7f1);context.b+=mix(h+0x3c9);}
 return {position,context};
}
// The same keys from a placement history, without replaying it: stone i belongs to player ((i + 1) / 2) % 2.
std::pair<Key,Key> keys(const std::vector<Cell>& history) {
 const size_t n=history.size();const int player=int((n+1)/2%2),remaining=n==0 || n%2==0?1:2;
 Key position{mix(player*3+remaining+17),mix(player*3+remaining+71)};
 for(size_t i=0;i<n;++i){auto h=CellHash{}(history[i]);int p=int((i+1)/2%2);position.a+=mix(h^mix(p+1));position.b+=mix(h+mix(p+911));}
 Key context=position;const size_t start=n%2 || n==0?n:n-1;
 if(start<n){auto h=CellHash{}(history[start]);context.a^=mix(h+0x51);context.b^=mix(h+0x93);}
 for(size_t i=start>=2?start-2:0;i<start;++i){auto h=CellHash{}(history[i]);context.a+=mix(h+0x7f1);context.b+=mix(h+0x3c9);}
 return {position,context};
}
struct Edge { Cell action;double logit=0,prior=0,sum=0,gumbel=0,weight=-1;int visits=0,pending=0,epoch=0,exact_winner=-1,distance=-1;bool eligible=true,bound=false;std::shared_ptr<Node> child; };
// An exact winner comes with a distance: the placements within which that winner completes six from this position
// (an edge counts its own placement) against any defence, combined by min at the winner's choices and max at the
// loser's. It is exact for terminal and tactical results; `bound` marks an upper bound, which certificates give.
// With graph search a node also keeps its visits `n`, its utility `q` for its mover (the MCGS value) and its parents.
// In a shared graph `dirty` marks a value that a descendant's statistics have changed since it was computed, `used`
// the last search step that touched the node, `context` its key in the store, and `carried` and `carried_sum` the
// visits and value sum (for its mover) an evicted node of its context had when it left the store, less its own
// network value, which the node's expansion supplies again.
struct Node : std::enable_shared_from_this<Node> { int player=0,remaining=1,exact_winner=-1,distance=-1,n=0,stones=0,carried=0;bool expanded=false,pending=false,bound=false,dirty=false;double value=0,q=0,carried_sum=0;uint64_t used=0;Key position,context;std::vector<Edge> edges;std::vector<std::weak_ptr<Node>> parents; };
// A pending leaf: its history, its legal moves in sorted order and, with tactics, the side to move's completions
// (own) and the opponent's (threats), both restricted to fully legal ones.
struct Path { Node* leaf=nullptr;std::vector<std::pair<Node*,int>> edges;std::vector<Cell> history,legal;int player=0,remaining=1;std::vector<std::vector<Cell>> own,threats; };
struct Tree {
 Board board;std::shared_ptr<Node> root=std::make_shared<Node>();std::map<int,Path> requests;
 // Graph search (opt-in): nodes shared by turn-context key, proven outcomes shared by
 // position key. Tree search gives every edge its own child and keeps both tables empty.
 bool graph=false;std::unordered_map<Key,std::weak_ptr<Node>,KeyHash> nodes;std::unordered_map<Key,std::vector<std::weak_ptr<Node>>,KeyHash> positions;std::unordered_map<Key,Outcome,KeyHash> outcomes;
 // Shared game graph (opt-in, implies graph): `store` owns every node by turn-context key so roots can move to any
 // position (root_at) and back; an edge reads its child's visits and value (current); `limit` bounds the expanded
 // nodes kept (evict); `lineage` holds the stored positions the root's history passes through, each with the edge of
 // that history out of it (-1 when unexpanded), credited with each playout; `version` counts root changes.
 bool shared=false;size_t limit=0;uint64_t clock=0;int64_t evicted=0;std::unordered_map<Key,std::shared_ptr<Node>,KeyHash> store;std::vector<std::pair<Node*,int>> lineage;int64_t version=0;
 // Shared graph: the visits and value (for its mover) of evicted nodes, by context, for a node created again there;
 // evict keeps at most four times `limit` of them, those with the most visits.
 struct Summary { int n=0;double q=0,value=0;Key position; };std::unordered_map<Key,Summary,KeyHash> evicted_stats;
 std::mt19937_64 rng;int budget=0,started=0,completed=0,next_id=1,samples=0,last=0;bool tactics=false,hold=false;std::vector<int> sequence;
 // Root actions sampled first in the opening phase of the current search; ordering only (set_priority).
 std::vector<Cell> priority;
 std::map<Cell,double> defence;
 double range_floor=0;  // least Q range of the completed-Q rescale (transformed)
 double root_noise=0;   // uniform share of the root's candidate sampling distribution (sampling)
 double bonus(const Edge& e)const {auto i=defence.find(e.action);return i==defence.end()?0:i->second;}
 std::vector<double> work;
 explicit Tree(uint64_t seed):rng(seed){}
 bool done()const {return requests.empty() && (board.winner>=0 || (root->expanded && root->exact_winner>=0) || completed>=budget);}
 // Edge value for the node's mover: exact, else in a graph the child's MCGS value (which may come from other
 // parents), else the tree's running mean, else the node's own network value.
 double value(const Node& node,const Edge& e)const {
  if(e.exact_winner>=0)return e.exact_winner==node.player?1:-1;
  if(graph && e.child && e.child->n)return e.child->player==node.player?e.child->q:-e.child->q;
  return e.visits?e.sum/e.visits:node.value;
 }
 bool known(const Edge& e)const {return e.exact_winner>=0 || e.visits || (graph && e.child && e.child->n);}
 // The node for the tree's board as a new child: a fresh node, or with graph search the shared node of this turn
 // context, created with any outcome already proven for the position.
 std::shared_ptr<Node> child_here() {
  if(!graph){auto n=std::make_shared<Node>();n->player=board.player;return n;}
  auto [position,context]=keys(board);auto& slot=nodes[context];
  if(auto n=slot.lock())return n;
  auto n=std::make_shared<Node>();n->player=board.player;n->position=position;n->context=context;n->stones=int(board.cells.size());slot=n;positions[position].push_back(n);
  if(shared){
   store[context]=n;n->used=clock;
   if(auto old=evicted_stats.find(context);old!=evicted_stats.end()){
    auto& o=old->second;n->n=o.n;n->q=o.q;n->carried=o.n-1;n->carried_sum=o.q*o.n-o.value;evicted_stats.erase(old);
   }
  }
  if(auto o=outcomes.find(position);o!=outcomes.end())apply(o->second,*n);
  return n;
 }
 // Shared graph: marks `node` and every ancestor stale. A stale node's ancestors are always stale, so the walk stops
 // at a node already marked.
 void stale(Node& node) {
  std::vector<Node*> work{&node};
  while(!work.empty()){
   Node* x=work.back();work.pop_back();
   if(x->dirty)continue;
   x->dirty=true;
   for(auto& w:x->parents)if(auto p=w.lock())work.push_back(p.get());
  }
 }
 // Shared graph: recomputes every stale node below and including `top`, children first.
 void clean(Node& top) {
  if(!top.dirty)return;
  std::vector<std::pair<Node*,size_t>> work{{&top,0}};
  while(!work.empty()){
   auto& [x,i]=work.back();
   if(i<x->edges.size()){auto& e=x->edges[i++];if(e.child && e.child->dirty)work.emplace_back(e.child.get(),0);continue;}
   Node* done=x;work.pop_back();
   refresh(*done);done->dirty=false;
  }
 }
 // Shared graph: brings the node's children up to date before its edges' values are read. An edge keeps its own
 // visits (the playouts that went through it, and the credits of roots below it on their history) and reads its
 // child's value, so a child reached by both orders of a turn is not weighted twice.
 void current(Node& node) {
  for(auto& e:node.edges)if(e.child)clean(*e.child);
 }
 void renew(Node& node) {current(node);refresh(node);node.dirty=false;}
 // Shared graph: the stored positions `history`'s strict prefixes reach, the longest first, each with the index of
 // its edge along `history` (-1 when the node is unexpanded).
 std::vector<std::pair<Node*,int>> prefixes(const std::vector<Cell>& history) {
  std::vector<std::pair<Node*,int>> out;std::vector<Cell> prefix(history);
  while(!prefix.empty()){
   const Cell action=prefix.back();prefix.pop_back();
   auto found=nodes.find(keys(prefix).second);if(found==nodes.end())continue;
   auto n=found->second.lock();if(!n)continue;
   int index=-1;for(int i=0;i<int(n->edges.size());++i)if(n->edges[i].action==action){index=i;break;}
   out.emplace_back(n.get(),index);
  }
  return out;
 }
 // Shared graph: makes `child` the child of `parent`'s edge `e`; the edge keeps its visits. An edge whose evicted
 // child left no summary (dropped by evict) and whose new child has no visits restarts at zero, so one new sample
 // never stands for the edge's history. An exact child settles the edge and the parent's verdict reaches its own
 // stored parents (revise); otherwise the parent's value becomes stale.
 void attach(Node& parent,Edge& e,const std::shared_ptr<Node>& child) {
  if(e.visits && !child->n && e.exact_winner<0){e.visits=0;e.sum=0;}
  e.child=child;child->parents.push_back(parent.weak_from_this());
  if(child->exact_winner>=0 && tighten(child->exact_winner,child->distance+1,child->bound,e.exact_winner,e.distance,e.bound)){
   settle(parent);learn(parent);revise(parent);
  }
  else stale(parent);
 }
 // Shared graph: attaches to a newly expanded node every legal move whose turn context the store already holds.
 // A move's position key follows from the node's own, so the full key is computed only for positions present.
 void link(const Path& path,Node& node) {
  const int p=path.player,r=path.remaining,after=r==2?p:1-p,left=r==2?1:2;
  const uint64_t a=node.position.a-mix(p*3+r+17)+mix(after*3+left+17),b=node.position.b-mix(p*3+r+71)+mix(after*3+left+71);
  std::vector<Cell> history=path.history;history.push_back({});
  for(auto& e:node.edges){
   if(e.child)continue;
   auto h=CellHash{}(e.action);
   if(!positions.contains(Key{a+mix(h^mix(p+1)),b+mix(h+mix(p+911))}))continue;
   history.back()=e.action;
   if(auto found=nodes.find(keys(history).second);found!=nodes.end())if(auto child=found->second.lock())attach(node,e,child);
  }
 }
 // Shared graph: attaches `node`, the position after `history`, under the expanded nodes that reach it by its last
 // stone: the position before it and, when the last two stones are one turn, the other order of that turn.
 void adopt(const std::shared_ptr<Node>& node,const std::vector<Cell>& history) {
  const size_t n=history.size();
  auto join=[&](const std::vector<Cell>& before,Cell action){
   auto found=nodes.find(keys(before).second);if(found==nodes.end())return;
   auto parent=found->second.lock();if(!parent || !parent->expanded)return;
   for(auto& e:parent->edges)if(e.action==action){if(!e.child)attach(*parent,e,node);return;}
  };
  auto mover=[](size_t i){return (i+1)/2%2;};
  if(!n)return;
  join({history.begin(),history.end()-1},history.back());
  if(n>=2 && mover(n-1)==mover(n-2)){std::vector<Cell> other(history.begin(),history.end()-2);other.push_back(history.back());join(other,history[n-2]);}
 }
 // Shared graph: while more than `limit` expanded nodes are stored, removes the least recently used leaves (nodes
 // without a child) other than the root, down to seven eighths of the limit, and bounds the summaries and outcomes
 // kept for positions no longer stored. A removed child's last visits and value
 // stay on its parents' edges.
 void evict() {
  if(!shared || !limit || !requests.empty())return;
  auto count=[&]{size_t k=0;for(auto& [key,n]:store)k+=n->expanded;return k;};
  size_t expanded=count();if(expanded<=limit)return;
  const size_t target=limit-limit/8;
  while(expanded>target){
   std::vector<Node*> leaves;
   for(auto& [key,n]:store)if(n!=root && !n->pending && std::none_of(n->edges.begin(),n->edges.end(),[](const Edge& e){return bool(e.child);}))leaves.push_back(n.get());
   if(leaves.empty())break;
   std::sort(leaves.begin(),leaves.end(),[](const Node* x,const Node* y){return x->used<y->used;});
   for(Node* x:leaves){
    if(expanded<=target)break;
    clean(*x);
    for(auto& w:x->parents)if(auto p=w.lock())for(auto& e:p->edges)if(e.child.get()==x){
     e.sum=(e.exact_winner>=0?(e.exact_winner==p->player?1:-1):x->player==p->player?x->q:-x->q)*e.visits;
     e.child.reset();
    }
    if(x->n)evicted_stats[x->context]={x->n,x->q,x->value,x->position};
    expanded-=x->expanded;++evicted;store.erase(x->context);
   }
  }
  // At most four times `limit` summaries: those with the fewest visits leave first.
  if(evicted_stats.size()>4*limit){
   std::vector<std::pair<int,Key>> order;for(auto& [key,stats]:evicted_stats)order.emplace_back(stats.n,key);
   std::nth_element(order.begin(),order.begin()+(order.size()-4*limit),order.end(),[](const auto& x,const auto& y){return x.first<y.first;});
   for(size_t i=0;i<order.size()-4*limit;++i)evicted_stats.erase(order[i].second);
  }
  std::erase_if(nodes,[](const auto& entry){return entry.second.expired();});
  for(auto& [key,list]:positions)std::erase_if(list,[](const auto& w){return w.expired();});
  std::erase_if(positions,[](const auto& entry){return entry.second.empty();});
  // Proven outcomes stay while a stored node or a kept summary holds their position; beyond sixteen times `limit` the
  // others go.
  if(outcomes.size()>16*limit){
   std::unordered_set<Key,KeyHash> summarised;for(auto& [key,stats]:evicted_stats)summarised.insert(stats.position);
   std::erase_if(outcomes,[&](const auto& entry){return !positions.contains(entry.first) && !summarised.contains(entry.first);});
  }
 }
 // Shared graph: moves the root to the position after `history`, a node of the store or a new one attached under
 // its stored parents; every node keeps its statistics.
 void root_at(const std::vector<Cell>& history) {
  if(!shared || !requests.empty())throw std::runtime_error("Root changes need a shared graph and no pending requests");
  Board next;for(auto c:history){if(!next.legal(c))throw std::runtime_error("Illegal root history");next.make(c);}
  board=next;priority.clear();defence.clear();hold=false;budget=started=completed=0;lineage.clear();++version;
  root=child_here();root->player=board.player;root->used=++clock;adopt(root,history);
  evict();
 }
 // MCGS value of a graph node for its mover from its network value and its edges' visits and current values.
 void refresh(Node& node) {
  double total=node.value+node.carried_sum;int count=1+node.carried;
  for(auto& e:node.edges)if(e.visits){total+=e.visits*value(node,e);count+=e.visits;}
  node.q=node.exact_winner>=0?(node.exact_winner==node.player?1:-1):total/count;
 }
 // After a backup changed `from`, bring its parents other than `skip` (its parent on the backup path) and their
 // ancestors up to date: an exact child's winner, distance and bound are copied to each incoming edge and the parent
 // is settled, then its value is refreshed, so no parent keeps a verdict or value from before its shared child's
 // latest visits.
 void propagate(Node& from,const Node* skip) {
  std::vector<std::pair<Node*,Node*>> work;
  for(auto& w:from.parents)if(auto p=w.lock())if(p.get()!=skip)work.emplace_back(p.get(),&from);
  while(!work.empty()){
   auto [p,c]=work.back();work.pop_back();
   const double before=p->q;const int winner=p->exact_winner,distance=p->distance;const bool bound=p->bound;
   if(c->exact_winner>=0){
    bool changed=false;
    for(auto& e:p->edges)if(e.child.get()==c)changed|=tighten(c->exact_winner,c->distance+1,c->bound,e.exact_winner,e.distance,e.bound);
    if(changed){settle(*p);learn(*p);}
   }
   // A shared graph marks values stale for the next read and walks on only with a changed verdict.
   if(shared){
    stale(*p);
    if(p->exact_winner!=winner || p->distance!=distance || p->bound!=bound)for(auto& w:p->parents)if(auto g=w.lock())work.emplace_back(g.get(),p);
    continue;
   }
   refresh(*p);
   if(p->q!=before || p->exact_winner!=winner || p->distance!=distance || p->bound!=bound)for(auto& w:p->parents)if(auto g=w.lock())work.emplace_back(g.get(),p);
  }
 }
 // Records a proven node's outcome for its position (graph search), keeping the shortest bound.
 // A new or better outcome also reaches the live nodes of the same position in other turn contexts.
 void learn(const Node& node) {
  if(!graph || node.exact_winner<0)return;
  Outcome outcome{node.player,node.exact_winner,node.distance,node.stones,node.bound};
  if(node.expanded)for(auto& e:node.edges)if(e.exact_winner>=0)outcome.edges.push_back({e.action,e.exact_winner,e.distance,e.bound});
  if(!record(node.position,outcome))return;
  // Peers take any improvement: a better outcome, or new edge proofs under an unchanged one.
  if(auto list=positions.find(node.position);list!=positions.end())for(auto& w:std::vector(list->second))if(auto n=w.lock())if(n.get()!=&node)share(node,outcomes[node.position],*n);
 }
 // Merges `outcome` into the table entry of `position`: a tighter verdict and any new or tighter edge proof; true when
 // the entry changed.
 bool record(const Key& position,const Outcome& outcome) {
  auto [o,added]=outcomes.try_emplace(position,outcome);
  if(added)return true;
  auto& old=o->second;bool changed=false;
  if(old.winner!=outcome.winner){old=outcome;return true;}
  changed|=tighten(outcome.winner,outcome.distance,outcome.bound,old.winner,old.distance,old.bound);
  std::vector<EdgeProof> merged;merged.reserve(old.edges.size()+outcome.edges.size());
  auto i=old.edges.cbegin();auto j=outcome.edges.cbegin();
  while(i!=old.edges.cend() || j!=outcome.edges.cend()){
   if(j==outcome.edges.cend() || (i!=old.edges.cend() && i->action<j->action)){merged.push_back(*i++);continue;}
   if(i==old.edges.cend() || j->action<i->action){merged.push_back(*j++);changed=true;continue;}
   EdgeProof e=*i++;changed|=tighten(j->winner,j->distance,j->bound,e.winner,e.distance,e.bound);++j;merged.push_back(e);
  }
  old.edges=std::move(merged);
  return changed;
 }
 // Installs a proven outcome on a node of its position: an unexpanded node takes the verdict; an expanded node takes
 // its edge proofs and, for a loss, every still unproven edge as a bounded loss, and is settled. True when anything
 // improved.
 bool apply(const Outcome& o,Node& into) {
  if(!into.expanded)return tighten(o.winner,o.distance,o.bound,into.exact_winner,into.distance,into.bound);
  bool changed=false;auto j=o.edges.begin();
  for(auto& e:into.edges){
   while(j!=o.edges.end() && j->action<e.action)++j;
   if(j!=o.edges.end() && j->action==e.action)changed|=tighten(j->winner,j->distance,j->bound,e.exact_winner,e.distance,e.bound);
   else if(o.winner!=into.player && e.exact_winner<0){e.exact_winner=o.winner;e.distance=o.distance;e.bound=true;changed=true;}
  }
  if(changed)settle(into);
  return changed;
 }
 // Copies to `into` every edge proof of `from`, another expanded node of the same position (the legal moves and their
 // order are the same), that is new or tighter; true when anything changed. The caller settles `into`.
 bool copy(const Node& from,Node& into) {
  bool changed=false;
  if(from.edges.size()!=into.edges.size())return false;
  for(size_t i=0;i<from.edges.size();++i){
   auto& f=from.edges[i];auto& e=into.edges[i];
   if(f.exact_winner>=0 && f.action==e.action)changed|=tighten(f.exact_winner,f.distance,f.bound,e.exact_winner,e.distance,e.bound);
  }
  return changed;
 }
 // Gives `into`, a node of the same position in another turn context, the verdicts `from` holds: all its proven edges
 // when both are expanded (the legal moves are the same), else its outcome; then updates its parents.
 void share(const Node& from,const Outcome& outcome,Node& into) {
  bool changed=false;
  if(from.expanded && into.expanded){if((changed=copy(from,into)))settle(into);}
  else changed=apply(outcome,into);
  if(changed)revise(into);
 }
 // Updates a node whose verdict changed outside a backup, then its parents (propagate). A shared graph marks its
 // value stale instead of recomputing it.
 void revise(Node& node) {
  if(shared)stale(node);else refresh(node);
  propagate(node,nullptr);
 }
 // Completed Q in value units for the node's mover: each known edge's value, the mixed value (the network value
 // blended with the prior-weighted values of visited eligible edges) for the rest. `maximum` receives the largest
 // visit count among eligible edges.
 std::vector<double>& completed_q(Node& node,int& maximum) {
  double weighted=0,mass=0;int total=0;maximum=0;
  for(auto& e:node.edges)if(e.eligible){total+=e.visits;maximum=std::max(maximum,e.visits);if(e.visits){weighted+=e.prior*value(node,e);mass+=e.prior;}}
  double mixed=(node.value+total*(mass?weighted/mass:node.value))/(total+1);
  auto& q=work;q.clear();q.reserve(node.edges.size());
  for(auto& e:node.edges)q.push_back(known(e)?value(node,e):mixed);
  return q;
 }
 // Completed Q (mctx mixed value, min-max rescale, (50 + max visits) * 0.1) over the eligible edges only: proven
 // losses and the non-winning edges of a won node set neither the mixed value, the visit scale nor the range.
 // The rescale divides by max(1e-8, range_floor, hi - lo), so a spread below range_floor is not stretched to the full scale.
 // Entries of ineligible edges are returned on the same scale but every caller discards them.
 std::vector<double>& transformed(Node& node) {
  int maximum=0;auto& q=completed_q(node,maximum);double lo=1e300,hi=-1e300;
  for(size_t i=0;i<q.size();++i)if(node.edges[i].eligible){lo=std::min(lo,q[i]);hi=std::max(hi,q[i]);}
  if(lo>hi)lo=hi=0;
  double range=std::max(std::max(1e-8,range_floor),hi-lo);
  for(auto& x:q)x=(x-lo)/range*(50+maximum)*0.1;
  return q;
 }
 // Stable interior softmax weights. Unknown edges share the same completed Q, so reuse their exp(logit)
 // from expansion. Subnormal weights and a non-finite multiplier use the original shifted exponential.
 int interior(Node& node,std::vector<double>& q) {
  double maxlog=-1e300,unknown=-std::numeric_limits<double>::infinity();int visits=0;
  for(int i=0;i<int(q.size());++i){auto& e=node.edges[i];if(e.eligible && !known(e))unknown=q[i];q[i]=e.eligible?q[i]+e.logit:-std::numeric_limits<double>::infinity();maxlog=std::max(maxlog,q[i]);visits+=e.visits+e.pending;}
  double factor=std::exp(unknown-maxlog);
  for(int i=0;i<int(q.size());++i){auto& e=node.edges[i];double x=0;if(e.eligible){if(!known(e) && std::isfinite(factor)){if(e.weight<0)e.weight=std::exp(e.logit);x=e.weight>=std::numeric_limits<double>::min() && std::isfinite(e.weight)?e.weight*factor:std::exp(q[i]-maxlog);}else x=std::exp(q[i]-maxlog);}q[i]=x;}
  return visits;
 }
 // Root candidate sampling logits: each edge's logit, or with root_noise e > 0 log((1 - e) p + e / N) for the N
 // eligible edges, p their softmax over the eligible logits. Only the opening phase's Gumbel-top-k draws on these;
 // halving, the final choice, the improved policy and every non-root node use the edge logits.
 std::vector<double> sampling(const Node& node)const {
  std::vector<double> out;double maximum=-1e300,total=0;int n=0;
  for(auto& e:node.edges){out.push_back(e.logit);if(e.eligible){maximum=std::max(maximum,e.logit);++n;}}
  if(root_noise<=0 || !n)return out;
  for(auto& e:node.edges)if(e.eligible)total+=std::exp(e.logit-maximum);
  for(size_t i=0;i<out.size();++i)if(node.edges[i].eligible)out[i]=std::log((1-root_noise)*std::exp(out[i]-maximum)/total+root_noise/n);
  return out;
 }
 // `last` is the simulation index where the final candidate count begins (the last halving boundary), or the
 // budget when the schedule never halves.
 void schedule(int count) {
  sequence.clear();last=budget;int m=std::min({std::max(samples,int(defence.size())),budget,count});if(!m)return;
  std::vector<int> v(m);int considered=m,previous=-1,halving=0,rounds=std::max(1,int(std::ceil(std::log2(m))));
  while(int(sequence.size())<budget){if(considered!=previous){halving=int(sequence.size());previous=considered;}int extra=std::max(1,budget/(rounds*considered));for(int k=0;k<extra && int(sequence.size())<budget;++k)for(int i=0;i<considered;++i){sequence.push_back(v[i]++);if(int(sequence.size())==budget)break;}considered=m==1?1:std::max(2,considered/2);}
  if(halving)last=halving;
 }
 // Leaf data classify needs, taken while the tree's board stands at the leaf.
 void capture(Path& path) {
  path.legal=board.legal_moves();path.player=board.player;path.remaining=board.remaining;
  if(!tactics)return;
  path.own=board.completions(board.player,board.remaining);path.threats=board.completions(1-board.player);
  auto illegal=[&](const auto& c){return !std::all_of(c.begin(),c.end(),[&](Cell p){return board.legal(p);});};
  std::erase_if(path.own,illegal);std::erase_if(path.threats,illegal);
 }
 void classify(const Path& path,Node& node) {
  const auto& own=path.own;const auto& threats=path.threats;
  auto contains=[](const std::vector<Cell>& completion,Cell c){return std::find(completion.begin(),completion.end(),c)!=completion.end();};
  constexpr int none=std::numeric_limits<int>::max();
  // The opponent's fastest completion left open by `a` (and `b`), none when every threat is hit.
  auto open=[&](Cell a,const Cell* b){int k=none;for(auto& t:threats)if(!contains(t,a) && !(b && contains(t,*b)))k=std::min(k,int(t.size()));return k;};
  for(auto& edge:node.edges){
   // Every completion cell is within five of an existing stone, hence legal.
   int win=none;
   for(auto& completion:own){int n=int(completion.size());if(contains(completion,edge.action))win=std::min(win,n);else if(n<path.remaining)win=std::min(win,n+1);}
   if(win!=none){edge.exact_winner=path.player;edge.distance=win;edge.bound=false;continue;}
   // A lost edge resists longest with the second stone that leaves the slowest completion open.
   int lost=open(edge.action,nullptr);
   if(lost!=none && path.remaining==2)for(auto& t:threats){for(auto second:t){int k=open(edge.action,&second);lost=std::max(lost,k);if(k==none)break;}if(lost==none)break;}
   if(lost!=none){edge.exact_winner=1-path.player;edge.distance=path.remaining+lost;edge.bound=false;}
  }
  settle(node);
 }
 // Node verdict, distance and eligibility from its edges' exact winners. A won node offers its shortest guaranteed
 // wins (least upper bound); its distance is exact when an exact edge attains it and no bounded win could be faster.
 // A lost node offers every loss that may resist longest: those whose distance reaches the largest lower bound among
 // its losses. An exact distance is its own lower bound. When tactics classified the node, a bounded result is not a
 // one-turn result: a bounded win needs the mover's remaining stones, the opponent's turn and one more stone
 // (remaining + 3); a bounded loss also lets the mover play its next turn first (remaining + 5). Without tactics a
 // bound's lower bound is 1. Otherwise only the unproven edges stay eligible.
 void settle(Node& node) {
  // Only expansion installs the complete legal action list. A pending/unexpanded leaf is never a universal proof.
  if(!node.expanded || node.edges.empty())return;
  const int none=std::numeric_limits<int>::max();
  bool winning=false,safe=false,attained=false;int fastest=none,slowest=0,longest=0,quickest=none;
  auto lower=[&](const Edge& e,int turn){return e.bound?(tactics?node.remaining+turn:1):e.distance;};
  for(auto& edge:node.edges){
   if(edge.exact_winner==node.player){winning=true;fastest=std::min(fastest,edge.distance);quickest=std::min(quickest,lower(edge,3));}
   else if(edge.exact_winner<0)safe=true;
   else {slowest=std::max(slowest,edge.distance);longest=std::max(longest,lower(edge,5));}
  }
  for(auto& edge:node.edges)attained|=edge.exact_winner==node.player && !edge.bound && edge.distance==fastest;
  for(auto& edge:node.edges)edge.eligible=winning?edge.exact_winner==node.player && edge.distance==fastest
   :!safe?edge.distance>=longest:edge.exact_winner<0;
  if(winning){node.exact_winner=node.player;node.distance=fastest;node.bound=!attained || quickest<fastest;}
  else if(!safe){node.exact_winner=1-node.player;node.distance=slowest;node.bound=slowest>longest;}
 }
 // Records an externally proven winner of the root edge `action` within `distance` placements (the edge's own
 // included): its value becomes exact (Q = +-1 for the root's mover) and the root is settled, so a lost edge leaves
 // the remaining halving rounds and the final selection. A root that is already exact is left unchanged.
 // In a shared graph the root's stored parents take the new verdict and value.
 void mark(Cell action,int winner,int distance) {
  if(!root->expanded || (winner!=0 && winner!=1) || distance<1)throw std::runtime_error("Mark needs an expanded root, a winner and a distance");
  auto edge=std::find_if(root->edges.begin(),root->edges.end(),[&](const Edge& e){return e.action==action;});
  if(edge==root->edges.end())throw std::runtime_error("Mark action is not a root edge");
  if(root->exact_winner>=0)return;
  edge->exact_winner=winner;edge->distance=distance;edge->bound=true;edge->sum=winner==root->player?edge->visits:-edge->visits;settle(*root);
  // A shared graph hands the verdict and the changed value on to the root's stored parents.
  if(shared){learn(*root);revise(*root);}
 }
 void begin(int simulations,int sample) {
  if(!requests.empty()||simulations<1||sample<1)throw std::runtime_error("Invalid search budget or pending requests");
  budget=simulations;samples=sample;started=completed=0;hold=false;priority.clear();defence.clear();
  if(shared){
   evict();current(*root);
   std::vector<Cell> history;for(auto& u:board.history)history.push_back(u.c);
   lineage=prefixes(history);
  }
  schedule(root->expanded?int(std::count_if(root->edges.begin(),root->edges.end(),[](auto& e){return e.eligible;})):int(board.legal_moves().size()));
  for(auto& e:root->edges){e.epoch=0;double u=std::generate_canonical<double,53>(rng);e.gumbel=-std::log(-std::log(std::clamp(u,1e-15,1-1e-15)));}
 }
 // Tree search: each edge keeps the running mean of the values backed up through it. Graph search (MCGS): each node
 // recomputes q = (network value + sum over edges of visits * edge value) / (1 + sum of visits) from its children's
 // current values, so a child shared by several parents is weighted by this node's own edge visits. `fresh` counts
 // the leaf as visited; a playout reusing a transposed child's value leaves the child unchanged.
 void backup(Path& path,double value,bool fresh=true) {
  Node* child=path.leaf;
  if(shared){++child->n;renew(*child);learn(*child);}
  else if(graph && fresh){++child->n;child->q=child->exact_winner>=0?(child->exact_winner==child->player?1:-1):child->n==1?value:child->q;learn(*child);}
  for(auto i=path.edges.rbegin();i!=path.edges.rend();++i){
   auto& [node,index]=*i;auto& edge=node->edges[index];
   if(child->player!=node->player)value=-value;
   if(child->exact_winner>=0 && tighten(child->exact_winner,child->distance+1,child->bound,edge.exact_winner,edge.distance,edge.bound))settle(*node);
   ++edge.visits;--edge.pending;
   if(edge.exact_winner>=0){value=edge.exact_winner==node->player?1:-1;edge.sum=value*edge.visits;}
   else edge.sum+=value;
   if(graph){++node->n;if(shared)renew(*node);else refresh(*node);learn(*node);}
   child=node;
  }
  // Nodes on the path are current; their other parents are refreshed upwards (marked stale in a shared graph, where
  // the root's own parents follow too and the positions its history passes through count the playout).
  if(graph && !path.edges.empty()){
   propagate(*path.leaf,path.edges.back().first);
   for(size_t j=1;j<path.edges.size();++j)propagate(*path.edges[j].first,path.edges[j-1].first);
  }
  if(shared){
   propagate(*root,nullptr);
   // Only an edge whose value is known takes the credit, and only its node counts the visit, so no visit enters an
   // edge or a node without a value.
   for(auto [x,index]:lineage){
    if(index<0)continue;
    auto& e=x->edges[index];
    if(e.exact_winner>=0){++e.visits;e.sum=(e.exact_winner==x->player?1:-1)*e.visits;}
    else if(e.child && e.child->n){++e.visits;e.sum+=e.child->player==x->player?e.child->q:-e.child->q;}
    else continue;
    ++x->n;stale(*x);
   }
  }
  if(!path.edges.empty())++completed;
 }
 // Next leaf request id; 0 when nothing can be requested now, -1 after a simulation that ended on an exact edge or
 // node, and -3 while a hold is armed and the search stands, with no request pending, at `last`.
 int request() {
  if(board.winner>=0 || (root->expanded && root->exact_winner>=0))return 0;
  if(hold && started==last)return requests.empty()?-3:0;
  if(started>=budget)return 0;
  // Descends by make on the tree's own board; Restore undoes every placement on return.
  Restore restore(board);Node* node=root.get();Path path;path.leaf=node;
  for(auto& u:board.history)path.history.push_back(u.c);
  if(shared)++clock;
  while(node->expanded){
   if(shared){current(*node);node->used=clock;}
   auto& q=transformed(*node);int chosen=-1;double best=-1e300;
   if(node==root.get()){
    int considered=sequence[started];
    // A shared graph samples root candidates with their completed Q as well: an edge whose stored child already
    // holds evidence (a win found from a later position) competes on it, not on its prior alone.
    // Finish each visit layer before its values decide the next halving round.
    if(started && considered!=sequence[started-1] && !requests.empty())return 0;
    auto first=[&](const Edge& e){return considered==0 && std::find(priority.begin(),priority.end(),e.action)!=priority.end();};
    bool forced=false;auto logits=considered?std::vector<double>():sampling(*node);
    for(int i=0;i<int(node->edges.size());++i){auto& e=node->edges[i];if(!e.eligible || e.epoch!=considered)continue;bool admit=considered==0 && defence.contains(e.action);double score=e.gumbel+(considered?e.logit:logits[i])+(considered||shared?q[i]:0)+(first(e)?1e6:0)+bonus(e);if((admit && !forced) || (admit==forced && score>best)){forced=admit;best=score;chosen=i;}}
    // Marked-lost candidates can leave a round short of candidates; the best of the latest-eliminated ones step in.
    int reached=-1;
    if(chosen<0)for(int i=0;i<int(node->edges.size());++i){auto& e=node->edges[i];if(!e.eligible || e.epoch>considered)continue;double score=e.gumbel+e.logit+q[i]+bonus(e);if(e.epoch>reached || (e.epoch==reached && score>best)){reached=e.epoch;best=score;chosen=i;}}
    // Proofs may leave fewer survivors than this round scheduled. If all survivors already finished its
    // layer, continue the least-visited survivor rather than waiting for an eliminated action forever.
    if(chosen<0){int least=std::numeric_limits<int>::max();for(int i=0;i<int(node->edges.size());++i){auto& e=node->edges[i];if(!e.eligible)continue;double score=e.gumbel+e.logit+q[i]+bonus(e);if(e.epoch<least || (e.epoch==least && score>best)){least=e.epoch;best=score;chosen=i;}}}
   } else {
    int visits=interior(*node,q);double total=std::accumulate(q.begin(),q.end(),0.);
    for(int i=0;i<int(q.size());++i){auto& e=node->edges[i];if(!e.eligible || (e.child && e.child->pending))continue;double score=q[i]/total-double(e.visits+e.pending)/(1+visits);if(score>best){best=score;chosen=i;}}
   }
   if(chosen<0)return 0;
   auto& edge=node->edges[chosen];if(edge.child && edge.child->pending)return 0;
   path.edges.emplace_back(node,chosen);board.make(edge.action);path.history.push_back(edge.action);
   if(!edge.child){if(shared)attach(*node,edge,child_here());else {edge.child=child_here();if(graph)edge.child->parents.push_back(node->weak_from_this());}}
   node=edge.child.get();path.leaf=node;if(shared)node->used=clock;
   if(board.winner>=0 || edge.exact_winner>=0 || node->exact_winner>=0){
    if(board.winner>=0){node->exact_winner=board.winner;node->distance=0;node->bound=false;}
    else if(edge.exact_winner>=0 && node->exact_winner<0){node->exact_winner=edge.exact_winner;node->distance=edge.distance-1;node->bound=edge.bound;}
    int winner=node->exact_winner;for(auto [parent,index]:path.edges)++parent->edges[index].pending;++root->edges[path.edges.front().second].epoch;++started;backup(path,winner==node->player?1:-1);return -1;}
   // A transposed child already holds more visits than this edge: take its value without evaluating (MCGS).
   if(graph && !shared && node->expanded && node->n>edge.visits){for(auto [parent,index]:path.edges)++parent->edges[index].pending;++root->edges[path.edges.front().second].epoch;++started;backup(path,node->q,false);return -1;}
  }
  if(node->pending)return 0;
  node->pending=true;node->player=board.player;capture(path);
  for(auto [parent,index]:path.edges)++parent->edges[index].pending;
  if(!path.edges.empty()){++root->edges[path.edges.front().second].epoch;++started;}
  int id=next_id++;requests.emplace(id,std::move(path));return id;
 }
 void fulfill(int id,const int64_t* actions,const double* logits,const double* values,int count,int exact=-1,Cell witness={},int distance=-1){
  auto found=requests.find(id);if(found==requests.end())throw std::runtime_error("Unknown request");auto& path=found->second;
  const auto& legal=path.legal;
  if(count!=int(legal.size())||count<1)throw std::runtime_error("Incomplete legal actions");
  double maximum=-1e300;for(int i=0;i<count;++i){if(legal[i]!=Cell{actions[2*i],actions[2*i+1]}||!std::isfinite(logits[i])||!std::isfinite(values[i])||std::abs(values[i])>1)throw std::runtime_error("Invalid evaluation");maximum=std::max(maximum,logits[i]);}
  auto& node=*path.leaf;double total=0;std::vector<double> weights(count);for(int i=0;i<count;++i)total+=weights[i]=std::exp(logits[i]-maximum);
  // Only root edges read their Gumbel noise and begin() redraws it, so interior edges just advance the stream.
  const bool at_root=&node==root.get();
  node.value=0;node.edges.reserve(count);for(int i=0;i<count;++i){Edge edge;edge.action=legal[i];edge.logit=logits[i]-maximum;edge.prior=weights[i]/total;edge.weight=weights[i];node.value+=edge.prior*values[i];double u=std::generate_canonical<double,53>(rng);if(at_root)edge.gumbel=-std::log(-std::log(std::clamp(u,1e-15,1-1e-15)));node.edges.push_back(std::move(edge));}
  node.expanded=true;node.remaining=path.remaining;
  // A retained proven loss covers every legal continuation even if this node had not needed expansion yet.
  if(node.exact_winner>=0 && node.exact_winner!=node.player)for(auto& edge:node.edges){edge.exact_winner=node.exact_winner;edge.distance=node.distance;edge.bound=true;}
  if(tactics)classify(path,node);
  if(graph){
   // A live expanded peer of the position hands over its edge proofs; the shared outcome keeps them when no peer
   // is alive (a win's witnesses, a loss's per-move resistances).
   if(auto list=positions.find(node.position);list!=positions.end())
    for(auto& w:std::vector(list->second))if(auto peer=w.lock())if(peer.get()!=&node && peer->expanded && copy(*peer,node))settle(node);
   if(auto o=outcomes.find(node.position);o!=outcomes.end())apply(o->second,node);
   if(shared)link(path,node);
  }
  // A certificate adds its witness as a winning edge; settle keeps any shorter tactical win found by classify.
  if(exact>=0){for(auto& edge:node.edges)if(edge.action==witness && (edge.exact_winner!=exact || edge.distance>distance)){edge.exact_winner=exact;edge.distance=distance;edge.bound=true;}settle(node);}
  node.pending=false;if(at_root)schedule(int(std::count_if(node.edges.begin(),node.edges.end(),[](auto& e){return e.eligible;})));backup(path,node.exact_winner<0?node.value:node.exact_winner==node.player?1:-1);requests.erase(found);
 }
 // Installs a caller-verified certificate at pending leaf `id`: its first turn `moves` and `turns`, the most attacker
 // turns on any certificate path, which bound the win within move_count + 4 * (turns - 1) placements.
 void prove(int id,const int64_t* history,int count,int player,int remaining,const int64_t* moves,int move_count,int turns){
  auto found=requests.find(id);if(found==requests.end())throw std::runtime_error("Unknown proof request");
  auto& path=found->second;
  if(count!=int(path.history.size()))throw std::runtime_error("Proof history mismatch");
  Board position;for(int i=0;i<count;++i){Cell c{history[2*i],history[2*i+1]};if(c!=path.history[i])throw std::runtime_error("Proof history mismatch");position.make(c);}
  if(move_count<1 || move_count>remaining || turns<1)throw std::runtime_error("Invalid proof turn");
  const int distance=move_count+4*(turns-1);
  Cell witness{moves[0],moves[1]};
  if(player!=position.player || remaining!=position.remaining || !position.legal(witness))throw std::runtime_error("Proof phase or move mismatch");
  Board after=position;for(int i=0;i<move_count;++i){Cell c{moves[2*i],moves[2*i+1]};if(!after.legal(c))throw std::runtime_error("Illegal proof turn");after.make(c);}
  if(after.winner<0 && after.player==player)throw std::runtime_error("Incomplete proof turn");
  Node* node=path.leaf;
  auto legal=position.legal_moves();std::vector<int64_t> actions;for(auto c:legal){actions.push_back(c.q);actions.push_back(c.r);}
  std::vector<double> zeros(legal.size());fulfill(id,actions.data(),zeros.data(),zeros.data(),int(legal.size()),player,witness,distance);
  if(move_count==2){
   position.make(witness);
   if(graph){
    // An existing node of this turn context takes the second stone as its witness instead of being replaced.
    auto [p,c]=keys(position);
    if(auto existing=nodes[c].lock()){
     existing->parents.push_back(node->weak_from_this());
     const Outcome o{player,player,distance-1,int(position.cells.size()),true,{EdgeProof{Cell{moves[2],moves[3]},player,distance-1,true}}};
     // The outcome keeps the witness even while the node is unexpanded; its parents and peers take the proof.
     if(apply(o,*existing))revise(*existing);
     record(p,o);
     for(auto& w:std::vector(positions[p]))if(auto peer=w.lock())if(peer!=existing)share(*existing,outcomes[p],*peer);
     for(auto& e:node->edges)if(e.action==witness){e.child=existing;break;}
     return;
    }
   }
   auto next_legal=position.legal_moves();
   auto child=std::make_shared<Node>();child->player=player;child->remaining=1;child->expanded=true;child->exact_winner=player;child->distance=distance-1;child->bound=true;
   for(auto c:next_legal){Edge e;e.action=c;e.prior=1./next_legal.size();e.eligible=c==Cell{moves[2],moves[3]};if(e.eligible){e.exact_winner=player;e.distance=distance-1;e.bound=true;}child->edges.push_back(std::move(e));}
   if(graph){auto [p,c]=keys(position);child->position=p;child->context=c;child->stones=int(position.cells.size());child->n=1;child->q=1;child->parents.push_back(node->weak_from_this());nodes[c]=child;positions[p].push_back(child);if(shared)store[c]=child;learn(*child);}
   for(auto& e:node->edges)if(e.action==witness){e.child=std::move(child);break;}
  }
 }
 void cancel(){for(auto& [id,path]:requests){path.leaf->pending=false;for(auto [node,index]:path.edges)--node->edges[index].pending;if(!path.edges.empty()){--root->edges[path.edges.front().second].epoch;--started;}}requests.clear();}
 void advance(Cell action){if(!requests.empty()||!board.legal(action))throw std::runtime_error("Invalid advance");std::shared_ptr<Node> next;int winner=-1,distance=-1;bool bound=false;priority.clear();defence.clear();hold=false;
  for(auto& e:root->edges)if(e.action==action){winner=e.exact_winner;distance=e.distance;bound=e.bound;next=shared?e.child:std::move(e.child);break;}
  board.make(action);root=next?std::move(next):child_here();root->player=board.player;budget=started=completed=0;
  // A shared graph keeps the siblings and every earlier position; a new root joins its stored parents.
  if(shared){std::vector<Cell> history;for(auto& u:board.history)history.push_back(u.c);adopt(root,history);lineage.clear();root->used=++clock;++version;}
  if(winner>=0 && winner!=root->player){root->exact_winner=winner;root->distance=distance-1;root->bound=bound;}
  if(graph){
   std::erase_if(nodes,[](const auto& entry){return entry.second.expired();});
   for(auto& [key,list]:positions)std::erase_if(list,[](const auto& w){return w.expired();});
   std::erase_if(positions,[](const auto& entry){return entry.second.empty();});
   // Stones are never removed, so positions with fewer stones than the board can not recur (a shared graph keeps
   // them for roots that return to them).
   if(!shared)std::erase_if(outcomes,[&](const auto& entry){return entry.second.stones<int(board.cells.size());});
  }
  // A root won without a known witness (a winning edge classified without a child, or a shared outcome that kept
  // none) needs one. A shared witness is installed when the root expands, which the first request does even for an
  // exact root (only an expanded exact root stops the search); otherwise immediate tactical choices are
  // reconstructed on the actual board, and general certificates retain their two-placement child.
  auto shared=graph?outcomes.find(root->position):outcomes.end();
  const bool witnessed=shared!=outcomes.end() && shared->second.player==root->player && shared->second.witnessed();
  if((winner==root->player || root->exact_winner==root->player) && !root->expanded && board.winner<0 && !witnessed){
   root->exact_winner=-1;
   if(tactics){Path path;capture(path);if(!path.own.empty()){root->expanded=true;root->remaining=path.remaining;for(auto c:path.legal){Edge e;e.action=c;e.prior=1./path.legal.size();root->edges.push_back(std::move(e));}classify(path,*root);}}
  }
 }
};
thread_local std::string error;
// Encode a pending, non-terminal leaf directly from the history and complete legal list captured by search.
// This is the deterministic hexcrop layout; no rules-board replay or dense legal-mask regeneration is needed.
int encode(const Path& path,uint8_t* planes,int capacity,int64_t* cells,int64_t* info){
 auto transform=[](Cell c,int k){if(k>=6)std::swap(c.q,c.r);for(int i=0;i<k%6;++i)c={-c.r,c.q+c.r};return c;};
 std::array<int64_t,3> low{},high{};bool empty=true;
 auto include=[&](Cell c){std::array<int64_t,3> v{c.q,c.r,c.q+c.r};if(empty){low=high=v;empty=false;}else for(int i=0;i<3;++i){low[i]=std::min(low[i],v[i]);high[i]=std::max(high[i],v[i]);}};
 auto bounds=[&](bool far){empty=true;for(auto c:path.history)include(c);if(!far)for(auto c:path.legal)include(c);};
 auto frame=[&](int k,int halo){auto q=transform({1,0},k),r=transform({0,1},k);std::array<int64_t,4> f{};for(int i=0;i<2;++i){int a=i?q.r:q.q,b=i?r.r:r.q,axis=a&&b?2:b?1:0;bool positive=(a?a:b)>0;f[i]=(positive?low[axis]:-high[axis])-halo;f[i+2]=high[axis]-low[axis]+1+2*halo;}return f;};
 bounds(false);int symmetry=0,halo=0;int64_t side=std::numeric_limits<int64_t>::max();
 auto choose=[&]{side=std::numeric_limits<int64_t>::max();for(int k=0;k<12;++k){auto f=frame(k,halo);auto need=std::max(f[2],f[3]);if(need<side){side=need;symmetry=k;}}};
 choose();if(side>256){halo=4;bounds(true);choose();if(side>256)return -2;}
 int size=0;for(int b:{24,32,40,48,64,96,128,192,256})if(b>=side){size=b;break;}
 auto f=frame(symmetry,halo);int64_t ox=(size-f[2])/2,oy=(size-f[3])/2;
 auto xy=[&](Cell c){auto p=transform(c,symmetry);return Cell{p.q+ox-f[0],p.r+oy-f[1]};};
 const int area=size*size,n=int(path.history.size());int far=0;
 if(planes){if(capacity<8*area)throw std::runtime_error("Leaf plane buffer too small");std::fill(planes,planes+8*area,0);}
 for(size_t i=0;i<path.legal.size();++i){auto p=xy(path.legal[i]);bool inside=!halo || (p.q>=ox && p.r>=oy && p.q<ox+f[2] && p.r<oy+f[3]);int64_t index=inside?p.r*size+p.q:-1;if(cells)cells[i]=index;if(planes && inside)planes[2*area+index]=1;far+=!inside;}
 if(planes){
  for(int i=0;i<n;++i){auto p=xy(path.history[i]);int owner=((i+1)/2)%2;planes[(owner==path.player?0:area)+p.r*size+p.q]=1;}
  for(int64_t y=oy;y<oy+f[3];++y)std::fill(planes+3*area+y*size+ox,planes+3*area+y*size+ox+f[2],1);
  std::fill(planes+(path.remaining==1?4:5)*area,planes+(path.remaining==1?5:6)*area,1);
  int start=path.remaining==2 || !n?n:n-1;
  if(start<n){auto p=xy(path.history.back());planes[6*area+p.r*size+p.q]=1;}
  for(int i=std::max(0,start-2);i<start;++i){auto p=xy(path.history[i]);planes[7*area+p.r*size+p.q]=1;}
 }
 if(info){std::array<int64_t,9> metadata{int64_t(path.legal.size()),path.player,path.remaining,symmetry,f[0],f[1],ox,oy,far};std::copy(metadata.begin(),metadata.end(),info);}
 return size;
}
}
extern "C" {
HX_API const char* hxg_error(){return gumbel::error.c_str();}
// Returns crop side, -2 for an unencodable span, or 0 on invalid request/buffer. info has nine int64 entries;
// cells has hxg_legal entries, planes has capacity bytes. Null outputs query the required layout.
HX_API int hxg_encode(void* p,int id,uint8_t* planes,int capacity,int64_t* cells,int64_t* info){try{auto& requests=static_cast<gumbel::Tree*>(p)->requests;auto found=requests.find(id);if(found==requests.end())throw std::runtime_error("Unknown encode request");return gumbel::encode(found->second,planes,capacity,cells,info);}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API void* hxg_new(uint64_t seed){try{return new gumbel::Tree(seed);}catch(...){return nullptr;}}
HX_API void hxg_free(void* p){delete static_cast<gumbel::Tree*>(p);}
HX_API int hxg_tactics(void* p,int enabled){auto& t=*static_cast<gumbel::Tree*>(p);if(t.root->expanded || !t.requests.empty())return 0;t.tactics=enabled!=0;return 1;}
// Switches graph search on (enabled != 0) or off before the root is expanded: transposed turn contexts share one node
// and proven outcomes are shared by position.
HX_API int hxg_graph(void* p,int enabled){auto& t=*static_cast<gumbel::Tree*>(p);if(t.root->expanded || !t.requests.empty())return 0;
 t.graph=enabled!=0;t.nodes.clear();t.outcomes.clear();t.positions.clear();
 if(t.graph){auto [position,context]=gumbel::keys(t.board);t.root->position=position;t.root->stones=int(t.board.cells.size());t.nodes[context]=t.root;t.positions[position].push_back(t.root);}
 return 1;}
// Makes the tree a shared game graph (graph search whose store keeps every node until evicted) before the root is
// expanded; `limit` bounds the expanded nodes kept between searches (0: no bound). 0 with the error set otherwise.
HX_API int hxg_share(void* p,int64_t limit){auto& t=*static_cast<gumbel::Tree*>(p);if(t.root->expanded || !t.requests.empty() || limit<0){gumbel::error="A shared graph needs an unexpanded root and a nonnegative limit";return 0;}
 if(!hxg_graph(p,1))return 0;
 t.shared=true;t.limit=size_t(limit);t.root->context=gumbel::keys(t.board).second;t.store[t.root->context]=t.root;return 1;}
// Shared graph: moves the root to the position after `history` (n int64 q/r pairs), keeping every node's statistics
// (Tree::root_at); 0 with the error set for an illegal history, an unshared tree or pending requests.
HX_API int hxg_root_at(void* p,const int64_t* history,int n){try{std::vector<Cell> h;for(int i=0;i<n;++i)h.push_back({history[2*i],history[2*i+1]});static_cast<gumbel::Tree*>(p)->root_at(h);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
// Shared graph: the number of root changes so far (hxg_root_at, hxg_advance), for callers that track the root.
HX_API int64_t hxg_root_version(void* p){return static_cast<gumbel::Tree*>(p)->version;}
// Shared graph store: out = {stored nodes, expanded stored nodes, nodes evicted so far, limit, evicted summaries kept,
// proven outcomes kept}; 0 when unshared.
HX_API int hxg_store(void* p,int64_t* out){auto& t=*static_cast<gumbel::Tree*>(p);if(!t.shared)return 0;int64_t expanded=0;for(auto& [key,n]:t.store)expanded+=n->expanded;
 out[0]=int64_t(t.store.size());out[1]=expanded;out[2]=t.evicted;out[3]=int64_t(t.limit);out[4]=int64_t(t.evicted_stats.size());out[5]=int64_t(t.outcomes.size());return 1;}
// Completed Q in value units for the root's mover, per root edge in hxg_stats order (Tree::completed_q); the edge
// count, 0 before the root is expanded.
HX_API int hxg_q(void* p,double* out){auto& t=*static_cast<gumbel::Tree*>(p);auto& n=*t.root;if(!n.expanded)return 0;if(t.shared)t.current(n);int maximum=0;auto& q=t.completed_q(n,maximum);if(out)std::copy(q.begin(),q.end(),out);return int(q.size());}
// Sets the least Q range of the completed-Q rescale (0, the default, keeps 1e-8) for every later search and target;
// 0 with no change when `floor` is negative or not finite.
HX_API int hxg_q_range_floor(void* p,double floor){if(!std::isfinite(floor) || floor<0){gumbel::error="Invalid Q range floor";return 0;}static_cast<gumbel::Tree*>(p)->range_floor=floor;return 1;}
// Sets the uniform share of the root's candidate sampling (Tree::sampling; 0, the default, samples by the prior)
// for every later search; 0 with no change unless 0 <= `noise` < 1.
HX_API int hxg_root_noise(void* p,double noise){if(!(noise>=0 && noise<1)){gumbel::error="Invalid root noise";return 0;}static_cast<gumbel::Tree*>(p)->root_noise=noise;return 1;}
// Diagnostic census of the structure reachable from the root: out = {nodes, expanded, exact, expanded nodes whose turn
// context was already expanded elsewhere (tree duplicates; 0 in a graph)}.
HX_API int hxg_census(void* p,int64_t* out){auto& t=*static_cast<gumbel::Tree*>(p);
 std::unordered_map<const gumbel::Node*,int> seen;std::unordered_map<gumbel::Key,int,gumbel::KeyHash> contexts;int64_t nodes=0,expanded=0,exact=0,duplicates=0;
 Restore restore(t.board);
 std::function<void(const gumbel::Node&)> walk=[&](const gumbel::Node& n){
  if(!seen.emplace(&n,1).second)return;
  ++nodes;exact+=n.exact_winner>=0;
  if(!n.expanded)return;
  ++expanded;duplicates+=contexts[gumbel::keys(t.board).second]++>0;
  for(auto& e:n.edges)if(e.child){t.board.make(e.action);walk(*e.child);t.board.undo();}
 };
 walk(*t.root);out[0]=nodes;out[1]=expanded;out[2]=exact;out[3]=duplicates;return 1;}
HX_API int hxg_exact(void* p){auto& t=*static_cast<gumbel::Tree*>(p);return t.board.winner>=0?t.board.winner:t.root->exact_winner;}
// Placements within which hxg_exact's winner completes six from the root (0 on a finished board), -1 when not exact.
HX_API int hxg_distance(void* p){auto& t=*static_cast<gumbel::Tree*>(p);return t.board.winner>=0?0:t.root->exact_winner>=0?t.root->distance:-1;}
HX_API int hxg_begin(void* p,int simulations,int sample){try{static_cast<gumbel::Tree*>(p)->begin(simulations,sample);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxg_next(void* p){try{return static_cast<gumbel::Tree*>(p)->request();}catch(const std::exception& e){gumbel::error=e.what();return -2;}}
HX_API int hxg_history(void* p,int id,int64_t* out){auto& h=static_cast<gumbel::Tree*>(p)->requests.at(id).history;if(out)for(int i=0;i<int(h.size());++i){out[2*i]=h[i].q;out[2*i+1]=h[i].r;}return int(h.size());}
// Legal moves of a pending request in sorted (q, r) order, the actions hxg_fulfill must be given.
HX_API int hxg_legal(void* p,int id,int64_t* out){auto& l=static_cast<gumbel::Tree*>(p)->requests.at(id).legal;if(out)for(int i=0;i<int(l.size());++i){out[2*i]=l[i].q;out[2*i+1]=l[i].r;}return int(l.size());}
HX_API int hxg_fulfill(void* p,int id,const int64_t* a,const double* logits,const double* q,int n){try{static_cast<gumbel::Tree*>(p)->fulfill(id,a,logits,q,n);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
// Caller must independently verify the strategy certificate before this entry.
// Exact history and placement phase prevent applying it to a different request.
HX_API int hxg_prove(void* p,int id,const int64_t* h,int n,int player,int remaining,const int64_t* moves,int count,int turns){try{static_cast<gumbel::Tree*>(p)->prove(id,h,n,player,remaining,moves,count,turns);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API void hxg_cancel(void* p){static_cast<gumbel::Tree*>(p)->cancel();}
HX_API int hxg_advance(void* p,int64_t q,int64_t r){try{static_cast<gumbel::Tree*>(p)->advance({q,r});return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
// Final selection among eligible edges at the highest epoch this search reached; when proofs removed every visited edge
// the unvisited survivors compete. No simulation started (an expired budget) scores nothing.
HX_API int hxg_stats(void* p,int64_t* actions,int* visits,double* values,double* scores){auto& t=*static_cast<gumbel::Tree*>(p);auto& n=*t.root;if(!n.expanded)return 0;if(t.shared)t.current(n);auto q=t.transformed(n);int max_epoch=0,searched=0;for(auto& e:n.edges){searched=std::max(searched,e.epoch);if(e.eligible)max_epoch=std::max(max_epoch,e.epoch);}for(int i=0;i<int(n.edges.size());++i){auto& e=n.edges[i];if(actions){actions[2*i]=e.action.q;actions[2*i+1]=e.action.r;visits[i]=e.visits;values[i]=t.value(n,e);scores[i]=e.eligible && (n.exact_winner>=0 || (searched && e.epoch==max_epoch))?e.gumbel+e.logit+q[i]+t.bonus(e):-std::numeric_limits<double>::infinity();}}return int(n.edges.size());}
HX_API int hxg_policy(void* p,double* out){auto& t=*static_cast<gumbel::Tree*>(p);auto& n=*t.root;if(!n.expanded)return 0;if(t.shared)t.current(n);auto q=t.transformed(n);double maximum=-1e300,total=0;for(int i=0;i<int(q.size());++i){q[i]=n.edges[i].eligible?q[i]+n.edges[i].logit+t.bonus(n.edges[i]):-std::numeric_limits<double>::infinity();maximum=std::max(maximum,q[i]);}for(auto& v:q){v=std::exp(v-maximum);total+=v;}if(out)for(int i=0;i<int(q.size());++i)out[i]=q[i]/total;return int(q.size());}
HX_API int hxg_completed(void* p){return static_cast<gumbel::Tree*>(p)->completed;}
// Completion includes an exact root, but never permits advancement while leaf reservations are outstanding.
HX_API int hxg_done(void* p){return static_cast<gumbel::Tree*>(p)->done();}
// Arms (enabled != 0) or clears the hold of the current search: hxg_next returns -3 once the search reaches its last
// halving boundary (the end of the search when it never halves) with no request pending, until the hold is cleared.
HX_API int hxg_hold(void* p,int enabled){static_cast<gumbel::Tree*>(p)->hold=enabled!=0;return 1;}
// Root actions (n cells, int64 q/r pairs) the opening phase of the current search samples before any other.
HX_API int hxg_priority(void* p,const int64_t* cells,int n){auto& t=*static_cast<gumbel::Tree*>(p);t.priority.clear();for(int i=0;i<n;++i)t.priority.push_back({cells[2*i],cells[2*i+1]});return 1;}
// Verified threat-breaking candidates. Search-local bonuses never change stored network logits or exact winners.
HX_API int hxg_defence(void* p,const int64_t* cells,const double* bonuses,int n){try{
 auto& t=*static_cast<gumbel::Tree*>(p);
 if(t.started || !t.requests.empty() || n<0 || n>t.budget)throw std::runtime_error("Defence needs an unstarted search and at most budget candidates");
 for(int i=0;i<n;++i)if(!t.board.legal({cells[2*i],cells[2*i+1]}) || !std::isfinite(bonuses[i]) || bonuses[i]<0)throw std::runtime_error("Invalid defence candidate");
 t.defence.clear();for(int i=0;i<n;++i)t.defence[{cells[2*i],cells[2*i+1]}]=bonuses[i];
 t.schedule(t.root->expanded?int(std::count_if(t.root->edges.begin(),t.root->edges.end(),[](auto& e){return e.eligible;})):int(t.board.legal_moves().size()));return 1;
 }catch(const std::exception& e){gumbel::error=e.what();return 0;}}
// Caller must hold a verified proof that `winner` wins after root edge (q, r) within `distance` placements, the
// edge's own included; see Tree::mark.
HX_API int hxg_mark_exact(void* p,int64_t q,int64_t r,int winner,int distance){try{static_cast<gumbel::Tree*>(p)->mark({q,r},winner,distance);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
}
