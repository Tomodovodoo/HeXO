// Native placement-tree scheduling. Algorithm reference: DeepMind mctx.
#include "hexo.cpp"
#include <memory>
#include <memory_resource>
#include <utility>
#include <random>
#include <map>
#include <numeric>
#include <array>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <bitset>
#ifdef HEXO_RECLAIM_PROFILE
#include <chrono>
#include <cstdio>
#endif
namespace gumbel {
struct Node;
struct Tree;
struct Edge;
void materialized(Node*,Edge&);
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
// Derive a continuation from its parent's coloured set. Only the last turn's
// input context changes; the earlier stones need neither copying nor hashing.
std::pair<Key,Key> child_keys(Key parent,const std::vector<Cell>& history,Cell action) {
 const size_t n=history.size(),size=n+1;const int player=int((n+1)/2%2),remaining=n==0 || n%2==0?1:2;
 const int next_player=int((size+1)/2%2),next_remaining=size%2==0?1:2;auto h=CellHash{}(action);
 Key position{parent.a-mix(player*3+remaining+17)+mix(next_player*3+next_remaining+17)+mix(h^mix(player+1)),
              parent.b-mix(player*3+remaining+71)+mix(next_player*3+next_remaining+71)+mix(h+mix(player+911))};
 Key context=position;const size_t start=size%2?size:size-1;
 if(start<size){context.a^=mix(h+0x51);context.b^=mix(h+0x93);}
 for(size_t i=start>=2?start-2:0;i<start;++i){auto c=CellHash{}(i==n?action:history[i]);context.a+=mix(c+0x7f1);context.b+=mix(c+0x3c9);}
 return {position,context};
}
// Removing a possible last placement identifies its predecessor's rule position.
// Neural context is checked separately when that predecessor expands.
Key predecessor(Key child,size_t stones,Cell action) {
 const int mover=int(stones/2%2),player=int((stones+1)/2%2),remaining=stones%2?2:1;
 const int before=stones==1 || stones%2?1:2;auto h=CellHash{}(action);
 return {child.a-mix(player*3+remaining+17)+mix(mover*3+before+17)-mix(h^mix(mover+1)),
         child.b-mix(player*3+remaining+71)+mix(mover*3+before+71)-mix(h+mix(mover+911))};
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
// Untouched legal actions share an immutable empty state. Allocate mutable
// search/proof state only when an action receives work or a changed verdict.
struct EdgeState {
 double sum=0;int visits=0,pending=0,exact_winner=-1,distance=-1;
 bool eligible=true,bound=false,indexed=false,empty=true;
 std::shared_ptr<Node> child;std::pmr::memory_resource* resource;Node* owner=nullptr;
 explicit EdgeState(std::pmr::memory_resource* r):resource(r){}
};
struct Edge {
 Cell action;double logit=0,weight=-1;EdgeState* data;
 static EdgeState& empty_state(){static EdgeState state(std::pmr::new_delete_resource());return state;}
 explicit Edge(EdgeState* empty=&empty_state()):data(empty){}
 Edge(const Edge&)=delete;Edge& operator=(const Edge&)=delete;
 Edge(Edge&& other)noexcept:action(other.action),logit(other.logit),weight(other.weight),data(std::exchange(other.data,nullptr)){}
 Edge& operator=(Edge&& other)noexcept {
  if(this!=&other){release();action=other.action;logit=other.logit;weight=other.weight;data=std::exchange(other.data,nullptr);}return *this;
 }
 ~Edge(){release();}
 const EdgeState& read()const {return *data;}
 EdgeState& write(){
  if(data->empty){auto* resource=data->resource;auto* state=static_cast<EdgeState*>(resource->allocate(sizeof(EdgeState),alignof(EdgeState)));
   std::construct_at(state,*data);state->empty=false;data=state;if(state->owner)materialized(state->owner,*this);}
  return *data;
 }
 void eligibility(bool value){if(read().eligible!=value)write().eligible=value;}
 void release()noexcept {if(data && !data->empty){auto* resource=data->resource;std::destroy_at(data);resource->deallocate(data,sizeof(EdgeState),alignof(EdgeState));}}
};
// Legal lists are large and live together. Pool their buffers within one game,
// recycling evicted nodes' blocks instead of making one heap allocation per node.
// A node retains the resource because it can outlive the GameStore's indices.
struct EdgeMemory {
#ifdef HEXO_RECLAIM_PROFILE
 struct Upstream : std::pmr::memory_resource {
  uint64_t bytes=0,peak=0,allocations=0;
  void* do_allocate(size_t n,size_t alignment)override {auto p=std::pmr::new_delete_resource()->allocate(n,alignment);bytes+=n;peak=std::max(peak,bytes);++allocations;return p;}
  void do_deallocate(void* p,size_t n,size_t alignment)override {bytes-=n;std::pmr::new_delete_resource()->deallocate(p,n,alignment);}
  bool do_is_equal(const std::pmr::memory_resource& other)const noexcept override {return this==&other;}
 } upstream;
 std::pmr::unsynchronized_pool_resource pool{std::pmr::pool_options{8,262144},&upstream};
 std::pmr::unsynchronized_pool_resource states{std::pmr::pool_options{1024,sizeof(EdgeState)},&upstream};
 ~EdgeMemory(){auto start=std::chrono::steady_clock::now();states.release();pool.release();auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
  std::fprintf(stderr,"HEXO_RECLAIM {\"component\":\"edge_memory\",\"peak_bytes\":%llu,\"allocations\":%llu,\"remaining_bytes\":%llu,\"release_ns\":%llu}\n",
   (unsigned long long)upstream.peak,(unsigned long long)upstream.allocations,(unsigned long long)upstream.bytes,(unsigned long long)ns);
 }
#else
 std::pmr::unsynchronized_pool_resource pool{std::pmr::pool_options{8,262144}};
 std::pmr::unsynchronized_pool_resource states{std::pmr::pool_options{1024,sizeof(EdgeState)}};
#endif
};
// Cold actions share the unknown completed Q. Each small block retains its
// logit-relative mass and best action; removing a dominant prior cannot cancel
// the remaining mass or hide logits whose original exponent underflowed.
struct ColdBlock {double mass=0;int best=-1;bool dirty=true;};
// An exact winner comes with a distance: the placements within which that winner completes six from this position
// (an edge counts its own placement) against any defence, combined by min at the winner's choices and max at the
// loser's. It is exact for terminal and tactical results; `bound` marks an upper bound, which certificates give.
// With graph search a node also keeps its visits `n`, its utility `q` for its mover (the MCGS value) and its parents.
// In a shared graph `dirty` marks a value that a descendant's statistics have changed since it was computed, `used`
// the last search step that touched the node, `context` its key in the store, and `carried` and `carried_sum` the
// visits and value sum (for its mover) an evicted node of its context had when it left the store, less its own
// network value, which the node's expansion supplies again.
// An archived node must describe its coloured set after an ancestor disappears.
// Shared immutable history links cost one allocation per new searched position,
// only when archival is enabled; no Board or full history is copied per leaf.
struct HistoryLink {std::shared_ptr<const HistoryLink> before;Cell cell;int stones;
 HistoryLink(std::shared_ptr<const HistoryLink> p,Cell c):before(std::move(p)),cell(c),stones(before?before->stones+1:1){}
};
struct Node : std::enable_shared_from_this<Node> { int player=0,remaining=1,exact_winner=-1,distance=-1,n=0,stones=0,carried=0;bool expanded=false,pending=false,bound=false,dirty=false,indexed=false,dormant=false,raw_known=false;bool* archive_dirty=nullptr;double value=0,q=0,carried_sum=0,policy_mass=0,raw=0;uint64_t used=0,losses_seen=0;Cell first;Key position,context;std::shared_ptr<EdgeMemory> memory;std::shared_ptr<const HistoryLink> history;EdgeState empty;std::pmr::vector<Edge> edges;std::pmr::vector<int> tracked;std::vector<std::weak_ptr<Node>> parents;
 std::pmr::vector<int> active;std::pmr::vector<ColdBlock> cold;bool selective=false,cold_dirty=true;int cold_best=-1;double cold_mass=0;
 static constexpr int block_size=32;
 explicit Node(std::shared_ptr<EdgeMemory> resource=std::make_shared<EdgeMemory>()):memory(std::move(resource)),empty(&memory->states),edges(&memory->pool),tracked(&memory->pool),active(&memory->pool),cold(&memory->pool){empty.owner=this;}
 void archive_changed(){if(archive_dirty)*archive_dirty=true;}
 void activate(Edge& edge){
  archive_changed();
  if(!selective)return;
  int i=int(&edge-edges.data());active.insert(std::lower_bound(active.begin(),active.end(),i),i);
  cold[i/block_size].dirty=true;cold_dirty=true;
 }
 void selection_index(){
  if(selective)return;
  cold.resize((edges.size()+block_size-1)/block_size);
  for(int i=0;i<int(edges.size());++i){auto& e=edges[i];
   if(e.read().empty)e.data=&empty;
   else {e.data->owner=this;active.push_back(i);}
  }
  selective=true;
 }
 void cold_summary(){
  selection_index();if(!cold_dirty)return;
  cold_best=-1;cold_mass=0;
  for(int b=0;b<int(cold.size());++b){auto& block=cold[b];
   if(block.dirty){
    int begin=b*block_size,end=std::min(begin+block_size,int(edges.size()));block.best=-1;block.mass=0;
    for(int i=begin;i<end;++i)if(edges[i].read().empty && (block.best<0 || edges[i].logit>edges[block.best].logit))block.best=i;
    if(block.best>=0){double factor=std::exp(-edges[block.best].logit);
     for(int i=begin;i<end;++i)if(edges[i].read().empty){auto& e=edges[i];
      block.mass+=e.weight>=std::numeric_limits<double>::min() && std::isfinite(factor)?e.weight*factor:std::exp(e.logit-edges[block.best].logit);
     }
    }
    block.dirty=false;
   }
   if(block.best>=0 && (cold_best<0 || edges[block.best].logit>edges[cold_best].logit))cold_best=block.best;
  }
  if(cold_best>=0)for(auto& block:cold)if(block.best>=0)cold_mass+=block.mass*std::exp(edges[block.best].logit-edges[cold_best].logit);
  cold_dirty=false;
 }
 // Synthetic proof expansions use a uniform policy; neural expansions retain
 // their normalization once per node instead of once per legal action.
 double prior(const Edge& e)const {return policy_mass?e.weight/policy_mass:1./edges.size();}
 // Graph updates need edges that have held a child or visits. This is not an
 // action cap or a proof-coverage list; `edges` always holds every legal move.
 const std::pmr::vector<int>& tracked_edges(){
  if(!indexed){
   archive_changed();
   for(size_t i=0;i<edges.size();++i)if(edges[i].read().child || edges[i].read().visits){edges[i].write().indexed=true;tracked.push_back(int(i));}
   indexed=true;
  }
  return tracked;
 }
 void track(Edge& edge){
  tracked_edges();if(edge.read().indexed)return;
  archive_changed();
  int index=int(&edge-edges.data());tracked.insert(std::lower_bound(tracked.begin(),tracked.end(),index),index);edge.write().indexed=true;
 }
};
void materialized(Node* node,Edge& edge){node->activate(edge);}
// A pending leaf: its history, its legal moves in sorted order and, with tactics, the side to move's completions
// (own) and the opponent's (threats), both restricted to fully legal ones.
struct Path { Node* leaf=nullptr;std::vector<std::pair<Node*,int>> edges;std::vector<Cell> history,legal;int player=0,remaining=1,round_slot=-1;std::vector<std::vector<Cell>> own,threats; };
struct Summary { int n=0;double q=0,value=0;Key position; };
struct ColouredCell {Cell cell;int player;bool operator==(const ColouredCell&)const=default;};
struct ColouredHash {size_t operator()(const ColouredCell& c)const{return CellHash{}(c.cell)^mix(c.player+1);}};
// Dormant nodes hold evidence, not scheduling rights. Each entry owns itself,
// independently of its ancestors. The bounded membership index ranks future
// reconvergence; only the exact neural context key authorizes reactivation.
struct Archive {
 static constexpr size_t slots=256;
 struct Entry {std::shared_ptr<Node> node;size_t bytes=0;};
 std::array<Entry,slots> entries;
 std::unordered_map<Key,size_t,KeyHash> contexts;
 std::unordered_map<ColouredCell,std::bitset<slots>,ColouredHash> membership;
 std::bitset<slots> occupied,compatible,conflicting;
 std::shared_ptr<const HistoryLink> focus;
 size_t limit,bytes=0;bool dirty=false,forward=false;uint64_t retained=0,reused=0,discarded=0;
 explicit Archive(size_t budget):limit(budget){}
 ~Archive(){for(auto& e:entries)if(e.node)e.node->archive_dirty=nullptr;}
 static size_t payload(const Node& n){
  size_t states=n.selective?n.active.size():std::count_if(n.edges.begin(),n.edges.end(),[](const Edge& e){return !e.read().empty;});
  // Charge every history link conservatively even when prefixes are shared.
  return sizeof(Node)+2*sizeof(void*)+n.edges.capacity()*sizeof(Edge)+states*sizeof(EdgeState)
   +(n.active.capacity()+n.tracked.capacity())*sizeof(int)+n.cold.capacity()*sizeof(ColdBlock)
   +n.parents.capacity()*sizeof(std::weak_ptr<Node>)+size_t(n.stones)*(sizeof(HistoryLink)+2*sizeof(void*));
 }
 size_t index_bytes()const{
  return sizeof(Archive)
   +(contexts.bucket_count()+membership.bucket_count())*sizeof(void*)
   +contexts.size()*(sizeof(decltype(contexts)::value_type)+2*sizeof(void*))
   +membership.size()*(sizeof(decltype(membership)::value_type)+2*sizeof(void*));
 }
 size_t total_bytes()const{return bytes+index_bytes();}
 int focus_stones()const{return focus?focus->stones:0;}
 void set_focus(std::shared_ptr<const HistoryLink> h){
  focus=std::move(h);
  compatible=occupied;conflicting.reset();
  for(auto p=focus;p;p=p->before){int player=(p->stones/2)%2;
   auto same=membership.find({p->cell,player});if(same==membership.end())compatible.reset();else compatible&=same->second;
   if(forward){auto opposite=membership.find({p->cell,1-player});if(opposite!=membership.end())conflicting|=opposite->second;}
   else if(compatible.none())break;
  }
 }
 void refresh(){bytes=0;for(auto& e:entries)if(e.node){e.bytes=payload(*e.node);bytes+=e.bytes;}dirty=false;}
 void insert(const std::shared_ptr<Node>& node){
  size_t slot=0;while(slot<slots && occupied[slot])++slot;
  if(slot==slots)throw std::runtime_error("Dormant archive has no free slot");
  auto& entry=entries[slot];entry={node,payload(*node)};bytes+=entry.bytes;contexts.emplace(node->context,slot);occupied.set(slot);compatible.set(slot);
  for(auto p=node->history;p;p=p->before)membership[{p->cell,(p->stones/2)%2}].set(slot);
  for(auto p=focus;p;p=p->before){int player=(p->stones/2)%2;
   auto same=membership.find({p->cell,player});if(same==membership.end() || !same->second[slot])compatible.reset(slot);
   if(forward){auto opposite=membership.find({p->cell,1-player});if(opposite!=membership.end() && opposite->second[slot])conflicting.set(slot);}
   else if(!compatible[slot])break;
  }
  node->dormant=true;node->archive_dirty=&dirty;++retained;
 }
 std::shared_ptr<Node> remove(size_t slot){
  auto node=std::move(entries[slot].node);node->archive_dirty=nullptr;bytes-=entries[slot].bytes;entries[slot].bytes=0;contexts.erase(node->context);occupied.reset(slot);compatible.reset(slot);conflicting.reset(slot);
  for(auto p=node->history;p;p=p->before){auto found=membership.find({p->cell,(p->stones/2)%2});found->second.reset(slot);if(found->second.none())membership.erase(found);}
  if(occupied.none()){contexts.rehash(0);membership.rehash(0);}
  return node;
 }
 template<class Allowed> size_t victim(Allowed allowed)const{
  size_t chosen=slots;double best=std::numeric_limits<double>::infinity();
  for(size_t i=0;i<slots;++i)if(occupied[i] && allowed(entries[i].node.get())){auto& e=entries[i];
   // Colour containment is necessary for forward reuse, not a reachability
   // proof. Keep incompatible entries less eagerly so undo remains useful.
   double benefit=(compatible[i]?4.:1.)*std::log1p(e.node->n)/(1+std::abs(e.node->stones-focus_stones()));
   double score=benefit/std::max(size_t(1),e.bytes);
   if(chosen==slots || score<best || (score==best && e.node->used<entries[chosen].node->used)){chosen=i;best=score;}
  }
  return chosen;
 }
};
struct GameStore {
 std::shared_ptr<EdgeMemory> memory=std::make_shared<EdgeMemory>();
 std::unordered_map<Key,std::weak_ptr<Node>,KeyHash> nodes;
 std::unordered_map<Key,std::vector<std::weak_ptr<Node>>,KeyHash> positions;
 struct Continuation {Cell action;std::weak_ptr<Node> child;};
 std::unordered_map<Key,std::vector<Continuation>,KeyHash> continuations;
 std::unordered_map<Key,Outcome,KeyHash> outcomes;
 std::unordered_map<Key,std::shared_ptr<Node>,KeyHash> store;
 std::unordered_map<Key,Summary,KeyHash> evicted_stats;
 std::unique_ptr<Archive> archive;Tree* primary=nullptr;
 size_t limit=0;uint64_t clock=0;int64_t evicted=0;void* scheduler_owner=nullptr;
 // Nodes discarded so far, and that count when evict last swept expired index entries.
 uint64_t released=0,swept=0;
 uint64_t half_losses=0;
 std::unordered_map<const void*,std::vector<Node*>> pins;
 // A native owner may observe installed evidence. Workers never call this.
 void* evidence_owner=nullptr;void (*evidence)(void*,Tree&,const Path&)=nullptr;
#ifdef HEXO_RECLAIM_PROFILE
 ~GameStore(){
  using Clock=std::chrono::steady_clock;
  auto ns=[](auto start){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count());};
  auto start=Clock::now();uint64_t edges=0,children=0,parents=0,bytes=0,count=store.size();
  for(auto& [key,node]:store){edges+=node->edges.size();parents+=node->parents.size();bytes+=node->edges.capacity()*sizeof(Edge);
   for(auto& edge:node->edges)children+=bool(edge.read().child);}
  auto inventory=ns(start);std::array<uint64_t,6> times;
  start=Clock::now();for(auto& [key,node]:store)node->edges.clear();auto edges_ns=ns(start);
  start=Clock::now();for(auto& [key,node]:store)std::pmr::vector<Edge>(node->edges.get_allocator()).swap(node->edges);auto edge_free_ns=ns(start);
  start=Clock::now();pins.clear();times[0]=ns(start);
  start=Clock::now();evicted_stats.clear();times[1]=ns(start);
  start=Clock::now();store.clear();times[2]=ns(start);
  start=Clock::now();outcomes.clear();times[3]=ns(start);
  start=Clock::now();positions.clear();times[4]=ns(start);
  start=Clock::now();nodes.clear();times[5]=ns(start);
  std::fprintf(stderr,"HEXO_RECLAIM {\"component\":\"game_store\",\"nodes\":%llu,\"legal_edges\":%llu,\"child_refs\":%llu,\"parent_refs\":%llu,\"edge_capacity_bytes\":%llu,\"inventory_ns\":%llu,\"edge_destruct_ns\":%llu,\"edge_free_ns\":%llu,\"pins_ns\":%llu,\"summaries_ns\":%llu,\"store_ns\":%llu,\"outcomes_ns\":%llu,\"positions_ns\":%llu,\"nodes_ns\":%llu}\n",
   (unsigned long long)count,(unsigned long long)edges,(unsigned long long)children,(unsigned long long)parents,(unsigned long long)bytes,(unsigned long long)inventory,
   (unsigned long long)edges_ns,(unsigned long long)edge_free_ns,(unsigned long long)times[0],(unsigned long long)times[1],(unsigned long long)times[2],(unsigned long long)times[3],(unsigned long long)times[4],(unsigned long long)times[5]);
 }
#endif
 bool pinned(const Node* node)const {
  for(auto& [view,list]:pins)if(std::find(list.begin(),list.end(),node)!=list.end())return true;
  return false;
 }
};
struct RootEdge { double gumbel=0,opening_q=0;int epoch=0;uint64_t credits=0; };
struct RootRound {
 std::vector<int> widths,ends,members,counts,limits;std::vector<double> opening;int active=-1;
};
struct RootSession {
 std::vector<RootEdge> edges;bool prepared=false,hold=false;
 int budget=0,started=0,completed=0,samples=0,last=0;uint64_t issued=0,cancelled=0;
 std::vector<int> sequence;std::vector<Cell> priority;std::map<Cell,double> defence;RootRound round;
};
struct Tree {
 Board board;std::shared_ptr<Node> root;std::map<int,Path> requests;
 // Graph search (opt-in): nodes shared by turn-context key, proven outcomes shared by
 // position key. Tree search gives every edge its own child and keeps both tables empty.
 std::shared_ptr<GameStore> state;
 bool graph=false,scheduler_owned=false;
 void owner_access()const {if(state->scheduler_owner && !scheduler_owned)throw std::runtime_error("Search belongs to its native scheduler");}
 decltype(GameStore::nodes)& nodes;decltype(GameStore::positions)& positions;decltype(GameStore::outcomes)& outcomes;
 // Shared game graph (opt-in, implies graph): `store` owns every node by turn-context key so roots can move to any
 // position (root_at) and back; an edge reads its child's visits and value (current); `limit` bounds the expanded
 // nodes kept (evict); `lineage` holds the stored positions the root's history passes through, each with the edge of
 // that history out of it (-1 when unexpanded), credited with each playout; `version` counts root changes.
 bool shared=false;size_t& limit;uint64_t& clock;int64_t& evicted;
 decltype(GameStore::store)& store;std::vector<std::pair<Node*,int>> lineage;int64_t version=0;
 std::vector<RootEdge> root_edges;bool root_prepared=false;
 std::unordered_map<Key,RootSession,KeyHash> root_sessions;
 uint64_t issued=0,cancelled=0,proof_retired=0;
 // Shared graph: the visits and value (for its mover) of evicted nodes, by context, for a node created again there;
 // evict keeps at most four times `limit` of them, those with the most visits.
 decltype(GameStore::evicted_stats)& evicted_stats;
 std::mt19937_64 rng;int budget=0,started=0,completed=0,next_id=1,samples=0,last=0;bool tactics=false,hold=false;std::vector<int> sequence;
 // Root actions sampled first in the opening phase of the current search; ordering only (set_priority).
 std::vector<Cell> priority;
 std::map<Cell,double> defence;
 double range_floor=0;  // least Q range of the completed-Q rescale (transformed)
  bool round_barrier=false;RootRound round;
  double root_noise=0;   // uniform share of the root's candidate sampling distribution (sampling)
 double bonus(const Edge& e)const {auto i=defence.find(e.action);return i==defence.end()?0:i->second;}
 std::vector<double> work;
 explicit Tree(uint64_t seed,std::shared_ptr<GameStore> game=std::make_shared<GameStore>()):
  root(std::make_shared<Node>(game->memory)),state(std::move(game)),nodes(state->nodes),positions(state->positions),outcomes(state->outcomes),
  limit(state->limit),clock(state->clock),evicted(state->evicted),store(state->store),evicted_stats(state->evicted_stats),rng(seed){pin();}
 ~Tree(){cancel();state->pins.erase(this);if(state->primary==this)state->primary=nullptr;}
 Tree(const Tree&)=delete;
 Tree& operator=(const Tree&)=delete;
 void pin(){if(!shared)return;reactivate(root);auto& list=state->pins[this];list.clear();list.push_back(root.get());
  for(auto [node,index]:lineage){reactivate(node->shared_from_this());list.push_back(node);}
 }
  void save_root(){if(shared && root_prepared)root_sessions[keys(board).second]={root_edges,true,hold,budget,started,completed,samples,last,issued,cancelled,sequence,priority,defence,round};}
 void restore_root(){
  root_edges.clear();root_prepared=false;hold=false;budget=started=completed=samples=last=0;issued=cancelled=0;
   sequence.clear();priority.clear();defence.clear();round={};
  if(auto old=root_sessions.find(keys(board).second);old!=root_sessions.end() && root->expanded && old->second.edges.size()==root->edges.size()){
   auto& r=old->second;root_edges=r.edges;root_prepared=r.prepared;hold=r.hold;
   budget=r.budget;started=r.started;completed=r.completed;samples=r.samples;last=r.last;issued=r.issued;cancelled=r.cancelled;
    sequence=r.sequence;priority=r.priority;defence=r.defence;round=r.round;
  }
  if(shared && completed<budget){std::vector<Cell> history;for(auto& u:board.history)history.push_back(u.c);lineage=this->prefixes(history);}
 }
 void prepare_root(){
  if(!root->expanded || root_prepared)return;
  root_edges.assign(root->edges.size(),{});
  if(!budget)return;  // Reading a cold view's metrics must not consume its future sampling randomness.
  for(auto& e:root_edges){double u=std::generate_canonical<double,53>(rng);e.gumbel=-std::log(-std::log(std::clamp(u,1e-15,1-1e-15)));}
  root_prepared=true;
  schedule(int(std::count_if(root->edges.begin(),root->edges.end(),[](const auto& e){return e.read().eligible;})));
  freeze_root_q();
 }
 // Root sampling uses the evidence available at this search's start. Keep it with the view's root session so
 // predictions from early candidates or other views cannot change the remaining Gumbel-top-k draws.
 void freeze_root_q(){
  if(!shared)return;
  current(*root);auto& q=transformed(*root);
  for(size_t i=0;i<q.size();++i)root_edges[i].opening_q=q[i];
 }
 bool done()const {return requests.empty() && (board.winner>=0 || (root->expanded && root->exact_winner>=0) || completed>=budget);}
 // Edge value for the node's mover: exact, else in a graph the child's MCGS value (which may come from other
 // parents), else the tree's running mean, else the node's own network value.
 double value(const Node& node,const Edge& e)const {
  if(e.read().exact_winner>=0)return e.read().exact_winner==node.player?1:-1;
  if(graph && e.read().child && e.read().child->n)return e.read().child->player==node.player?e.read().child->q:-e.read().child->q;
  return e.read().visits?e.read().sum/e.read().visits:node.value;
 }
 bool known(const Edge& e)const {return e.read().exact_winner>=0 || e.read().visits || (graph && e.read().child && e.read().child->n);}
 // The node for the tree's board as a new child: a fresh node, or with graph search the shared node of this turn
 // context, created with any outcome already proven for the position.
 void reactivate(const std::shared_ptr<Node>& node){
  if(!node->dormant)return;
  auto& archive=*state->archive;auto found=archive.contexts.find(node->context);
  if(found==archive.contexts.end())throw std::runtime_error("Unowned dormant node");
  archive.remove(found->second);node->dormant=false;store[node->context]=node;++archive.reused;node->used=clock;
 }
 void index_child(const std::shared_ptr<Node>& node,const Board& position){
  const auto n=position.history.size();if(!n)return;
  auto add=[&](Cell action){state->continuations[predecessor(node->position,n,action)].push_back({action,node});};
  add(position.history.back().c);
  if(n>=3 && n%2)add(position.history[n-2].c);
 }
 void prune_continuations(){
  for(auto& [key,list]:state->continuations)std::erase_if(list,[](const auto& e){return e.child.expired();});
  std::erase_if(state->continuations,[](const auto& e){return e.second.empty();});
 }
 void archive_focus(){if(state->archive)state->archive->set_focus(root->history);}
 std::shared_ptr<Node> child_here(std::shared_ptr<const HistoryLink> before={},bool descended=false) {
  if(!graph){auto n=std::make_shared<Node>(state->memory);n->player=board.player;return n;}
  auto [position,context]=keys(board);auto& slot=nodes[context];
  if(auto n=slot.lock()){reactivate(n);return n;}
  auto n=std::make_shared<Node>(state->memory);n->player=board.player;n->remaining=board.remaining;n->position=position;n->context=context;n->stones=int(board.cells.size());
  if(board.remaining==1 && !board.history.empty())n->first=board.history.back().c;
  if(state->archive){
   if(descended)n->history=std::make_shared<HistoryLink>(std::move(before),board.history.back().c);
   else for(auto& u:board.history)n->history=std::make_shared<HistoryLink>(n->history,u.c);
  }
  slot=n;positions[position].push_back(n);if(shared)index_child(n,board);
  if(shared){
   store[context]=n;n->used=clock;
   if(auto old=evicted_stats.find(context);old!=evicted_stats.end()){
    auto& o=old->second;n->n=o.n;n->q=o.q;n->value=o.value;n->carried=o.n-1;n->carried_sum=o.q*o.n-o.value;evicted_stats.erase(old);
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
   const auto& tracked=x->tracked_edges();
   if(i<tracked.size()){auto& e=x->edges[tracked[i++]];if(e.read().child && e.read().child->dirty)work.emplace_back(e.read().child.get(),0);continue;}
   Node* done=x;work.pop_back();
   refresh(*done);done->dirty=false;
  }
 }
 // Shared graph: brings the node's children up to date before its edges' values are read. An edge keeps its own
 // visits (the playouts that went through it, and the credits of roots below it on their history) and reads its
 // child's value, so a child reached by both orders of a turn is not weighted twice.
 void current(Node& node) {
  inherit_half_losses(node);
  for(int i:node.tracked_edges()){auto& e=node.edges[i];if(e.read().child)clean(*e.read().child);}
 }
 // Losing after A with one stone left proves every A,B continuation lost.
 // At the sibling half-turn after B, mark A without materializing all pairs.
 // Only new half-turn loss evidence rescans the existing legal edge list.
 void inherit_half_losses(Node& node) {
  if(state->scheduler_owner && !scheduler_owned)return;
  if(!graph || !node.expanded || node.remaining!=1 || !node.stones || node.losses_seen==state->half_losses)return;
  node.losses_seen=state->half_losses;bool changed=false;
  const auto old=CellHash{}(node.first);const int p=node.player;
  for(auto& edge:node.edges){
   const auto h=CellHash{}(edge.action);
   const Key other{node.position.a-mix(old^mix(p+1))+mix(h^mix(p+1)),
                   node.position.b-mix(old+mix(p+911))+mix(h+mix(p+911))};
   auto found=outcomes.find(other);
   if(found==outcomes.end() || found->second.winner==p || found->second.distance<2)continue;
   const auto& loss=found->second;
   // The alternate half-turn includes one remaining defender placement, just
   // as this edge does. A different response can only use a shorter distance.
   changed|=tighten(loss.winner,loss.distance,true,edge.write().exact_winner,edge.write().distance,edge.write().bound);
  }
  if(changed){settle(node);learn(node);revise(node);}
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
  if(e.read().visits && !child->n && e.read().exact_winner<0){e.write().visits=0;e.write().sum=0;}
  e.write().child=child;parent.track(e);child->archive_changed();child->parents.push_back(parent.weak_from_this());
  if(child->exact_winner>=0 && tighten(child->exact_winner,child->distance+1,child->bound,e.write().exact_winner,e.write().distance,e.write().bound)){
   settle(parent);learn(parent);revise(parent);
  }
  else stale(parent);
 }
 // Shared graph: find stored continuations by predecessor position, then check
 // the exact neural context before attaching. Untouched legal actions need no lookup.
 void link(const Path& path,Node& node) {
  auto found=state->continuations.find(node.position);if(found==state->continuations.end())return;
  auto& candidates=found->second;std::erase_if(candidates,[](const auto& e){return e.child.expired();});
  for(auto& candidate:candidates)if(auto child=candidate.child.lock()){
   if(child_keys(node.position,path.history,candidate.action).second!=child->context)continue;
   // Expanded legal lists are sorted. No policy move is removed by this index.
   auto e=std::lower_bound(node.edges.begin(),node.edges.end(),candidate.action,[](const Edge& e,Cell action){return e.action<action;});
   if(e!=node.edges.end() && e->action==candidate.action && !e->read().child)attach(node,*e,child);
  }
 }
 // Shared graph: attaches `node`, the position after `history`, under the expanded nodes that reach it by its last
 // stone: the position before it and, when the last two stones are one turn, the other order of that turn.
 void adopt(const std::shared_ptr<Node>& node,const std::vector<Cell>& history) {
  const size_t n=history.size();
  auto join=[&](const std::vector<Cell>& before,Cell action){
   auto found=nodes.find(keys(before).second);if(found==nodes.end())return;
   auto parent=found->second.lock();if(!parent || !parent->expanded)return;
   for(auto& e:parent->edges)if(e.action==action){if(!e.read().child)attach(*parent,e,node);return;}
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
 // Drop an archive entry or an ordinary evicted node. Its descendants remain
 // independently owned. Incoming edges freeze the last numerical evidence;
 // exact facts still belong to the rule-position table.
 void discard(const std::shared_ptr<Node>& node){
  clean(*node);
  for(auto& w:node->parents)if(auto p=w.lock())for(int i:p->tracked_edges()){auto& e=p->edges[i];if(e.read().child==node){
   e.write().sum=(e.read().exact_winner>=0?(e.read().exact_winner==p->player?1:-1):node->player==p->player?node->q:-node->q)*e.read().visits;
   e.write().child.reset();
  }}
  if(node->n)evicted_stats[node->context]={node->n,node->q,node->value,node->position};
  node->dormant=false;++state->released;
 }
 void trim_archive(bool need_slot=false,bool refresh=true){
  if(!state->archive)return;
  auto& archive=*state->archive;if(refresh || archive.dirty)archive.refresh();
  auto allowed=[&](const Node* n){return !n->pending && !state->pinned(n);};
  while(archive.occupied.any()){
   size_t slot=Archive::slots;
   // Coloured stones cannot change in forward play. Missing stones can still
   // be supplied by a descendant; only opposite-colour occupation is final.
   if(archive.forward && archive.conflicting.any())for(size_t i=0;i<Archive::slots;++i)
    if(archive.conflicting[i] && allowed(archive.entries[i].node.get())){slot=i;break;}
   if(slot==Archive::slots){
    if(archive.total_bytes()<=archive.limit && !(need_slot && archive.occupied.all()))break;
    slot=archive.victim(allowed);
   }
   if(slot==Archive::slots)break;
   auto node=archive.remove(slot);discard(node);++archive.discarded;if(archive.dirty)archive.refresh();
  }
 }
 void evict() {
  if(!shared || !limit || !requests.empty())return;
  trim_archive();
  // Expanded nodes are a subset of the store, so a store within its limit has
  // nothing to evict. Index entries expire only when discard releases a node.
  if(store.size()<=limit && evicted_stats.size()<=4*limit && outcomes.size()<=16*limit && state->released==state->swept)return;
  auto count=[&]{size_t k=0;for(auto& [key,n]:store)k+=n->expanded;return k;};
  size_t expanded=count();
  const size_t target=expanded>limit?limit-limit/8:expanded;
  while(expanded>target){
   std::vector<Node*> leaves;
   for(auto& [key,n]:store)if(!state->pinned(n.get()) && !n->pending && std::none_of(n->tracked_edges().begin(),n->tracked_edges().end(),[&](int i){auto& child=n->edges[i].read().child;return child && !child->dormant;}))leaves.push_back(n.get());
   if(leaves.empty())break;
   std::sort(leaves.begin(),leaves.end(),[](const Node* x,const Node* y){return x->used<y->used;});
   for(Node* x:leaves){
    if(expanded<=target)break;
    // Keep the incoming links while dormant: exact and numerical improvements
    // must still reach every surviving parent. Only actual discard cuts them.
    auto node=store.at(x->context);expanded-=x->expanded;++evicted;store.erase(x->context);
    if(state->archive && node->expanded && Archive::payload(*node)+sizeof(Archive)<=state->archive->limit){
     trim_archive(true,false);
     if(!state->archive->occupied.all()){state->archive->insert(node);trim_archive(false,false);}
     else discard(node);
    }
    else discard(node);
   }
  }
  // At most four times `limit` summaries: those with the fewest visits leave first.
  if(evicted_stats.size()>4*limit){
   std::vector<std::pair<int,Key>> order;for(auto& [key,stats]:evicted_stats)order.emplace_back(stats.n,key);
   std::nth_element(order.begin(),order.begin()+(order.size()-4*limit),order.end(),[](const auto& x,const auto& y){return x.first<y.first;});
   for(size_t i=0;i<order.size()-4*limit;++i)evicted_stats.erase(order[i].second);
  }
  if(state->released!=state->swept){
   std::erase_if(nodes,[](const auto& entry){return entry.second.expired();});
   for(auto& [key,list]:positions)std::erase_if(list,[](const auto& w){return w.expired();});
   std::erase_if(positions,[](const auto& entry){return entry.second.empty();});prune_continuations();
   state->swept=state->released;
  }
  // Proven outcomes stay while a stored node or a kept summary holds their position; beyond sixteen times `limit` the
  // others go.
  if(outcomes.size()>16*limit){
   std::unordered_set<Key,KeyHash> summarised;for(auto& [key,stats]:evicted_stats)summarised.insert(stats.position);
   std::erase_if(outcomes,[&](const auto& entry){return !positions.contains(entry.first) && !summarised.contains(entry.first);});
  }
 }
 // Shared graph: moves the root to the position after `history`, a node of the store or a new one attached under
 // its stored parents; every node keeps its statistics.
 void root_at(const std::vector<Cell>& history) {owner_access();
  if(!shared || !requests.empty())throw std::runtime_error("Root changes need a shared graph and no pending requests");
  Board next;for(auto c:history){if(!next.legal(c))throw std::runtime_error("Illegal root history");next.make(c);}
  save_root();board=next;priority.clear();defence.clear();hold=false;budget=started=completed=0;lineage.clear();++version;
  if(!state->primary)state->primary=this;
  root=child_here();root->player=board.player;root->used=++clock;
  if(state->primary==this)archive_focus();
  restore_root();pin();adopt(root,history);
  evict();
 }
 // MCGS value of a graph node for its mover from its network value and its edges' visits and current values.
 void refresh(Node& node) {
  double total=node.value+node.carried_sum;int count=1+node.carried;
  for(int i:node.tracked_edges()){auto& e=node.edges[i];if(e.read().visits){total+=e.read().visits*value(node,e);count+=e.read().visits;}}
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
    for(int i:p->tracked_edges()){auto& e=p->edges[i];if(e.read().child.get()==c)changed|=tighten(c->exact_winner,c->distance+1,c->bound,e.write().exact_winner,e.write().distance,e.write().bound);}
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
  if(node.expanded)for(auto& e:node.edges)if(e.read().exact_winner>=0)outcome.edges.push_back({e.action,e.read().exact_winner,e.read().distance,e.read().bound});
  if(!record(node.position,outcome))return;
  // Peers take any improvement: a better outcome, or new edge proofs under an unchanged one.
  if(auto list=positions.find(node.position);list!=positions.end())for(auto& w:std::vector(list->second))if(auto n=w.lock())if(n.get()!=&node)share(node,outcomes[node.position],*n);
 }
 // Merges `outcome` into the table entry of `position`: a tighter verdict and any new or tighter edge proof; true when
 // the entry changed.
 bool record(const Key& position,const Outcome& outcome) {
  auto [o,added]=outcomes.try_emplace(position,outcome);
  auto changed_loss=[&](){if(outcome.winner!=outcome.player && outcome.stones>0 && outcome.stones%2==0)++state->half_losses;};
  if(added){changed_loss();return true;}
  auto& old=o->second;bool changed=false;
  if(old.winner!=outcome.winner){old=outcome;changed_loss();return true;}
  changed|=tighten(outcome.winner,outcome.distance,outcome.bound,old.winner,old.distance,old.bound);
  std::vector<EdgeProof> merged;merged.reserve(old.edges.size()+outcome.edges.size());
  auto i=old.edges.cbegin();auto j=outcome.edges.cbegin();
  while(i!=old.edges.cend() || j!=outcome.edges.cend()){
   if(j==outcome.edges.cend() || (i!=old.edges.cend() && i->action<j->action)){merged.push_back(*i++);continue;}
   if(i==old.edges.cend() || j->action<i->action){merged.push_back(*j++);changed=true;continue;}
   EdgeProof e=*i++;changed|=tighten(j->winner,j->distance,j->bound,e.winner,e.distance,e.bound);++j;merged.push_back(e);
  }
  old.edges=std::move(merged);
  if(changed)changed_loss();
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
   if(j!=o.edges.end() && j->action==e.action)changed|=tighten(j->winner,j->distance,j->bound,e.write().exact_winner,e.write().distance,e.write().bound);
   else if(o.winner!=into.player && e.read().exact_winner<0){e.write().exact_winner=o.winner;e.write().distance=o.distance;e.write().bound=true;changed=true;}
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
   if(f.read().exact_winner>=0 && f.action==e.action)changed|=tighten(f.read().exact_winner,f.read().distance,f.read().bound,e.write().exact_winner,e.write().distance,e.write().bound);
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
  for(auto& e:node.edges)if(e.read().eligible){total+=e.read().visits;maximum=std::max(maximum,e.read().visits);if(e.read().visits){weighted+=node.prior(e)*value(node,e);mass+=node.prior(e);}}
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
  for(size_t i=0;i<q.size();++i)if(node.edges[i].read().eligible){lo=std::min(lo,q[i]);hi=std::max(hi,q[i]);}
  if(lo>hi)lo=hi=0;
  double range=std::max(std::max(1e-8,range_floor),hi-lo);
  for(auto& x:q)x=(x-lo)/range*(50+maximum)*0.1;
  return q;
 }
 // Selection needs individual Qs only for materialized edges. Every cold edge
 // has the same mixed Q; root exports still construct the full legal vector.
 struct SelectionQ {
  const Tree* tree;const Node* node;double mixed,lo,range,scale;
  double raw(const Edge& edge)const {return tree->known(edge)?tree->value(*node,edge):mixed;}
  double operator[](int i)const {return (raw(node->edges[i])-lo)/range*scale;}
  double unknown()const {return (mixed-lo)/range*scale;}
 };
 SelectionQ selection_q(Node& node){
  node.selection_index();double weighted=0,mass=0;int total=0,maximum=0;
  for(int i:node.active){auto& e=node.edges[i];if(e.read().eligible){total+=e.read().visits;maximum=std::max(maximum,e.read().visits);
   if(e.read().visits){weighted+=node.prior(e)*value(node,e);mass+=node.prior(e);}
  }}
  double mixed=(node.value+total*(mass?weighted/mass:node.value))/(total+1),lo=1e300,hi=-1e300;
  if(node.active.size()<node.edges.size())lo=hi=mixed;
  for(int i:node.active){auto& e=node.edges[i];if(e.read().eligible){double q=known(e)?value(node,e):mixed;lo=std::min(lo,q);hi=std::max(hi,q);}}
  if(lo>hi)lo=hi=0;
  return {this,&node,mixed,lo,std::max(std::max(1e-8,range_floor),hi-lo),(50+maximum)*.1};
 }
 int select_interior(Node& node,const SelectionQ& q){
  node.cold_summary();int visits=0;double maxlog=node.cold_best>=0?q.unknown()+node.edges[node.cold_best].logit:-std::numeric_limits<double>::infinity();
  for(int i:node.active){auto& e=node.edges[i];visits+=e.read().visits+e.read().pending;if(e.read().eligible)maxlog=std::max(maxlog,q[i]+e.logit);}
  if(!std::isfinite(maxlog))return -1;
  double total=node.cold_best>=0?node.cold_mass*std::exp(q.unknown()+node.edges[node.cold_best].logit-maxlog):0;
  for(int i:node.active)if(node.edges[i].read().eligible)total+=std::exp(q[i]+node.edges[i].logit-maxlog);
  int chosen=node.cold_best;double best=chosen>=0?std::exp(q.unknown()+node.edges[chosen].logit-maxlog)/total:-std::numeric_limits<double>::infinity();
  for(int i:node.active){auto& e=node.edges[i];if(!e.read().eligible || (e.read().child && e.read().child->pending))continue;
   double score=std::exp(q[i]+e.logit-maxlog)/total-double(e.read().visits+e.read().pending)/(1+visits);
   if(score>best || (score==best && (chosen<0 || i<chosen))){best=score;chosen=i;}
  }
  return chosen;
 }
 // Root candidate sampling logits: each edge's logit, or with root_noise e > 0 log((1 - e) p + e / N) for the N
 // eligible edges, p their softmax over the eligible logits. Only the opening phase's Gumbel-top-k draws on these;
 // halving, the final choice, the improved policy and every non-root node use the edge logits.
 std::vector<double> sampling(const Node& node)const {
  std::vector<double> out;double maximum=-1e300,total=0;int n=0;
  for(auto& e:node.edges){out.push_back(e.logit);if(e.read().eligible){maximum=std::max(maximum,e.logit);++n;}}
  if(root_noise<=0 || !n)return out;
  for(auto& e:node.edges)if(e.read().eligible)total+=std::exp(e.logit-maximum);
  for(size_t i=0;i<out.size();++i)if(node.edges[i].read().eligible)out[i]=std::log((1-root_noise)*std::exp(out[i]-maximum)/total+root_noise/n);
  return out;
 }
 // `last` is the simulation index where the final candidate count begins (the last halving boundary), or the
 // budget when the schedule never halves.
 void schedule(int count) {
   sequence.clear();round={};last=budget;int m=std::min({std::max(samples,int(defence.size())),budget,count});if(!m)return;
  std::vector<int> v(m);int considered=m,previous=-1,halving=0,rounds=std::max(1,int(std::ceil(std::log2(m))));
   while(int(sequence.size())<budget){
    if(considered!=previous){halving=int(sequence.size());previous=considered;if(round_barrier){round.widths.push_back(considered);round.ends.push_back(halving);}}
    int extra=std::max(1,budget/(rounds*considered));
    for(int k=0;k<extra && int(sequence.size())<budget;++k)for(int i=0;i<considered;++i){
     sequence.push_back(v[i]++);if(int(sequence.size())==budget)break;}
    if(round_barrier)round.ends.back()=int(sequence.size());considered=m==1?1:std::max(2,considered/2);
   }
  if(halving)last=halving;
  }
  // A round owns a fixed sampled set. Its reservations can advance while earlier
  // leaves are in flight; the halving decision waits for completed evidence.
  int select_round(Node& node,const SelectionQ& q,const std::vector<int>& blocked){
   const int phase=int(std::upper_bound(round.ends.begin(),round.ends.end(),started)-round.ends.begin()),width=round.widths[phase];
   if(phase==0 && round.opening.empty())round.opening=sampling(node);
   auto score=[&](int i){auto& e=node.edges[i];bool first=phase==0 && std::find(priority.begin(),priority.end(),e.action)!=priority.end();
    return root_edges[i].gumbel+(phase?e.logit:round.opening[i])+(phase?q[i]:root_edges[i].opening_q)+(first?1e6:0)+bonus(e);};
   auto better=[&](int a,int b){bool da=phase==0 && defence.contains(node.edges[a].action),db=phase==0 && defence.contains(node.edges[b].action);
    if(da!=db)return da;double x=score(a),y=score(b);return x!=y?x>y:a<b;};
   if(round.active!=phase){
    if(!requests.empty())return -1;
    std::vector<int> rank;
    if(round.active<0){for(int i=0;i<int(node.edges.size());++i)if(node.edges[i].read().eligible)rank.push_back(i);}
    else for(int i:round.members)if(i>=0 && node.edges[i].read().eligible)rank.push_back(i);
    std::stable_sort(rank.begin(),rank.end(),better);if(rank.size()>size_t(width))rank.resize(width);
    round.members=std::move(rank);round.active=phase;round.counts.assign(width,0);round.limits.assign(width,0);
    const int work=round.ends[phase]-started;
    for(int i=0;i<width;++i)round.limits[i]=work/width+(i<work%width);
   }
   // A refutation replaces a slot's remaining allowance. Completed credits stay
   // on the old action, and cancellation uses the original reservation's slot.
   for(int& i:round.members)if(i>=0 && !node.edges[i].read().eligible)i=-1;
   while(round.members.size()<size_t(width))round.members.push_back(-1);
   for(int& slot:round.members)if(slot<0){
    int best=-1;
    for(int i=0;i<int(node.edges.size());++i)if(node.edges[i].read().eligible && std::find(round.members.begin(),round.members.end(),i)==round.members.end())
     if(best<0 || root_edges[i].epoch>root_edges[best].epoch || (root_edges[i].epoch==root_edges[best].epoch && better(i,best)))best=i;
    slot=best;
   }
   for(size_t i=0;i<round.members.size();++i)if(round.members[i]<0 && round.counts[i]<round.limits[i]){
    int recipient=-1;for(size_t j=0;j<round.members.size();++j)if(round.members[j]>=0 && (recipient<0 || round.limits[j]<round.limits[recipient]))recipient=int(j);
    if(recipient>=0){round.limits[recipient]+=round.limits[i]-round.counts[i];round.limits[i]=round.counts[i];}
   }
   int chosen=-1,least=std::numeric_limits<int>::max();
   for(size_t slot=0;slot<round.members.size();++slot){int i=round.members[slot];if(i<0 || round.counts[slot]>=round.limits[slot] || std::find(blocked.begin(),blocked.end(),i)!=blocked.end())continue;
    auto& e=node.edges[i];if(!e.read().eligible || (e.read().child && e.read().child->pending))continue;
    if(round.counts[slot]<least || (round.counts[slot]==least && (chosen<0 || better(i,chosen)))){least=round.counts[slot];chosen=i;}
   }return chosen;
  }
  int reserve_root(int edge){
   ++root_edges[edge].epoch;++started;++issued;
   if(!round_barrier)return -1;
   auto i=std::find(round.members.begin(),round.members.end(),edge);if(i==round.members.end())throw std::runtime_error("Root work outside its round");
   int slot=int(i-round.members.begin());++round.counts[slot];return slot;
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
   if(win!=none){edge.write().exact_winner=path.player;edge.write().distance=win;edge.write().bound=false;continue;}
   // A lost edge resists longest with the second stone that leaves the slowest completion open.
   int lost=open(edge.action,nullptr);
   if(lost!=none && path.remaining==2)for(auto& t:threats){for(auto second:t){int k=open(edge.action,&second);lost=std::max(lost,k);if(k==none)break;}if(lost==none)break;}
   if(lost!=none){edge.write().exact_winner=1-path.player;edge.write().distance=path.remaining+lost;edge.write().bound=false;}
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
  auto lower=[&](const Edge& e,int turn){return e.read().bound?(tactics?node.remaining+turn:1):e.read().distance;};
  for(auto& edge:node.edges){
   if(edge.read().exact_winner==node.player){winning=true;fastest=std::min(fastest,edge.read().distance);quickest=std::min(quickest,lower(edge,3));}
   else if(edge.read().exact_winner<0)safe=true;
   else {slowest=std::max(slowest,edge.read().distance);longest=std::max(longest,lower(edge,5));}
  }
  for(auto& edge:node.edges)attained|=edge.read().exact_winner==node.player && !edge.read().bound && edge.read().distance==fastest;
  for(auto& edge:node.edges)edge.eligibility(winning?edge.read().exact_winner==node.player && edge.read().distance==fastest
   :!safe?edge.read().distance>=longest:edge.read().exact_winner<0);
  if(winning){node.exact_winner=node.player;node.distance=fastest;node.bound=!attained || quickest<fastest;}
  else if(!safe){node.exact_winner=1-node.player;node.distance=slowest;node.bound=slowest>longest;}
 }
 // Records an externally proven winner of the root edge `action` within `distance` placements (the edge's own
 // included): its value becomes exact (Q = +-1 for the root's mover) and the root is settled, so a lost edge leaves
 // the remaining halving rounds and the final selection. Later proofs may tighten an already settled root.
 // In a shared graph the root's stored parents take the new verdict and value.
 void mark(Cell action,int winner,int distance) {
  if(!root->expanded || (winner!=0 && winner!=1) || distance<1)throw std::runtime_error("Mark needs an expanded root, a winner and a distance");
  auto edge=std::find_if(root->edges.begin(),root->edges.end(),[&](const Edge& e){return e.action==action;});
  if(edge==root->edges.end())throw std::runtime_error("Mark action is not a root edge");
  if(root->exact_winner>=0 && root->exact_winner!=winner)return;
  if(!tighten(winner,distance,true,edge->write().exact_winner,edge->write().distance,edge->write().bound))return;
  edge->write().sum=winner==root->player?edge->read().visits:-edge->read().visits;settle(*root);
  if(graph && root->remaining==2 && winner!=root->player && distance>1){
   const int p=root->player;const auto h=CellHash{}(action);
   const Key half{root->position.a-mix(p*3+2+17)+mix(p*3+1+17)+mix(h^mix(p+1)),
                  root->position.b-mix(p*3+2+71)+mix(p*3+1+71)+mix(h+mix(p+911))};
   const Outcome loss{p,winner,distance-1,root->stones+1,true,{}};
   if(record(half,loss))if(auto list=positions.find(half);list!=positions.end())
    for(auto& w:std::vector(list->second))if(auto n=w.lock())if(apply(loss,*n)){learn(*n);revise(*n);}
  }
  // A shared graph hands the verdict and the changed value on to the root's stored parents.
  if(shared){learn(*root);revise(*root);trim_archive();}
 }
 void begin(int simulations,int sample) {owner_access();
  if(!requests.empty()||simulations<1||sample<1)throw std::runtime_error("Invalid search budget or pending requests");
  budget=simulations;samples=sample;started=completed=0;issued=cancelled=0;root_edges.clear();root_prepared=false;hold=false;priority.clear();defence.clear();
  if(shared){
   evict();current(*root);
   std::vector<Cell> history;for(auto& u:board.history)history.push_back(u.c);
   lineage=prefixes(history);pin();
  }
  schedule(root->expanded?int(std::count_if(root->edges.begin(),root->edges.end(),[](auto& e){return e.read().eligible;})):int(board.legal_moves().size()));
  prepare_root();
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
   if(child->exact_winner>=0 && tighten(child->exact_winner,child->distance+1,child->bound,edge.write().exact_winner,edge.write().distance,edge.write().bound))settle(*node);
   if(graph)node->track(edge);++edge.write().visits;--edge.write().pending;
   if(edge.read().exact_winner>=0){value=edge.read().exact_winner==node->player?1:-1;edge.write().sum=value*edge.read().visits;}
   else edge.write().sum+=value;
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
    if(e.read().exact_winner>=0){++e.write().visits;e.write().sum=(e.read().exact_winner==x->player?1:-1)*e.read().visits;}
    else if(e.read().child && e.read().child->n){++e.write().visits;e.write().sum+=e.read().child->player==x->player?e.read().child->q:-e.read().child->q;}
    else continue;
    x->track(e);++x->n;stale(*x);
   }
  }
  if(!path.edges.empty()){++completed;if(path.edges.front().first==root.get() && path.edges.front().second<int(root_edges.size()))++root_edges[path.edges.front().second].credits;}
 }
 // Next leaf request id; 0 when nothing can be requested now, -1 after a simulation that ended on an exact edge or
 // node, and -3 while a hold is armed and the search stands, with no request pending, at `last`.
 int request() {owner_access();
  proof_root();if(board.winner>=0 || (root->expanded && root->exact_winner>=0))return 0;
  prepare_root();
  if(hold && started==last)return requests.empty()?-3:0;
  if(started>=budget)return 0;
  // Descends by make on the tree's own board; Restore undoes every placement on return.
  Restore restore(board);Node* node=root.get();Path path;path.leaf=node;std::vector<int> blocked;
  for(auto& u:board.history)path.history.push_back(u.c);
  if(shared)++clock;
  while(node->expanded){
   if(graph && !shared)inherit_half_losses(*node);
   if(shared){current(*node);node->used=clock;}
   auto q=selection_q(*node);int chosen=-1;double best=-1e300;
   if(node==root.get()){
    if(round_barrier)chosen=select_round(*node,q,blocked);
    else {
    int considered=sequence[started];
    // Finish each visit layer before its values decide the next halving round.
    if(started && considered!=sequence[started-1] && !requests.empty())return 0;
    auto first=[&](const Edge& e){return considered==0 && std::find(priority.begin(),priority.end(),e.action)!=priority.end();};
    bool forced=false;auto logits=considered?std::vector<double>():sampling(*node);
    for(int i=0;i<int(node->edges.size());++i){auto& e=node->edges[i];if(!e.read().eligible || root_edges[i].epoch!=considered)continue;bool admit=considered==0 && defence.contains(e.action);double score=root_edges[i].gumbel+(considered?e.logit:logits[i])+(considered?q[i]:root_edges[i].opening_q)+(first(e)?1e6:0)+bonus(e);if((admit && !forced) || (admit==forced && score>best)){forced=admit;best=score;chosen=i;}}
    // Marked-lost candidates can leave a round short of candidates; the best of the latest-eliminated ones step in.
    int reached=-1;
    if(chosen<0)for(int i=0;i<int(node->edges.size());++i){auto& e=node->edges[i];if(!e.read().eligible || root_edges[i].epoch>considered)continue;double score=root_edges[i].gumbel+e.logit+q[i]+bonus(e);if(root_edges[i].epoch>reached || (root_edges[i].epoch==reached && score>best)){reached=root_edges[i].epoch;best=score;chosen=i;}}
    // Proofs may leave fewer survivors than this round scheduled. If all survivors already finished its
    // layer, continue the least-visited survivor rather than waiting for an eliminated action forever.
    if(chosen<0){int least=std::numeric_limits<int>::max();for(int i=0;i<int(node->edges.size());++i){auto& e=node->edges[i];if(!e.read().eligible)continue;double score=root_edges[i].gumbel+e.logit+q[i]+bonus(e);if(root_edges[i].epoch<least || (root_edges[i].epoch==least && score>best)){least=root_edges[i].epoch;best=score;chosen=i;}}}
   }
   } else chosen=select_interior(*node,q);
   if(chosen<0){
    if(!round_barrier || path.edges.empty())return 0;
    blocked.push_back(path.edges.front().second);for(size_t i=0;i<path.edges.size();++i)board.undo();
    path.history.resize(path.history.size()-path.edges.size());path.edges.clear();node=root.get();path.leaf=node;continue;
   }
   auto& edge=node->edges[chosen];if(edge.read().child && edge.read().child->pending)return 0;
   path.edges.emplace_back(node,chosen);board.make(edge.action);path.history.push_back(edge.action);
   if(!edge.read().child){if(shared)attach(*node,edge,child_here(node->history,true));else {edge.write().child=child_here();if(graph){node->track(edge);edge.read().child->parents.push_back(node->weak_from_this());}}}
   if(shared)reactivate(edge.read().child);
   node=edge.read().child.get();path.leaf=node;if(shared)node->used=clock;
   if(board.winner>=0 || edge.read().exact_winner>=0 || node->exact_winner>=0){
    if(board.winner>=0){node->exact_winner=board.winner;node->distance=0;node->bound=false;}
    else if(edge.read().exact_winner>=0 && node->exact_winner<0){node->exact_winner=edge.read().exact_winner;node->distance=edge.read().distance-1;node->bound=edge.read().bound;}
    int winner=node->exact_winner;for(auto [parent,index]:path.edges)++parent->edges[index].write().pending;path.round_slot=reserve_root(path.edges.front().second);backup(path,winner==node->player?1:-1);return -1;}
   // A transposed child already holds more visits than this edge: take its value without evaluating (MCGS).
   if(graph && !shared && node->expanded && node->n>edge.read().visits){for(auto [parent,index]:path.edges)++parent->edges[index].write().pending;path.round_slot=reserve_root(path.edges.front().second);backup(path,node->q,false);return -1;}
  }
  if(node->pending)return 0;
  node->pending=true;node->player=board.player;capture(path);
  for(auto [parent,index]:path.edges)++parent->edges[index].write().pending;
  if(!path.edges.empty()){path.round_slot=reserve_root(path.edges.front().second);}
  const bool immediate=!path.own.empty();
  int id=next_id++;requests.emplace(id,std::move(path));
  if(immediate){
   // capture already found a legal win within this turn. Materialize the complete legal policy and
   // install its exact evidence through the ordinary fulfillment/backup path, without requesting NN work.
   const auto& legal=requests.at(id).legal;std::vector<int64_t> actions;actions.reserve(2*legal.size());
   for(auto c:legal){actions.push_back(c.q);actions.push_back(c.r);}
   std::vector<double> zeros(legal.size());fulfill(id,actions.data(),zeros.data(),zeros.data(),int(legal.size()));
   return -1;
  }
  return id;
 }
 // Exact evidence from another view dominates a late prediction. Finish this view's reservation once.
 bool proof_closed(Path& path){
  // A cold root without a materialized witness still needs its first legal policy. Descendant work and
  // already expanded roots can retire immediately; proof_root() handles retained shared root witnesses.
  if(path.edges.empty() && !path.leaf->expanded)return false;
  bool exact=path.leaf->exact_winner>=0;
  for(auto [node,index]:path.edges)exact|=node->exact_winner>=0 || node->edges[index].read().exact_winner>=0;
  if(!exact)return false;
  path.leaf->pending=false;for(auto [node,index]:path.edges)--node->edges[index].write().pending;
  if(!path.edges.empty()){++completed;++root_edges[path.edges.front().second].credits;}
  ++proof_retired;return true;
 }
 // An exact root needs its legal policy, not another neural evaluation.
 void proof_root(){
  if(!shared || root->expanded || root->exact_winner<0 || board.winner>=0)return;
  auto o=outcomes.find(root->position);
  if(o==outcomes.end() || (o->second.winner==root->player && !o->second.witnessed()))return;
  auto legal=board.legal_moves();root->expanded=true;root->remaining=board.remaining;root_prepared=false;
  root->value=root->q=root->exact_winner==root->player?1:-1;
  root->edges.reserve(legal.size());for(auto c:legal){Edge e(&root->empty);e.action=c;root->edges.push_back(std::move(e));}
  apply(o->second,*root);settle(*root);learn(*root);revise(*root);
 }
 void fulfill(int id,const int64_t* actions,const double* logits,const double* values,int count,int exact=-1,Cell witness={},int distance=-1){
  auto found=requests.find(id);if(found==requests.end())throw std::runtime_error("Unknown request");auto& path=found->second;
  if(exact<0 && proof_closed(path)){requests.erase(found);return;}
  const auto& legal=path.legal;
  if(count!=int(legal.size())||count<1)throw std::runtime_error("Incomplete legal actions");
  double maximum=-1e300;for(int i=0;i<count;++i){if(legal[i]!=Cell{actions[2*i],actions[2*i+1]}||!std::isfinite(logits[i])||!std::isfinite(values[i])||std::abs(values[i])>1)throw std::runtime_error("Invalid evaluation");maximum=std::max(maximum,logits[i]);}
  // Another proof producer may have expanded this leaf already. A late certificate can tighten it but
  // must not append a duplicate action list or reset this view's root sampling record.
  if(exact>=0 && path.leaf->expanded){
   auto& node=*path.leaf;
   if(node.exact_winner>=0 && node.exact_winner!=exact)throw std::runtime_error("Conflicting leaf proof");
   for(auto& edge:node.edges)if(edge.action==witness)tighten(exact,distance,true,edge.write().exact_winner,edge.write().distance,edge.write().bound);
   settle(node);learn(node);revise(node);proof_closed(path);if(state->evidence)state->evidence(state->evidence_owner,*this,path);requests.erase(found);return;
  }
  auto& node=*path.leaf;const bool at_root=&node==root.get();
  // Build each legal edge once. Normalize the weighted value after accumulating
  // its mass, without a temporary weights array or a division for every action.
  if(at_root){root_edges.assign(count,{});root_prepared=true;}
  double total=0,weighted=0;node.edges.reserve(count);
  for(int i=0;i<count;++i){Edge edge(&node.empty);edge.action=legal[i];edge.logit=logits[i]-maximum;edge.weight=std::exp(edge.logit);
   total+=edge.weight;weighted+=edge.weight*values[i];
   if(at_root){double u=std::generate_canonical<double,53>(rng);root_edges[i].gumbel=-std::log(-std::log(std::clamp(u,1e-15,1-1e-15)));}
   node.edges.push_back(std::move(edge));
  }
  node.policy_mass=total;node.value=weighted/total;
  node.expanded=true;node.remaining=path.remaining;
  if(graph && path.remaining==1 && !path.history.empty())node.first=path.history.back();
  // A retained proven loss covers every legal continuation even if this node had not needed expansion yet.
  if(node.exact_winner>=0 && node.exact_winner!=node.player)for(auto& edge:node.edges){edge.write().exact_winner=node.exact_winner;edge.write().distance=node.distance;edge.write().bound=true;}
  if(tactics)classify(path,node);
  if(graph){
   // A live expanded peer of the position hands over its edge proofs; the shared outcome keeps them when no peer
   // is alive (a win's witnesses, a loss's per-move resistances).
   if(auto list=positions.find(node.position);list!=positions.end())
    for(auto& w:std::vector(list->second))if(auto peer=w.lock())if(peer.get()!=&node && peer->expanded && copy(*peer,node))settle(node);
   if(auto o=outcomes.find(node.position);o!=outcomes.end())apply(o->second,node);
   inherit_half_losses(node);
   if(shared)link(path,node);
  }
  // A certificate adds its witness as a winning edge; settle keeps any shorter tactical win found by classify.
  if(exact>=0){for(auto& edge:node.edges)if(edge.action==witness && (edge.read().exact_winner!=exact || edge.read().distance>distance)){edge.write().exact_winner=exact;edge.write().distance=distance;edge.write().bound=true;}settle(node);}
  node.pending=false;if(at_root)schedule(int(std::count_if(node.edges.begin(),node.edges.end(),[](auto& e){return e.read().eligible;})));backup(path,node.exact_winner<0?node.value:node.exact_winner==node.player?1:-1);if(at_root)freeze_root_q();if(state->evidence)state->evidence(state->evidence_owner,*this,path);requests.erase(found);
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
  if(node->exact_winner>=0 && node->exact_winner!=player)throw std::runtime_error("Conflicting leaf proof");
  auto legal=position.legal_moves();std::vector<int64_t> actions;for(auto c:legal){actions.push_back(c.q);actions.push_back(c.r);}
  std::vector<double> zeros(legal.size());fulfill(id,actions.data(),zeros.data(),zeros.data(),int(legal.size()),player,witness,distance);
  if(move_count==2){
   position.make(witness);
   if(graph){
    // An existing node of this turn context takes the second stone as its witness instead of being replaced.
    auto [p,c]=keys(position);
    if(auto existing=nodes[c].lock()){
     existing->archive_changed();existing->parents.push_back(node->weak_from_this());
     const Outcome o{player,player,distance-1,int(position.cells.size()),true,{EdgeProof{Cell{moves[2],moves[3]},player,distance-1,true}}};
     // The outcome keeps the witness even while the node is unexpanded; its parents and peers take the proof.
     if(apply(o,*existing))revise(*existing);
     record(p,o);
     for(auto& w:std::vector(positions[p]))if(auto peer=w.lock())if(peer!=existing)share(*existing,outcomes[p],*peer);
     for(auto& e:node->edges)if(e.action==witness){e.write().child=existing;if(graph)node->track(e);break;}
     return;
    }
   }
   auto next_legal=position.legal_moves();
   auto child=std::make_shared<Node>(state->memory);child->player=player;child->remaining=1;child->expanded=true;child->exact_winner=player;child->distance=distance-1;child->bound=true;
   if(state->archive)child->history=std::make_shared<HistoryLink>(node->history,witness);
   child->edges.reserve(next_legal.size());for(auto c:next_legal){Edge e(&child->empty);e.action=c;e.eligibility(c==Cell{moves[2],moves[3]});if(e.read().eligible){e.write().exact_winner=player;e.write().distance=distance-1;e.write().bound=true;}child->edges.push_back(std::move(e));}
   if(graph){auto [p,c]=keys(position);child->position=p;child->context=c;child->stones=int(position.cells.size());child->n=1;child->q=1;child->parents.push_back(node->weak_from_this());nodes[c]=child;positions[p].push_back(child);if(shared){index_child(child,position);store[c]=child;}learn(*child);}
   for(auto& e:node->edges)if(e.action==witness){e.write().child=std::move(child);if(graph)node->track(e);break;}
  }
 }
 void cancel(){for(auto& [id,path]:requests){path.leaf->pending=false;for(auto [node,index]:path.edges)--node->edges[index].write().pending;if(!path.edges.empty()){--root_edges[path.edges.front().second].epoch;--started;++cancelled;if(round_barrier && path.round_slot>=0)--round.counts[path.round_slot];}}requests.clear();}
 void advance(Cell action){owner_access();if(!requests.empty()||!board.legal(action))throw std::runtime_error("Invalid advance");std::shared_ptr<Node> next;int winner=-1,distance=-1;bool bound=false;
  for(auto& e:root->edges)if(e.action==action){winner=e.read().exact_winner;distance=e.read().distance;bound=e.read().bound;next=shared?e.read().child:std::move(e.write().child);break;}
  save_root();auto before=root->history;board.make(action);root=next?std::move(next):child_here(std::move(before),true);reactivate(root);root->player=board.player;restore_root();
  // A shared graph keeps the siblings and every earlier position; a new root joins its stored parents.
  if(shared){std::vector<Cell> history;for(auto& u:board.history)history.push_back(u.c);if(state->primary==this)archive_focus();pin();adopt(root,history);root->used=++clock;++version;}
  if(winner>=0 && winner!=root->player){root->exact_winner=winner;root->distance=distance-1;root->bound=bound;}
  if(graph){
   std::erase_if(nodes,[](const auto& entry){return entry.second.expired();});
   for(auto& [key,list]:positions)std::erase_if(list,[](const auto& w){return w.expired();});
   std::erase_if(positions,[](const auto& entry){return entry.second.empty();});prune_continuations();
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
   if(tactics){Path path;capture(path);if(!path.own.empty()){root_prepared=false;root->expanded=true;root->remaining=path.remaining;root->edges.reserve(path.legal.size());for(auto c:path.legal){Edge e(&root->empty);e.action=c;root->edges.push_back(std::move(e));}classify(path,*root);}}
  }
 }
};
thread_local std::string error;
// Encode a pending, non-terminal leaf directly from the history and complete legal list captured by search.
// This is the deterministic hexcrop layout; no rules-board replay or dense legal-mask regeneration is needed.
int encode(const Path& path,uint8_t* planes,int capacity,int64_t* cells,int64_t* info,bool rectangular=false){
 auto transform=[](Cell c,int k){if(k>=6)std::swap(c.q,c.r);for(int i=0;i<k%6;++i)c={-c.r,c.q+c.r};return c;};
 std::array<int64_t,3> low{},high{};bool empty=true;
 auto include=[&](Cell c){std::array<int64_t,3> v{c.q,c.r,c.q+c.r};if(empty){low=high=v;empty=false;}else for(int i=0;i<3;++i){low[i]=std::min(low[i],v[i]);high[i]=std::max(high[i],v[i]);}};
 auto bounds=[&](bool far){empty=true;for(auto c:path.history)include(c);if(!far)for(auto c:path.legal)include(c);};
 auto frame=[&](int k,int halo){auto q=transform({1,0},k),r=transform({0,1},k);std::array<int64_t,4> f{};for(int i=0;i<2;++i){int a=i?q.r:q.q,b=i?r.r:r.q,axis=a&&b?2:b?1:0;bool positive=(a?a:b)>0;f[i]=(positive?low[axis]:-high[axis])-halo;f[i+2]=high[axis]-low[axis]+1+2*halo;}return f;};
 bounds(false);int symmetry=0,halo=0;int64_t side=std::numeric_limits<int64_t>::max();
 auto choose=[&]{side=std::numeric_limits<int64_t>::max();for(int k=0;k<12;++k){auto f=frame(k,halo);auto need=std::max(f[2],f[3]);if(need<side){side=need;symmetry=k;}}};
 choose();if(side>256){halo=4;bounds(true);choose();if(side>256)return -2;}
 int size=0;for(int b:{24,32,40,48,64,96,128,192,256})if(b>=side){size=b;break;}
 auto f=frame(symmetry,halo);
 int width=rectangular?int(std::max<int64_t>(24,(f[2]+7)/8*8)):size;
 int height=rectangular?int(std::max<int64_t>(24,(f[3]+7)/8*8)):size;
 int64_t ox=(width-f[2])/2,oy=(height-f[3])/2;
 auto xy=[&](Cell c){auto p=transform(c,symmetry);return Cell{p.q+ox-f[0],p.r+oy-f[1]};};
 const int area=height*width,n=int(path.history.size());int far=0;
 if(planes){if(capacity<8*area)throw std::runtime_error("Leaf plane buffer too small");std::fill(planes,planes+8*area,0);}
 for(size_t i=0;i<path.legal.size();++i){auto p=xy(path.legal[i]);bool inside=!halo || (p.q>=ox && p.r>=oy && p.q<ox+f[2] && p.r<oy+f[3]);int64_t index=inside?p.r*width+p.q:-1;if(cells)cells[i]=index;if(planes && inside)planes[2*area+index]=1;far+=!inside;}
 if(planes){
  for(int i=0;i<n;++i){auto p=xy(path.history[i]);int owner=((i+1)/2)%2;planes[(owner==path.player?0:area)+p.r*width+p.q]=1;}
  for(int64_t y=oy;y<oy+f[3];++y)std::fill(planes+3*area+y*width+ox,planes+3*area+y*width+ox+f[2],1);
  std::fill(planes+(path.remaining==1?4:5)*area,planes+(path.remaining==1?5:6)*area,1);
  int start=path.remaining==2 || !n?n:n-1;
  if(start<n){auto p=xy(path.history.back());planes[6*area+p.r*width+p.q]=1;}
  for(int i=std::max(0,start-2);i<start;++i){auto p=xy(path.history[i]);planes[7*area+p.r*width+p.q]=1;}
 }
 if(info){std::array<int64_t,9> metadata{int64_t(path.legal.size()),path.player,path.remaining,symmetry,f[0],f[1],ox,oy,far};std::copy(metadata.begin(),metadata.end(),info);}
 if(rectangular && info)info[9]=width;
 return height;
}
}
extern "C" {
HX_API const char* hxg_error(){return gumbel::error.c_str();}
// Returns crop side, -2 for an unencodable span, or 0 on invalid request/buffer. info has nine int64 entries;
// cells has hxg_legal entries, planes has capacity bytes. Null outputs query the required layout.
HX_API int hxg_encode(void* p,int id,uint8_t* planes,int capacity,int64_t* cells,int64_t* info){try{auto& requests=static_cast<gumbel::Tree*>(p)->requests;auto found=requests.find(id);if(found==requests.end())throw std::runtime_error("Unknown encode request");return gumbel::encode(found->second,planes,capacity,cells,info);}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
// Rectangular layout: the same nine metadata entries, followed by width. Return value is height.
HX_API int hxg_encode_rect(void* p,int id,uint8_t* planes,int capacity,int64_t* cells,int64_t* info){try{auto& requests=static_cast<gumbel::Tree*>(p)->requests;auto found=requests.find(id);if(found==requests.end())throw std::runtime_error("Unknown encode request");return gumbel::encode(found->second,planes,capacity,cells,info,true);}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API void* hxg_new(uint64_t seed){try{return new gumbel::Tree(seed);}catch(...){return nullptr;}}
HX_API void hxg_free(void* p){delete static_cast<gumbel::Tree*>(p);}
HX_API int hxg_tactics(void* p,int enabled){auto& t=*static_cast<gumbel::Tree*>(p);if(t.root->expanded || !t.requests.empty())return 0;t.tactics=enabled!=0;return 1;}
// Switches graph search on (enabled != 0) or off before the root is expanded: transposed turn contexts share one node
// and proven outcomes are shared by position.
HX_API int hxg_graph(void* p,int enabled){auto& t=*static_cast<gumbel::Tree*>(p);if(t.root->expanded || !t.requests.empty() || t.state.use_count()>1)return 0;
 t.graph=enabled!=0;t.nodes.clear();t.outcomes.clear();t.positions.clear();t.state->continuations.clear();
 if(t.graph){auto [position,context]=gumbel::keys(t.board);t.root->position=position;t.root->context=context;t.root->stones=int(t.board.cells.size());t.root->remaining=t.board.remaining;if(!t.board.history.empty())t.root->first=t.board.history.back().c;t.nodes[context]=t.root;t.positions[position].push_back(t.root);t.index_child(t.root,t.board);}
 return 1;}
// Makes the tree a shared game graph (graph search whose store keeps every node until evicted) before the root is
// expanded; `limit` bounds the expanded nodes kept between searches (0: no bound). 0 with the error set otherwise.
HX_API int hxg_share(void* p,int64_t limit){auto& t=*static_cast<gumbel::Tree*>(p);if(t.root->expanded || !t.requests.empty() || limit<0){gumbel::error="A shared graph needs an unexpanded root and a nonnegative limit";return 0;}
 if(!hxg_graph(p,1))return 0;
 t.shared=true;t.limit=size_t(limit);t.root->context=gumbel::keys(t.board).second;t.store[t.root->context]=t.root;t.state->primary=&t;t.pin();return 1;}
// Optional byte-bounded dormant evidence. Configure once before the first
// expansion so every archived node has its own immutable history descriptor.
HX_API int hxg_archive(void* p,int64_t bytes){try{
 auto& t=*static_cast<gumbel::Tree*>(p);t.owner_access();
 if(!t.shared || t.state->archive || !t.requests.empty() || std::any_of(t.store.begin(),t.store.end(),[](const auto& entry){return entry.second->expanded || entry.second->stones;}) || bytes<65536)
  throw std::runtime_error("Archive needs an unexpanded shared graph and at least 64 KiB");
 t.state->archive=std::make_unique<gumbel::Archive>(size_t(bytes));
 t.archive_focus();return 1;
 }catch(const std::exception& e){gumbel::error=e.what();return 0;}}
// Forward-only games can release colour-conflicting dormant nodes without byte
// pressure. Primary focus owns this policy; pending/pinned views delay disposal.
// The default archive keeps incompatible evidence for undo and analysis.
HX_API int hxg_archive_forward(void* p,int enabled){try{
 auto& t=*static_cast<gumbel::Tree*>(p);t.owner_access();
 if(!t.state->archive || t.state->primary!=&t || (enabled!=0 && enabled!=1))
  throw std::runtime_error("Forward retention needs the primary archived graph and a boolean mode");
 auto& a=*t.state->archive;a.forward=enabled;a.set_focus(a.focus);t.trim_archive();return 1;
 }catch(const std::exception& e){gumbel::error=e.what();return 0;}}
// Managed dormant payload and estimated index allocations, not pool residency
// or process RSS. History prefixes are conservatively charged per entry.
HX_API int hxg_archive_stats(void* p,int64_t* out){auto& t=*static_cast<gumbel::Tree*>(p);auto* a=t.state->archive.get();if(!a)return 0;
 t.trim_archive();std::array<int64_t,10> values{int64_t(a->occupied.count()),int64_t(a->total_bytes()),int64_t(a->limit),int64_t(a->retained),int64_t(a->reused),int64_t(a->discarded),int64_t(a->compatible.count()),int64_t(a->index_bytes()),int64_t(a->membership.size()),int64_t(a->focus_stones())};
 std::copy(values.begin(),values.end(),out);return 1;}
// Shared graph: moves the root to the position after `history` (n int64 q/r pairs), keeping every node's statistics
// (Tree::root_at); 0 with the error set for an illegal history, an unshared tree or pending requests.
HX_API int hxg_root_at(void* p,const int64_t* history,int n){try{std::vector<Cell> h;for(int i=0;i<n;++i)h.push_back({history[2*i],history[2*i+1]});static_cast<gumbel::Tree*>(p)->root_at(h);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
// Shared graph: the number of root changes so far (hxg_root_at, hxg_advance), for callers that track the root.
HX_API int64_t hxg_root_version(void* p){return static_cast<gumbel::Tree*>(p)->version;}
// Shared graph store: out = {stored nodes, expanded stored nodes, nodes evicted so far, limit, evicted summaries kept,
// proven outcomes kept}; 0 when unshared.
HX_API int hxg_store(void* p,int64_t* out){auto& t=*static_cast<gumbel::Tree*>(p);if(!t.shared)return 0;int64_t expanded=0;for(auto& [key,n]:t.store)expanded+=n->expanded;
 out[0]=int64_t(t.store.size());out[1]=expanded;out[2]=t.evicted;out[3]=int64_t(t.limit);out[4]=int64_t(t.evicted_stats.size());out[5]=int64_t(t.outcomes.size());return 1;}
// Quiescent shared-store storage census: legal rows, materialized states,
// row capacity bytes, state bytes, nodes, index/cache capacity bytes,
// row size and state size. Node bodies, pool overhead and neural/proof caches are separate.
HX_API int hxg_storage(void* p,int64_t* out){auto& t=*static_cast<gumbel::Tree*>(p);if(!t.shared)return 0;
 std::array<int64_t,8> counts{};
 for(auto& [key,n]:t.store){counts[0]+=int64_t(n->edges.size());counts[2]+=int64_t(n->edges.capacity()*sizeof(gumbel::Edge));
  counts[5]+=int64_t((n->tracked.capacity()+n->active.capacity())*sizeof(int)+n->cold.capacity()*sizeof(gumbel::ColdBlock));for(auto& edge:n->edges)counts[1]+=!edge.read().empty;}
 counts[3]=counts[1]*sizeof(gumbel::EdgeState);counts[4]=int64_t(t.store.size());counts[6]=sizeof(gumbel::Edge);counts[7]=sizeof(gumbel::EdgeState);
 std::copy(counts.begin(),counts.end(),out);return 1;
}
// Completed Q in value units for the root's mover, per root edge in hxg_stats order (Tree::completed_q); the edge
// count, 0 before the root is expanded.
HX_API int hxg_q(void* p,double* out){auto& t=*static_cast<gumbel::Tree*>(p);auto& n=*t.root;if(!n.expanded)return 0;if(t.shared)t.current(n);int maximum=0;auto& q=t.completed_q(n,maximum);if(out)std::copy(q.begin(),q.end(),out);t.trim_archive(false,false);return int(q.size());}
// Sets the least Q range of the completed-Q rescale (0, the default, keeps 1e-8) for every later search and target;
// 0 with no change when `floor` is negative or not finite.
HX_API int hxg_q_range_floor(void* p,double floor){if(!std::isfinite(floor) || floor<0){gumbel::error="Invalid Q range floor";return 0;}static_cast<gumbel::Tree*>(p)->range_floor=floor;return 1;}
// Sets the uniform share of the root's candidate sampling (Tree::sampling; 0, the default, samples by the prior)
// for every later search; 0 with no change unless 0 <= `noise` < 1.
HX_API int hxg_round_barrier(void* p,int enabled){try{auto& t=*static_cast<gumbel::Tree*>(p);t.owner_access();
 if((enabled!=0 && enabled!=1) || t.started || !t.requests.empty())throw std::runtime_error("Configure rounds before search work");
 if(t.round_barrier!=bool(enabled))t.root_sessions.clear();t.round_barrier=enabled;if(t.budget)t.schedule(t.root->expanded?int(std::count_if(t.root->edges.begin(),t.root->edges.end(),[](const auto& e){return e.read().eligible;})):int(t.board.legal_moves().size()));return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
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
  for(auto& e:n.edges)if(e.read().child){t.board.make(e.action);walk(*e.read().child);t.board.undo();}
 };
 walk(*t.root);out[0]=nodes;out[1]=expanded;out[2]=exact;out[3]=duplicates;return 1;}
HX_API int hxg_exact(void* p){auto& t=*static_cast<gumbel::Tree*>(p);return t.board.winner>=0?t.board.winner:t.root->exact_winner;}
// Placements within which hxg_exact's winner completes six from the root (0 on a finished board), -1 when not exact.
HX_API int hxg_distance(void* p){auto& t=*static_cast<gumbel::Tree*>(p);return t.board.winner>=0?0:t.root->exact_winner>=0?t.root->distance:-1;}
// A read-only snapshot of reachable exact positions. Records are int64 words:
// history length, winner, placement upper bound, then q/r pairs. An exact edge
// need not have an allocated child. The caller supplies a bounded buffer; only
// complete records are written. Neural values never enter this snapshot.
HX_API int hxg_facts(void* p,int64_t* out,int capacity){
 auto& t=*static_cast<gumbel::Tree*>(p);int used=0;
 std::unordered_set<const gumbel::Node*> visited;
 std::unordered_set<gumbel::Key,gumbel::KeyHash> emitted;
 std::vector<Cell> history;for(auto& u:t.board.history)history.push_back(u.c);
 auto emit=[&](int winner,int distance){
  int size=3+int(2*history.size());
  if(winner<0 || distance<=0 || used+size>capacity || !emitted.insert(gumbel::keys(history).first).second)return;
  out[used++]=int64_t(history.size());out[used++]=winner;out[used++]=distance;
  for(auto c:history){out[used++]=c.q;out[used++]=c.r;}
 };
 std::function<void(const gumbel::Node&)> walk=[&](const gumbel::Node& n){
  if(used+3+int(2*history.size())>capacity || !visited.insert(&n).second)return;
  emit(n.exact_winner,n.distance);
  for(auto& e:n.edges)if(e.read().exact_winner>=0 || e.read().child){
   history.push_back(e.action);emit(e.read().exact_winner,e.read().distance-1);
   if(e.read().child)walk(*e.read().child);
   history.pop_back();
  }
 };
 walk(*t.root);return used;
}
// A verified defender-root certificate settles a loss without visiting every
// legal edge. Parents and rule-equivalent nodes receive it through normal graph
// propagation. No pending search is mutated by this synchronous entry point.
HX_API int hxg_prove_loss(void* p,int winner,int distance){try{
 auto& t=*static_cast<gumbel::Tree*>(p);auto& n=*t.root;
 if(!t.requests.empty() || winner!=1-n.player || distance<1 || (n.exact_winner>=0 && n.exact_winner!=winner))throw std::runtime_error("Invalid root loss proof");
 gumbel::Outcome outcome{n.player,winner,distance,n.stones,true,{}};
 t.apply(outcome,n);t.refresh(n);t.learn(n);t.propagate(n,nullptr);t.trim_archive();return 1;
 }catch(const std::exception& e){gumbel::error=e.what();return 0;}}
// The node estimate, for display when the listed candidates are all refuted.
HX_API double hxg_value(void* p){auto& t=*static_cast<gumbel::Tree*>(p);if(t.shared)t.renew(*t.root);t.trim_archive(false,false);return t.root->q;}
HX_API int hxg_begin(void* p,int simulations,int sample){try{auto& t=*static_cast<gumbel::Tree*>(p);t.begin(simulations,sample);t.trim_archive(false,false);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxg_next(void* p){try{auto& t=*static_cast<gumbel::Tree*>(p);int result=t.request();t.trim_archive(false,false);return result;}catch(const std::exception& e){gumbel::error=e.what();return -2;}}
HX_API int hxg_history(void* p,int id,int64_t* out){auto& h=static_cast<gumbel::Tree*>(p)->requests.at(id).history;if(out)for(int i=0;i<int(h.size());++i){out[2*i]=h[i].q;out[2*i+1]=h[i].r;}return int(h.size());}
// Legal moves of a pending request in sorted (q, r) order, the actions hxg_fulfill must be given.
HX_API int hxg_legal(void* p,int id,int64_t* out){auto& l=static_cast<gumbel::Tree*>(p)->requests.at(id).legal;if(out)for(int i=0;i<int(l.size());++i){out[2*i]=l[i].q;out[2*i+1]=l[i].r;}return int(l.size());}
// Network predictions arrive here. The expanded leaf keeps its raw value for
// later root records; proofs and immediate wins expand without one.
HX_API int hxg_fulfill(void* p,int id,const int64_t* a,const double* logits,const double* q,int n){try{auto& t=*static_cast<gumbel::Tree*>(p);
 auto found=t.requests.find(id);gumbel::Node* leaf=found==t.requests.end()?nullptr:found->second.leaf;bool expanded=leaf && leaf->expanded;
 t.fulfill(id,a,logits,q,n);if(leaf && !expanded && leaf->expanded && n>0){leaf->raw=q[0];leaf->raw_known=true;}
 t.trim_archive(false,false);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
// Caller must independently verify the strategy certificate before this entry.
// Exact history and placement phase prevent applying it to a different request.
HX_API int hxg_prove(void* p,int id,const int64_t* h,int n,int player,int remaining,const int64_t* moves,int count,int turns){try{auto& t=*static_cast<gumbel::Tree*>(p);t.prove(id,h,n,player,remaining,moves,count,turns);t.trim_archive(false,false);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API void hxg_cancel(void* p){static_cast<gumbel::Tree*>(p)->cancel();}
HX_API int hxg_advance(void* p,int64_t q,int64_t r){try{auto& t=*static_cast<gumbel::Tree*>(p);t.advance({q,r});t.trim_archive(false,false);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
// Final selection among eligible edges at the highest epoch this search reached; when proofs removed every visited edge
// the unvisited survivors compete. No simulation started (an expired budget) scores nothing.
HX_API int hxg_stats(void* p,int64_t* actions,int* visits,double* values,double* scores){
 auto& t=*static_cast<gumbel::Tree*>(p);t.proof_root();auto& n=*t.root;if(!n.expanded)return 0;
 t.prepare_root();if(t.shared)t.current(n);auto q=t.transformed(n);int max_epoch=0,searched=0;
 bool round_survivor=t.round_barrier && std::any_of(t.round.members.begin(),t.round.members.end(),[&](int i){return i>=0 && n.edges[i].read().eligible;});
 for(size_t i=0;i<n.edges.size();++i){auto& e=n.edges[i];auto& local=t.root_edges[i];searched=std::max(searched,local.epoch);if(e.read().eligible)max_epoch=std::max(max_epoch,local.epoch);}
 for(size_t i=0;i<n.edges.size();++i){auto& e=n.edges[i];auto& local=t.root_edges[i];if(actions){
  actions[2*i]=e.action.q;actions[2*i+1]=e.action.r;visits[i]=e.read().visits;values[i]=t.value(n,e);
  bool finalist=round_survivor?std::find(t.round.members.begin(),t.round.members.end(),int(i))!=t.round.members.end():searched && local.epoch==max_epoch;
 scores[i]=e.read().eligible && (n.exact_winner>=0 || finalist)?local.gumbel+e.logit+q[i]+t.bonus(e):-std::numeric_limits<double>::infinity();
 }}t.trim_archive(false,false);return int(n.edges.size());
}
HX_API int hxg_policy(void* p,double* out){auto& t=*static_cast<gumbel::Tree*>(p);auto& n=*t.root;if(!n.expanded)return 0;if(t.shared)t.current(n);auto q=t.transformed(n);double maximum=-1e300,total=0;for(int i=0;i<int(q.size());++i){q[i]=n.edges[i].read().eligible?q[i]+n.edges[i].logit+t.bonus(n.edges[i]):-std::numeric_limits<double>::infinity();maximum=std::max(maximum,q[i]);}for(auto& v:q){v=std::exp(v-maximum);total+=v;}if(out)for(int i=0;i<int(q.size());++i)out[i]=q[i]/total;t.trim_archive(false,false);return int(q.size());}
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
 t.schedule(t.root->expanded?int(std::count_if(t.root->edges.begin(),t.root->edges.end(),[](auto& e){return e.read().eligible;})):int(t.board.legal_moves().size()));return 1;
 }catch(const std::exception& e){gumbel::error=e.what();return 0;}}
// Caller must hold a verified proof that `winner` wins after root edge (q, r) within `distance` placements, the
// edge's own included; see Tree::mark.
HX_API int hxg_mark_exact(void* p,int64_t q,int64_t r,int winner,int distance){try{static_cast<gumbel::Tree*>(p)->mark({q,r},winner,distance);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
}


// Same-game views share position evidence, never sampling state. All graph APIs have one calling owner thread.
extern "C" HX_API void* hxg_view(void* source,const int64_t* history,int count,uint64_t seed){
 try{
  auto& original=*static_cast<gumbel::Tree*>(source);
  if(!original.shared || count<0)throw std::runtime_error("View needs a shared graph and valid history");
  auto view=std::make_unique<gumbel::Tree>(seed,original.state);view->shared=view->graph=true;
  view->tactics=original.tactics;view->range_floor=original.range_floor;view->root_noise=original.root_noise;view->round_barrier=original.round_barrier;
  std::vector<Cell> h;for(int i=0;i<count;++i)h.push_back({history[2*i],history[2*i+1]});view->root_at(h);
  return view.release();
 }catch(const std::exception& e){gumbel::error=e.what();return nullptr;}
}
// Current-root simulations issued, completed, cancelled; pending requests; retired results since construction;
// live views into this game's store. Root counters and credits resume with the view's saved comparison.
extern "C" HX_API void hxg_view_counters(void* p,uint64_t* out){
 auto& t=*static_cast<gumbel::Tree*>(p);
 out[0]=t.issued;out[1]=t.completed;out[2]=t.cancelled;out[3]=t.requests.size();out[4]=t.proof_retired;out[5]=t.state->pins.size();
}
// Direct completed sampling credits of this root search, in hxg_stats action order. Inherited visits are separate.
extern "C" HX_API int hxg_root_credits(void* p,uint64_t* out){
 auto& t=*static_cast<gumbel::Tree*>(p);t.prepare_root();
 if(out)for(size_t i=0;i<t.root_edges.size();++i)out[i]=t.root_edges[i].credits;
 return int(t.root_edges.size());
}
