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
namespace gumbel {
struct Node;
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
struct Edge { Cell action;double logit=0,prior=0,sum=0,gumbel=0;int visits=0,pending=0,epoch=0,exact_winner=-1,distance=-1;bool eligible=true,bound=false;std::shared_ptr<Node> child; };
// An exact winner comes with a distance: the placements within which that winner completes six from this position
// (an edge counts its own placement) against any defence, combined by min at the winner's choices and max at the
// loser's. It is exact for terminal and tactical results; `bound` marks an upper bound, which certificates give.
// With graph search a node also keeps its visits `n`, its utility `q` for its mover (the MCGS value) and its parents.
struct Node : std::enable_shared_from_this<Node> { int player=0,remaining=1,exact_winner=-1,distance=-1,n=0,stones=0;bool expanded=false,pending=false,bound=false;double value=0,q=0;Key position;std::vector<Edge> edges;std::vector<std::weak_ptr<Node>> parents; };
// A pending leaf: its history, its legal moves in sorted order and, with tactics, the side to move's completions
// (own) and the opponent's (threats), both restricted to fully legal ones.
struct Path { Node* leaf=nullptr;std::vector<std::pair<Node*,int>> edges;std::vector<Cell> history,legal;int player=0,remaining=1;std::vector<std::vector<Cell>> own,threats; };
struct Tree {
 Board board;std::shared_ptr<Node> root=std::make_shared<Node>();std::map<int,Path> requests;
 // Graph search (opt-in): nodes shared by turn-context key, proven outcomes {winner, distance, stones, bound} shared by
 // position key. Tree search gives every edge its own child and keeps both tables empty.
 bool graph=false;std::unordered_map<Key,std::weak_ptr<Node>,KeyHash> nodes;std::unordered_map<Key,std::vector<std::weak_ptr<Node>>,KeyHash> positions;std::unordered_map<Key,std::array<int,4>,KeyHash> outcomes;
 std::mt19937_64 rng;int budget=0,started=0,completed=0,next_id=1,samples=0,last=0;bool tactics=false,hold=false;std::vector<int> sequence;
 // Root actions sampled first in the opening phase of the current search; ordering only (set_priority).
 std::vector<Cell> priority;
 std::map<Cell,double> defence;
 double bonus(const Edge& e)const {auto i=defence.find(e.action);return i==defence.end()?0:i->second;}
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
  auto n=std::make_shared<Node>();n->player=board.player;n->position=position;n->stones=int(board.cells.size());slot=n;positions[position].push_back(n);
  if(auto o=outcomes.find(position);o!=outcomes.end()){n->exact_winner=o->second[0];n->distance=o->second[1];n->bound=o->second[3];}
  return n;
 }
 // MCGS value of a graph node for its mover from its network value and its edges' visits and current values.
 void refresh(Node& node) {
  double total=node.value;int count=1;
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
    for(auto& e:p->edges)if(e.child.get()==c && (e.exact_winner!=c->exact_winner || e.distance!=c->distance+1 || e.bound!=c->bound)){
     e.exact_winner=c->exact_winner;e.distance=c->distance+1;e.bound=c->bound;changed=true;
    }
    if(changed){settle(*p);learn(*p);}
   }
   refresh(*p);
   if(p->q!=before || p->exact_winner!=winner || p->distance!=distance || p->bound!=bound)for(auto& w:p->parents)if(auto g=w.lock())work.emplace_back(g.get(),p);
  }
 }
 // Records a proven node's outcome for its position (graph search), keeping the shortest bound.
 // A new or better outcome also reaches the live nodes of the same position in other turn contexts.
 void learn(const Node& node) {
  if(!graph || node.exact_winner<0)return;
  std::array<int,4> outcome{node.exact_winner,node.distance,node.stones,node.bound};
  auto [o,added]=outcomes.try_emplace(node.position,outcome);
  const bool better=o->second[0]==node.exact_winner && (node.distance<o->second[1] || (node.distance==o->second[1] && !node.bound && o->second[3]));
  if(!added && !better)return;
  o->second=outcome;
  if(auto list=positions.find(node.position);list!=positions.end())for(auto& w:std::vector(list->second))if(auto n=w.lock())if(n.get()!=&node)share(node,*n);
 }
 // Gives `into`, a node of the same position in another turn context, the verdicts `from` holds: its proven edges
 // (the legal moves are the same) or, when either is unexpanded, the node's own outcome; then updates its parents.
 void share(const Node& from,Node& into) {
  bool changed=false;
  if(from.expanded && into.expanded && from.edges.size()==into.edges.size()){
   for(size_t i=0;i<from.edges.size();++i){
    auto& f=from.edges[i];auto& e=into.edges[i];
    if(f.exact_winner>=0 && f.action==e.action && (e.exact_winner!=f.exact_winner || e.distance>f.distance || (e.distance==f.distance && e.bound && !f.bound))){
     e.exact_winner=f.exact_winner;e.distance=f.distance;e.bound=f.bound;changed=true;
    }
   }
   if(changed)settle(into);
  } else if(!into.expanded && (into.exact_winner<0 || into.distance>from.distance)){
   into.exact_winner=from.exact_winner;into.distance=from.distance;into.bound=from.bound;changed=true;
  }
  if(changed){refresh(into);propagate(into,nullptr);}
 }
 // Completed Q (mctx mixed value, min-max rescale, (50 + max visits) * 0.1) over the eligible edges only: proven
 // losses and the non-winning edges of a won node set neither the mixed value, the visit scale nor the range.
 // Entries of ineligible edges are returned on the same scale but every caller discards them.
 std::vector<double> transformed(Node& node) {
  double weighted=0,mass=0;int total=0,maximum=0;
  for(auto& e:node.edges)if(e.eligible){total+=e.visits;maximum=std::max(maximum,e.visits);if(e.visits){weighted+=e.prior*value(node,e);mass+=e.prior;}}
  double mixed=(node.value+total*(mass?weighted/mass:node.value))/(total+1),lo=1e300,hi=-1e300;
  std::vector<double> q;
  for(auto& e:node.edges){q.push_back(known(e)?value(node,e):mixed);if(e.eligible){lo=std::min(lo,q.back());hi=std::max(hi,q.back());}}
  if(lo>hi)lo=hi=0;
  double range=std::max(1e-8,hi-lo);
  for(auto& x:q)x=(x-lo)/range*(50+maximum)*0.1;
  return q;
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
 void mark(Cell action,int winner,int distance) {
  if(!root->expanded || (winner!=0 && winner!=1) || distance<1)throw std::runtime_error("Mark needs an expanded root, a winner and a distance");
  auto edge=std::find_if(root->edges.begin(),root->edges.end(),[&](const Edge& e){return e.action==action;});
  if(edge==root->edges.end())throw std::runtime_error("Mark action is not a root edge");
  if(root->exact_winner>=0)return;
  edge->exact_winner=winner;edge->distance=distance;edge->bound=true;edge->sum=winner==root->player?edge->visits:-edge->visits;settle(*root);
 }
 void begin(int simulations,int sample) {
  if(!requests.empty()||simulations<1||sample<1)throw std::runtime_error("Invalid search budget or pending requests");
  budget=simulations;samples=sample;started=completed=0;hold=false;priority.clear();defence.clear();
  schedule(root->expanded?int(std::count_if(root->edges.begin(),root->edges.end(),[](auto& e){return e.eligible;})):int(board.legal_moves().size()));
  for(auto& e:root->edges){e.epoch=0;double u=std::generate_canonical<double,53>(rng);e.gumbel=-std::log(-std::log(std::clamp(u,1e-15,1-1e-15)));}
 }
 // Tree search: each edge keeps the running mean of the values backed up through it. Graph search (MCGS): each node
 // recomputes q = (network value + sum over edges of visits * edge value) / (1 + sum of visits) from its children's
 // current values, so a child shared by several parents is weighted by this node's own edge visits. `fresh` counts
 // the leaf as visited; a playout reusing a transposed child's value leaves the child unchanged.
 void backup(Path& path,double value,bool fresh=true) {
  Node* child=path.leaf;
  if(graph && fresh){++child->n;child->q=child->exact_winner>=0?(child->exact_winner==child->player?1:-1):child->n==1?value:child->q;learn(*child);}
  for(auto i=path.edges.rbegin();i!=path.edges.rend();++i){
   auto& [node,index]=*i;auto& edge=node->edges[index];
   if(child->player!=node->player)value=-value;
   if(child->exact_winner>=0 && (edge.exact_winner!=child->exact_winner || edge.distance!=child->distance+1 || edge.bound!=child->bound)){edge.exact_winner=child->exact_winner;edge.distance=child->distance+1;edge.bound=child->bound;settle(*node);}
   ++edge.visits;--edge.pending;
   if(edge.exact_winner>=0){value=edge.exact_winner==node->player?1:-1;edge.sum=value*edge.visits;}
   else edge.sum+=value;
   if(graph){++node->n;refresh(*node);learn(*node);}
   child=node;
  }
  // Nodes on the path are current; their other parents are refreshed upwards.
  if(graph && !path.edges.empty()){
   propagate(*path.leaf,path.edges.back().first);
   for(size_t j=1;j<path.edges.size();++j)propagate(*path.edges[j].first,path.edges[j-1].first);
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
  while(node->expanded){
   auto q=transformed(*node);int chosen=-1;double best=-1e300;
   if(node==root.get()){
    int considered=sequence[started];
    // Finish each visit layer before its values decide the next halving round.
    if(started && considered!=sequence[started-1] && !requests.empty())return 0;
    auto first=[&](const Edge& e){return considered==0 && std::find(priority.begin(),priority.end(),e.action)!=priority.end();};
    bool forced=false;
    for(int i=0;i<int(node->edges.size());++i){auto& e=node->edges[i];if(!e.eligible || e.epoch!=considered)continue;bool admit=considered==0 && defence.contains(e.action);double score=e.gumbel+e.logit+(considered?q[i]:0)+(first(e)?1e6:0)+bonus(e);if((admit && !forced) || (admit==forced && score>best)){forced=admit;best=score;chosen=i;}}
    // Marked-lost candidates can leave a round short of candidates; the best of the latest-eliminated ones step in.
    int reached=-1;
    if(chosen<0)for(int i=0;i<int(node->edges.size());++i){auto& e=node->edges[i];if(!e.eligible || e.epoch>considered)continue;double score=e.gumbel+e.logit+q[i]+bonus(e);if(e.epoch>reached || (e.epoch==reached && score>best)){reached=e.epoch;best=score;chosen=i;}}
    // Proofs may leave fewer survivors than this round scheduled. If all survivors already finished its
    // layer, continue the least-visited survivor rather than waiting for an eliminated action forever.
    if(chosen<0){int least=std::numeric_limits<int>::max();for(int i=0;i<int(node->edges.size());++i){auto& e=node->edges[i];if(!e.eligible)continue;double score=e.gumbel+e.logit+q[i]+bonus(e);if(e.epoch<least || (e.epoch==least && score>best)){least=e.epoch;best=score;chosen=i;}}}
   } else {
    double maxlog=-1e300,total=0;int visits=0;for(int i=0;i<int(q.size());++i){q[i]=node->edges[i].eligible?q[i]+node->edges[i].logit:-std::numeric_limits<double>::infinity();maxlog=std::max(maxlog,q[i]);visits+=node->edges[i].visits+node->edges[i].pending;}
    for(auto& x:q){x=std::exp(x-maxlog);total+=x;}
    for(int i=0;i<int(q.size());++i){auto& e=node->edges[i];if(!e.eligible || (e.child && e.child->pending))continue;double score=q[i]/total-double(e.visits+e.pending)/(1+visits);if(score>best){best=score;chosen=i;}}
   }
   if(chosen<0)return 0;
   auto& edge=node->edges[chosen];if(edge.child && edge.child->pending)return 0;
   path.edges.emplace_back(node,chosen);board.make(edge.action);path.history.push_back(edge.action);
   if(!edge.child){edge.child=child_here();if(graph)edge.child->parents.push_back(node->weak_from_this());}
   node=edge.child.get();path.leaf=node;
   if(board.winner>=0 || edge.exact_winner>=0 || node->exact_winner>=0){
    if(board.winner>=0){node->exact_winner=board.winner;node->distance=0;node->bound=false;}
    else if(edge.exact_winner>=0 && node->exact_winner<0){node->exact_winner=edge.exact_winner;node->distance=edge.distance-1;node->bound=edge.bound;}
    int winner=node->exact_winner;for(auto [parent,index]:path.edges)++parent->edges[index].pending;++root->edges[path.edges.front().second].epoch;++started;backup(path,winner==node->player?1:-1);return -1;}
   // A transposed child already holds more visits than this edge: take its value without evaluating (MCGS).
   if(graph && node->expanded && node->n>edge.visits){for(auto [parent,index]:path.edges)++parent->edges[index].pending;++root->edges[path.edges.front().second].epoch;++started;backup(path,node->q,false);return -1;}
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
  node.value=0;node.edges.reserve(count);for(int i=0;i<count;++i){Edge edge;edge.action=legal[i];edge.logit=logits[i]-maximum;edge.prior=weights[i]/total;node.value+=edge.prior*values[i];double u=std::generate_canonical<double,53>(rng);if(at_root)edge.gumbel=-std::log(-std::log(std::clamp(u,1e-15,1-1e-15)));node.edges.push_back(std::move(edge));}
  node.expanded=true;node.remaining=path.remaining;
  // A retained proven loss covers every legal continuation even if this node had not needed expansion yet.
  if(node.exact_winner>=0 && node.exact_winner!=node.player)for(auto& edge:node.edges){edge.exact_winner=node.exact_winner;edge.distance=node.distance;edge.bound=true;}
  if(tactics)classify(path,node);
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
   position.make(witness);auto next_legal=position.legal_moves();
   auto child=std::make_shared<Node>();child->player=player;child->remaining=1;child->expanded=true;child->exact_winner=player;child->distance=distance-1;child->bound=true;
   for(auto c:next_legal){Edge e;e.action=c;e.prior=1./next_legal.size();e.eligible=c==Cell{moves[2],moves[3]};if(e.eligible){e.exact_winner=player;e.distance=distance-1;e.bound=true;}child->edges.push_back(std::move(e));}
   if(graph){auto [p,c]=keys(position);child->position=p;child->stones=int(position.cells.size());child->n=1;child->q=1;child->parents.push_back(node->weak_from_this());nodes[c]=child;positions[p].push_back(child);learn(*child);}
   for(auto& e:node->edges)if(e.action==witness){e.child=std::move(child);break;}
  }
 }
 void cancel(){for(auto& [id,path]:requests){path.leaf->pending=false;for(auto [node,index]:path.edges)--node->edges[index].pending;if(!path.edges.empty()){--root->edges[path.edges.front().second].epoch;--started;}}requests.clear();}
 void advance(Cell action){if(!requests.empty()||!board.legal(action))throw std::runtime_error("Invalid advance");std::shared_ptr<Node> next;int winner=-1,distance=-1;bool bound=false;priority.clear();defence.clear();hold=false;
  for(auto& e:root->edges)if(e.action==action){winner=e.exact_winner;distance=e.distance;bound=e.bound;next=std::move(e.child);break;}
  board.make(action);root=next?std::move(next):child_here();root->player=board.player;budget=started=completed=0;
  if(winner>=0 && winner!=root->player){root->exact_winner=winner;root->distance=distance-1;root->bound=bound;}
  if(graph){
   std::erase_if(nodes,[](const auto& entry){return entry.second.expired();});
   for(auto& [key,list]:positions)std::erase_if(list,[](const auto& w){return w.expired();});
   std::erase_if(positions,[](const auto& entry){return entry.second.empty();});
   // Stones are never removed, so positions with fewer stones than the board can not recur.
   std::erase_if(outcomes,[&](const auto& entry){return entry.second[2]<int(board.cells.size());});
  }
  // A root won without a known witness (a winning edge classified without a child, or a shared outcome) needs one.
  // Reconstruct immediate tactical choices on the actual board; general certificates retain their two-placement child.
  if((winner==root->player || root->exact_winner==root->player) && !root->expanded && board.winner<0){
   root->exact_winner=-1;
   if(tactics){Path path;capture(path);if(!path.own.empty()){root->expanded=true;root->remaining=path.remaining;for(auto c:path.legal){Edge e;e.action=c;e.prior=1./path.legal.size();root->edges.push_back(std::move(e));}classify(path,*root);}}
  }
 }
};
thread_local std::string error;
}
extern "C" {
HX_API const char* hxg_error(){return gumbel::error.c_str();}
HX_API void* hxg_new(uint64_t seed){try{return new gumbel::Tree(seed);}catch(...){return nullptr;}}
HX_API void hxg_free(void* p){delete static_cast<gumbel::Tree*>(p);}
HX_API int hxg_tactics(void* p,int enabled){auto& t=*static_cast<gumbel::Tree*>(p);if(t.root->expanded || !t.requests.empty())return 0;t.tactics=enabled!=0;return 1;}
// Switches graph search on (enabled != 0) or off before the root is expanded: transposed turn contexts share one node
// and proven outcomes are shared by position.
HX_API int hxg_graph(void* p,int enabled){auto& t=*static_cast<gumbel::Tree*>(p);if(t.root->expanded || !t.requests.empty())return 0;
 t.graph=enabled!=0;t.nodes.clear();t.outcomes.clear();t.positions.clear();
 if(t.graph){auto [position,context]=gumbel::keys(t.board);t.root->position=position;t.root->stones=int(t.board.cells.size());t.nodes[context]=t.root;t.positions[position].push_back(t.root);}
 return 1;}
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
HX_API int hxg_stats(void* p,int64_t* actions,int* visits,double* values,double* scores){auto& t=*static_cast<gumbel::Tree*>(p);auto& n=*t.root;if(!n.expanded)return 0;auto q=t.transformed(n);int max_epoch=0,searched=0;for(auto& e:n.edges){searched=std::max(searched,e.epoch);if(e.eligible)max_epoch=std::max(max_epoch,e.epoch);}for(int i=0;i<int(n.edges.size());++i){auto& e=n.edges[i];if(actions){actions[2*i]=e.action.q;actions[2*i+1]=e.action.r;visits[i]=e.visits;values[i]=t.value(n,e);scores[i]=e.eligible && (n.exact_winner>=0 || (searched && e.epoch==max_epoch))?e.gumbel+e.logit+q[i]+t.bonus(e):-std::numeric_limits<double>::infinity();}}return int(n.edges.size());}
HX_API int hxg_policy(void* p,double* out){auto& t=*static_cast<gumbel::Tree*>(p);auto& n=*t.root;if(!n.expanded)return 0;auto q=t.transformed(n);double maximum=-1e300,total=0;for(int i=0;i<int(q.size());++i){q[i]=n.edges[i].eligible?q[i]+n.edges[i].logit+t.bonus(n.edges[i]):-std::numeric_limits<double>::infinity();maximum=std::max(maximum,q[i]);}for(auto& v:q){v=std::exp(v-maximum);total+=v;}if(out)for(int i=0;i<int(q.size());++i)out[i]=q[i]/total;return int(q.size());}
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
