// Native placement-tree scheduling. Algorithm reference: DeepMind mctx.
#include "hexo.cpp"
#include <memory>
#include <random>
#include <map>
#include <numeric>
#include <string>
namespace gumbel {
struct Node;
struct Edge { Cell action;double logit=0,prior=0,sum=0,gumbel=0;int visits=0,pending=0,epoch=0,exact_winner=-1,distance=-1;bool eligible=true;std::unique_ptr<Node> child; };
// An exact winner comes with a distance: the placements within which that winner completes six from this position
// (an edge counts its own placement) against any defence. Exact for terminal and tactical results, an upper bound
// from certificates, combined by min at the winner's choices and max at the loser's.
struct Node { int player=0,exact_winner=-1,distance=-1;bool expanded=false,pending=false;double value=0;std::vector<Edge> edges; };
// A pending leaf: its history, its legal moves in sorted order and, with tactics, the side to move's completions
// (own) and the opponent's (threats), both restricted to fully legal ones.
struct Path { Node* leaf=nullptr;std::vector<std::pair<Node*,int>> edges;std::vector<Cell> history,legal;int player=0,remaining=1;std::vector<std::vector<Cell>> own,threats; };
struct Tree {
 Board board;std::unique_ptr<Node> root=std::make_unique<Node>();std::map<int,Path> requests;
 std::mt19937_64 rng;int budget=0,started=0,completed=0,next_id=1,samples=0,last=0;bool tactics=false,hold=false;std::vector<int> sequence;
 // Root actions sampled first in the opening phase of the current search; ordering only (set_priority).
 std::vector<Cell> priority;
 std::map<Cell,double> defence;
 double bonus(const Edge& e)const {auto i=defence.find(e.action);return i==defence.end()?0:i->second;}
 explicit Tree(uint64_t seed):rng(seed){}
 bool done()const {return requests.empty() && (board.winner>=0 || (root->expanded && root->exact_winner>=0) || completed>=budget);}
 double value(const Node& node,const Edge& e)const {return e.exact_winner>=0?(e.exact_winner==node.player?1:-1):e.visits?e.sum/e.visits:node.value;}
 // Completed Q (mctx mixed value, min-max rescale, (50 + max visits) * 0.1) over the eligible edges only: proven
 // losses and the non-winning edges of a won node set neither the mixed value, the visit scale nor the range.
 // Entries of ineligible edges are returned on the same scale but every caller discards them.
 std::vector<double> transformed(Node& node) {
  double weighted=0,mass=0;int total=0,maximum=0;
  for(auto& e:node.edges)if(e.eligible){total+=e.visits;maximum=std::max(maximum,e.visits);if(e.visits){weighted+=e.prior*value(node,e);mass+=e.prior;}}
  double mixed=(node.value+total*(mass?weighted/mass:node.value))/(total+1),lo=1e300,hi=-1e300;
  std::vector<double> q;
  for(auto& e:node.edges){q.push_back(e.exact_winner>=0 || e.visits?value(node,e):mixed);if(e.eligible){lo=std::min(lo,q.back());hi=std::max(hi,q.back());}}
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
   if(win!=none){edge.exact_winner=path.player;edge.distance=win;continue;}
   // A lost edge resists longest with the second stone that leaves the slowest completion open.
   int lost=open(edge.action,nullptr);
   if(lost!=none && path.remaining==2)for(auto& t:threats){for(auto second:t){int k=open(edge.action,&second);lost=std::max(lost,k);if(k==none)break;}if(lost==none)break;}
   if(lost!=none){edge.exact_winner=1-path.player;edge.distance=path.remaining+lost;}
  }
  settle(node);
 }
 // Node verdict, distance and eligibility from its edges' exact winners: a winning edge makes the node won at its
 // shortest winning distance; with no edge left undecided the node is lost at its longest distance; otherwise the
 // lost edges are ineligible.
 void settle(Node& node) {
  // Only expansion installs the complete legal action list. A pending/unexpanded leaf is never a universal proof.
  if(!node.expanded || node.edges.empty())return;
  bool winning=false,safe=false;int fastest=std::numeric_limits<int>::max(),slowest=0;
  for(auto& edge:node.edges){if(edge.exact_winner==node.player){winning=true;fastest=std::min(fastest,edge.distance);}else if(edge.exact_winner<0)safe=true;else slowest=std::max(slowest,edge.distance);}
  if(winning){node.exact_winner=node.player;node.distance=fastest;}
  else if(!safe){node.exact_winner=1-node.player;node.distance=slowest;}
  // A won node offers only its shortest wins, a lost node its longest resistance, otherwise every unproven edge.
  for(auto& edge:node.edges)edge.eligible=winning?edge.exact_winner==node.player && edge.distance==fastest:!safe?edge.distance==slowest:edge.exact_winner<0;
 }
 // Records an externally proven winner of the root edge `action` within `distance` placements (the edge's own
 // included): its value becomes exact (Q = +-1 for the root's mover) and the root is settled, so a lost edge leaves
 // the remaining halving rounds and the final selection. A root that is already exact is left unchanged.
 void mark(Cell action,int winner,int distance) {
  if(!root->expanded || (winner!=0 && winner!=1) || distance<1)throw std::runtime_error("Mark needs an expanded root, a winner and a distance");
  auto edge=std::find_if(root->edges.begin(),root->edges.end(),[&](const Edge& e){return e.action==action;});
  if(edge==root->edges.end())throw std::runtime_error("Mark action is not a root edge");
  if(root->exact_winner>=0)return;
  edge->exact_winner=winner;edge->distance=distance;edge->sum=winner==root->player?edge->visits:-edge->visits;settle(*root);
 }
 void begin(int simulations,int sample) {
  if(!requests.empty()||simulations<1||sample<1)throw std::runtime_error("Invalid search budget or pending requests");
  budget=simulations;samples=sample;started=completed=0;hold=false;priority.clear();defence.clear();
  schedule(root->expanded?int(std::count_if(root->edges.begin(),root->edges.end(),[](auto& e){return e.eligible;})):int(board.legal_moves().size()));
  for(auto& e:root->edges){e.epoch=0;double u=std::generate_canonical<double,53>(rng);e.gumbel=-std::log(-std::log(std::clamp(u,1e-15,1-1e-15)));}
 }
 void backup(Path& path,double value) {
  Node* child=path.leaf;
  for(auto i=path.edges.rbegin();i!=path.edges.rend();++i){
   auto& [node,index]=*i;auto& edge=node->edges[index];
   if(child->player!=node->player)value=-value;
   if(child->exact_winner>=0 && (edge.exact_winner!=child->exact_winner || edge.distance!=child->distance+1)){edge.exact_winner=child->exact_winner;edge.distance=child->distance+1;settle(*node);}
   ++edge.visits;--edge.pending;
   if(edge.exact_winner>=0){value=edge.exact_winner==node->player?1:-1;edge.sum=value*edge.visits;}
   else edge.sum+=value;
   child=node;
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
   if(!edge.child){edge.child=std::make_unique<Node>();edge.child->player=board.player;}
   node=edge.child.get();path.leaf=node;
   if(board.winner>=0 || edge.exact_winner>=0 || node->exact_winner>=0){
    if(board.winner>=0){node->exact_winner=board.winner;node->distance=0;}
    else if(edge.exact_winner>=0 && node->exact_winner<0){node->exact_winner=edge.exact_winner;node->distance=edge.distance-1;}
    int winner=node->exact_winner;for(auto [parent,index]:path.edges)++parent->edges[index].pending;++root->edges[path.edges.front().second].epoch;++started;backup(path,winner==node->player?1:-1);return -1;}
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
  node.expanded=true;
  // A retained proven loss covers every legal continuation even if this node had not needed expansion yet.
  if(node.exact_winner>=0 && node.exact_winner!=node.player)for(auto& edge:node.edges){edge.exact_winner=node.exact_winner;edge.distance=node.distance;}
  if(tactics)classify(path,node);
  if(exact>=0){node.exact_winner=exact;node.distance=distance;for(auto& edge:node.edges){edge.eligible=edge.action==witness;if(edge.eligible){edge.exact_winner=exact;edge.distance=distance;}}}
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
   auto child=std::make_unique<Node>();child->player=player;child->expanded=true;child->exact_winner=player;child->distance=distance-1;
   for(auto c:next_legal){Edge e;e.action=c;e.prior=1./next_legal.size();e.eligible=c==Cell{moves[2],moves[3]};if(e.eligible){e.exact_winner=player;e.distance=distance-1;}child->edges.push_back(std::move(e));}
   for(auto& e:node->edges)if(e.action==witness){e.child=std::move(child);break;}
  }
 }
 void cancel(){for(auto& [id,path]:requests){path.leaf->pending=false;for(auto [node,index]:path.edges)--node->edges[index].pending;if(!path.edges.empty()){--root->edges[path.edges.front().second].epoch;--started;}}requests.clear();}
 void advance(Cell action){if(!requests.empty()||!board.legal(action))throw std::runtime_error("Invalid advance");std::unique_ptr<Node> next;int winner=-1,distance=-1;priority.clear();defence.clear();hold=false;
  for(auto& e:root->edges)if(e.action==action){winner=e.exact_winner;distance=e.distance;next=std::move(e.child);break;}
  board.make(action);root=next?std::move(next):std::make_unique<Node>();root->player=board.player;budget=started=completed=0;
  if(winner>=0 && winner!=root->player){root->exact_winner=winner;root->distance=distance-1;}
  // A winning edge classified without a child needs a second-placement witness. Reconstruct immediate
  // tactical choices on the actual board; general certificates already retain their two-placement child.
  if(winner==root->player && !root->expanded && board.winner<0){
   root->exact_winner=-1;
   if(tactics){Path path;capture(path);if(!path.own.empty()){root->expanded=true;for(auto c:path.legal){Edge e;e.action=c;e.prior=1./path.legal.size();root->edges.push_back(std::move(e));}classify(path,*root);}}
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
