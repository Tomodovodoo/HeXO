// Native placement-tree scheduling. Algorithm reference: DeepMind mctx.
#include "hexo.cpp"
#include <memory>
#include <random>
#include <map>
#include <numeric>
#include <string>
namespace gumbel {
struct Node;
struct Edge { Cell action;double logit=0,prior=0,sum=0,gumbel=0;int visits=0,pending=0,epoch=0,exact_winner=-1;bool eligible=true;std::unique_ptr<Node> child; };
struct Node { int player=0,exact_winner=-1;bool expanded=false,pending=false;double value=0;std::vector<Edge> edges; };
// A pending leaf: its history, its legal moves in sorted order and, with tactics, the side to move's completions
// (own) and the opponent's (threats), both restricted to fully legal ones.
struct Path { Node* leaf=nullptr;std::vector<std::pair<Node*,int>> edges;std::vector<Cell> history,legal;int player=0,remaining=1;std::vector<std::vector<Cell>> own,threats; };
struct Tree {
 Board board;std::unique_ptr<Node> root=std::make_unique<Node>();std::map<int,Path> requests;
 std::mt19937_64 rng;int budget=0,started=0,completed=0,next_id=1,samples=0,last=0;bool tactics=false,hold=false;std::vector<int> sequence;
 // Root actions sampled first in the opening phase of the current search; ordering only (set_priority).
 std::vector<Cell> priority;
 explicit Tree(uint64_t seed):rng(seed){}
 std::vector<double> transformed(Node& node) {
  double weighted=0,mass=0;int total=0,maximum=0;
  for(auto& e:node.edges) {total+=e.visits;maximum=std::max(maximum,e.visits);if(e.visits){weighted+=e.prior*e.sum/e.visits;mass+=e.prior;}}
  double mixed=(node.value+total*(mass?weighted/mass:node.value))/(total+1);
  std::vector<double> q;for(auto& e:node.edges)q.push_back(e.visits?e.sum/e.visits:mixed);
  auto [lo,hi]=std::minmax_element(q.begin(),q.end());double a=*lo,range=std::max(1e-8,*hi-a);
  for(auto& x:q)x=(x-a)/range*(50+maximum)*0.1;
  return q;
 }
 // `last` is the simulation index where the final candidate count begins (the last halving boundary), or the
 // budget when the schedule never halves.
 void schedule(int count) {
  sequence.clear();last=budget;int m=std::min({samples,budget,count});if(!m)return;
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
  for(auto& edge:node.edges){
   // Every completion cell is within five of an existing stone, hence legal.
   bool win=false;
   for(auto& completion:own)if(completion.size()<size_t(path.remaining) || contains(completion,edge.action))win=true;
   if(win){edge.exact_winner=path.player;continue;}
   auto first=std::find_if(threats.begin(),threats.end(),[&](const auto& completion){return !contains(completion,edge.action);});
   bool cover=first==threats.end();
   if(!cover && path.remaining==2)for(auto second:*first){
    bool all=true;for(auto& completion:threats)if(!contains(completion,edge.action) && !contains(completion,second))all=false;
    if(all){cover=true;break;}
   }
   if(!cover)edge.exact_winner=1-path.player;
  }
  settle(node);
 }
 // Node verdict and eligibility from its edges' exact winners: a winning edge makes the node won and leaves only
 // winning edges eligible; with no edge left undecided the node is lost and every edge stays eligible; otherwise
 // the lost edges are ineligible.
 void settle(Node& node) {
  bool winning=false,safe=false;
  for(auto& edge:node.edges){if(edge.exact_winner==node.player)winning=true;else if(edge.exact_winner<0)safe=true;}
  if(winning)node.exact_winner=node.player;
  else if(!safe)node.exact_winner=1-node.player;
  for(auto& edge:node.edges)edge.eligible=winning?edge.exact_winner==node.player:(!safe || edge.exact_winner<0);
 }
 // Records an externally proven winner of the root edge `action`: its value becomes exact (Q = +-1 for the
 // root's mover) and the root is settled, so a lost edge leaves the remaining halving rounds and the final
 // selection. A root that is already exact is left unchanged.
 void mark(Cell action,int winner) {
  if(!root->expanded || (winner!=0 && winner!=1))throw std::runtime_error("Mark needs an expanded root and a winner");
  auto edge=std::find_if(root->edges.begin(),root->edges.end(),[&](const Edge& e){return e.action==action;});
  if(edge==root->edges.end())throw std::runtime_error("Mark action is not a root edge");
  if(root->exact_winner>=0)return;
  edge->exact_winner=winner;edge->sum=winner==root->player?edge->visits:-edge->visits;settle(*root);
 }
 void begin(int simulations,int sample) {
  if(!requests.empty()||simulations<1||sample<1)throw std::runtime_error("Invalid search budget or pending requests");
  budget=simulations;samples=sample;started=completed=0;hold=false;priority.clear();
  schedule(root->expanded?int(std::count_if(root->edges.begin(),root->edges.end(),[](auto& e){return e.eligible;})):int(board.legal_moves().size()));
  for(auto& e:root->edges){e.epoch=0;double u=std::generate_canonical<double,53>(rng);e.gumbel=-std::log(-std::log(std::clamp(u,1e-15,1-1e-15)));}
 }
 void backup(Path& path,double value) {
  int player=path.leaf->player;
  for(auto i=path.edges.rbegin();i!=path.edges.rend();++i){auto& [node,index]=*i;auto& edge=node->edges[index];if(player!=node->player)value=-value;player=node->player;edge.sum+=value;++edge.visits;--edge.pending;}
  if(!path.edges.empty())++completed;
 }
 // Next leaf request id; 0 when nothing can be requested now, -1 after a simulation that ended on an exact edge or
 // node, and -3 while a hold is armed and the search stands, with no request pending, at `last`.
 int request() {
  if(board.winner>=0)return 0;
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
    for(int i=0;i<int(node->edges.size());++i){auto& e=node->edges[i];if(!e.eligible || e.epoch!=considered)continue;double score=e.gumbel+e.logit+(considered?q[i]:0)+(first(e)?1e6:0);if(score>best){best=score;chosen=i;}}
    // Marked-lost candidates can leave a round short of candidates; the best of the latest-eliminated ones step in.
    int reached=-1;
    if(chosen<0)for(int i=0;i<int(node->edges.size());++i){auto& e=node->edges[i];if(!e.eligible || e.epoch>considered)continue;double score=e.gumbel+e.logit+q[i];if(e.epoch>reached || (e.epoch==reached && score>best)){reached=e.epoch;best=score;chosen=i;}}
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
   if(board.winner>=0 || edge.exact_winner>=0 || node->exact_winner>=0){int winner=board.winner>=0?board.winner:edge.exact_winner>=0?edge.exact_winner:node->exact_winner;for(auto [parent,index]:path.edges)++parent->edges[index].pending;++root->edges[path.edges.front().second].epoch;++started;backup(path,winner==node->player?1:-1);return -1;}
  }
  if(node->pending)return 0;
  node->pending=true;node->player=board.player;capture(path);
  for(auto [parent,index]:path.edges)++parent->edges[index].pending;
  if(!path.edges.empty()){++root->edges[path.edges.front().second].epoch;++started;}
  int id=next_id++;requests.emplace(id,std::move(path));return id;
 }
 void fulfill(int id,const int64_t* actions,const double* logits,const double* values,int count,int exact=-1,Cell witness={}){
  auto found=requests.find(id);if(found==requests.end())throw std::runtime_error("Unknown request");auto& path=found->second;
  const auto& legal=path.legal;
  if(count!=int(legal.size())||count<1)throw std::runtime_error("Incomplete legal actions");
  double maximum=-1e300;for(int i=0;i<count;++i){if(legal[i]!=Cell{actions[2*i],actions[2*i+1]}||!std::isfinite(logits[i])||!std::isfinite(values[i])||std::abs(values[i])>1)throw std::runtime_error("Invalid evaluation");maximum=std::max(maximum,logits[i]);}
  auto& node=*path.leaf;double total=0;std::vector<double> weights(count);for(int i=0;i<count;++i)total+=weights[i]=std::exp(logits[i]-maximum);
  // Only root edges read their Gumbel noise and begin() redraws it, so interior edges just advance the stream.
  const bool at_root=&node==root.get();
  node.value=0;node.edges.reserve(count);for(int i=0;i<count;++i){Edge edge;edge.action=legal[i];edge.logit=logits[i]-maximum;edge.prior=weights[i]/total;node.value+=edge.prior*values[i];double u=std::generate_canonical<double,53>(rng);if(at_root)edge.gumbel=-std::log(-std::log(std::clamp(u,1e-15,1-1e-15)));node.edges.push_back(std::move(edge));}
  if(tactics)classify(path,node);
  if(exact>=0){node.exact_winner=exact;for(auto& edge:node.edges){edge.eligible=edge.action==witness;if(edge.eligible)edge.exact_winner=exact;}}
  node.expanded=true;node.pending=false;if(at_root)schedule(int(std::count_if(node.edges.begin(),node.edges.end(),[](auto& e){return e.eligible;})));backup(path,node.exact_winner<0?node.value:node.exact_winner==node.player?1:-1);requests.erase(found);
 }
 void prove(int id,const int64_t* history,int count,int player,int remaining,const int64_t* moves,int move_count){
  auto found=requests.find(id);if(found==requests.end())throw std::runtime_error("Unknown proof request");
  auto& path=found->second;
  if(count!=int(path.history.size()))throw std::runtime_error("Proof history mismatch");
  Board position;for(int i=0;i<count;++i){Cell c{history[2*i],history[2*i+1]};if(c!=path.history[i])throw std::runtime_error("Proof history mismatch");position.make(c);}
  if(move_count<1 || move_count>remaining)throw std::runtime_error("Invalid proof turn");
  Cell witness{moves[0],moves[1]};
  if(player!=position.player || remaining!=position.remaining || !position.legal(witness))throw std::runtime_error("Proof phase or move mismatch");
  Board after=position;for(int i=0;i<move_count;++i){Cell c{moves[2*i],moves[2*i+1]};if(!after.legal(c))throw std::runtime_error("Illegal proof turn");after.make(c);}
  if(after.winner<0 && after.player==player)throw std::runtime_error("Incomplete proof turn");
  Node* node=path.leaf;
  auto legal=position.legal_moves();std::vector<int64_t> actions;for(auto c:legal){actions.push_back(c.q);actions.push_back(c.r);}
  std::vector<double> zeros(legal.size());fulfill(id,actions.data(),zeros.data(),zeros.data(),int(legal.size()),player,witness);
  if(move_count==2){
   position.make(witness);auto next_legal=position.legal_moves();
   auto child=std::make_unique<Node>();child->player=player;child->expanded=true;child->exact_winner=player;
   for(auto c:next_legal){Edge e;e.action=c;e.prior=1./next_legal.size();e.eligible=c==Cell{moves[2],moves[3]};if(e.eligible)e.exact_winner=player;child->edges.push_back(std::move(e));}
   for(auto& e:node->edges)if(e.action==witness){e.child=std::move(child);break;}
  }
 }
 void cancel(){for(auto& [id,path]:requests){path.leaf->pending=false;for(auto [node,index]:path.edges)--node->edges[index].pending;if(!path.edges.empty()){--root->edges[path.edges.front().second].epoch;--started;}}requests.clear();}
 void advance(Cell action){if(!requests.empty()||!board.legal(action))throw std::runtime_error("Invalid advance");std::unique_ptr<Node> next;priority.clear();hold=false;
  for(auto& e:root->edges)if(e.action==action){next=std::move(e.child);break;}
  board.make(action);root=next?std::move(next):std::make_unique<Node>();root->player=board.player;budget=started=completed=0;
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
HX_API int hxg_begin(void* p,int simulations,int sample){try{static_cast<gumbel::Tree*>(p)->begin(simulations,sample);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxg_next(void* p){try{return static_cast<gumbel::Tree*>(p)->request();}catch(const std::exception& e){gumbel::error=e.what();return -2;}}
HX_API int hxg_history(void* p,int id,int64_t* out){auto& h=static_cast<gumbel::Tree*>(p)->requests.at(id).history;if(out)for(int i=0;i<int(h.size());++i){out[2*i]=h[i].q;out[2*i+1]=h[i].r;}return int(h.size());}
// Legal moves of a pending request in sorted (q, r) order, the actions hxg_fulfill must be given.
HX_API int hxg_legal(void* p,int id,int64_t* out){auto& l=static_cast<gumbel::Tree*>(p)->requests.at(id).legal;if(out)for(int i=0;i<int(l.size());++i){out[2*i]=l[i].q;out[2*i+1]=l[i].r;}return int(l.size());}
HX_API int hxg_fulfill(void* p,int id,const int64_t* a,const double* logits,const double* q,int n){try{static_cast<gumbel::Tree*>(p)->fulfill(id,a,logits,q,n);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
// Caller must independently verify the strategy certificate before this entry.
// Exact history and placement phase prevent applying it to a different request.
HX_API int hxg_prove(void* p,int id,const int64_t* h,int n,int player,int remaining,const int64_t* moves,int count){try{static_cast<gumbel::Tree*>(p)->prove(id,h,n,player,remaining,moves,count);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API void hxg_cancel(void* p){static_cast<gumbel::Tree*>(p)->cancel();}
HX_API int hxg_advance(void* p,int64_t q,int64_t r){try{static_cast<gumbel::Tree*>(p)->advance({q,r});return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxg_stats(void* p,int64_t* actions,int* visits,double* values,double* scores){auto& t=*static_cast<gumbel::Tree*>(p);auto& n=*t.root;if(!n.expanded)return 0;auto q=t.transformed(n);int max_epoch=0;for(auto& e:n.edges)if(e.eligible)max_epoch=std::max(max_epoch,e.epoch);for(int i=0;i<int(n.edges.size());++i){auto& e=n.edges[i];if(actions){actions[2*i]=e.action.q;actions[2*i+1]=e.action.r;visits[i]=e.visits;values[i]=e.exact_winner>=0?(e.exact_winner==n.player?1:-1):e.visits?e.sum/e.visits:n.value;scores[i]=e.eligible && max_epoch && e.epoch==max_epoch?e.gumbel+e.logit+q[i]:-std::numeric_limits<double>::infinity();}}return int(n.edges.size());}
HX_API int hxg_policy(void* p,double* out){auto& t=*static_cast<gumbel::Tree*>(p);auto& n=*t.root;if(!n.expanded)return 0;auto q=t.transformed(n);double maximum=-1e300,total=0;for(int i=0;i<int(q.size());++i){q[i]=n.edges[i].eligible?q[i]+n.edges[i].logit:-std::numeric_limits<double>::infinity();maximum=std::max(maximum,q[i]);}for(auto& v:q){v=std::exp(v-maximum);total+=v;}if(out)for(int i=0;i<int(q.size());++i)out[i]=q[i]/total;return int(q.size());}
HX_API int hxg_completed(void* p){return static_cast<gumbel::Tree*>(p)->completed;}
// Arms (enabled != 0) or clears the hold of the current search: hxg_next returns -3 once the search reaches its last
// halving boundary (the end of the search when it never halves) with no request pending, until the hold is cleared.
HX_API int hxg_hold(void* p,int enabled){static_cast<gumbel::Tree*>(p)->hold=enabled!=0;return 1;}
// Root actions (n cells, int64 q/r pairs) the opening phase of the current search samples before any other.
HX_API int hxg_priority(void* p,const int64_t* cells,int n){auto& t=*static_cast<gumbel::Tree*>(p);t.priority.clear();for(int i=0;i<n;++i)t.priority.push_back({cells[2*i],cells[2*i+1]});return 1;}
// Caller must hold a verified proof that `winner` wins after root edge (q, r); see Tree::mark.
HX_API int hxg_mark_exact(void* p,int64_t q,int64_t r,int winner){try{static_cast<gumbel::Tree*>(p)->mark({q,r},winner);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
}
