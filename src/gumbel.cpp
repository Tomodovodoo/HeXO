// Native placement-tree scheduling. Algorithm reference: DeepMind mctx.
#include "hexo.cpp"
#include <memory>
#include <random>
#include <map>
#include <numeric>
#include <string>
namespace gumbel {
struct Node;
struct Edge { Cell action;double logit=0,prior=0,sum=0,gumbel=0;int visits=0,pending=0,epoch=0;std::unique_ptr<Node> child; };
struct Node { int player=0;bool expanded=false,pending=false;double value=0;std::vector<Edge> edges; };
struct Path { Node* leaf;std::vector<std::pair<Node*,int>> edges;std::vector<Cell> history; };
struct Tree {
 Board board;std::unique_ptr<Node> root=std::make_unique<Node>();std::map<int,Path> requests;
 std::mt19937_64 rng;int budget=0,started=0,completed=0,next_id=1;std::vector<int> sequence;
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
 void begin(int simulations,int sample) {
  if(!requests.empty()||simulations<1||sample<1)throw std::runtime_error("Invalid search budget or pending requests");
  budget=simulations;started=completed=0;sequence.clear();
  int m=std::min({sample,simulations,int(board.legal_moves().size())});if(!m)return;
  std::vector<int> v(m);int considered=m,rounds=std::max(1,int(std::ceil(std::log2(m))));
  while(int(sequence.size())<budget){int extra=std::max(1,budget/(rounds*considered));for(int k=0;k<extra && int(sequence.size())<budget;++k)for(int i=0;i<considered;++i){sequence.push_back(v[i]++);if(int(sequence.size())==budget)break;}considered=m==1?1:std::max(2,considered/2);}
  for(auto& e:root->edges){e.epoch=0;double u=std::generate_canonical<double,53>(rng);e.gumbel=-std::log(-std::log(std::clamp(u,1e-15,1-1e-15)));}
 }
 void backup(Path& path,double value) {
  int player=path.leaf->player;
  for(auto i=path.edges.rbegin();i!=path.edges.rend();++i){auto& [node,index]=*i;auto& edge=node->edges[index];if(player!=node->player)value=-value;player=node->player;edge.sum+=value;++edge.visits;--edge.pending;}
  if(!path.edges.empty())++completed;
 }
 int request() {
  if(board.winner>=0||started>=budget)return 0;
  Board position=board;Node* node=root.get();Path path{node,{}, {}};
  for(auto& u:board.history)path.history.push_back(u.c);
  while(node->expanded){
   auto q=transformed(*node);int chosen=-1;double best=-1e300;
   if(node==root.get()){
    int considered=sequence[started];
    // Finish each visit layer before its values decide the next halving round.
    if(started && considered!=sequence[started-1] && !requests.empty())return 0;
    for(int i=0;i<int(node->edges.size());++i){auto& e=node->edges[i];if(e.epoch!=considered)continue;double score=e.gumbel+e.logit+(considered?q[i]:0);if(score>best){best=score;chosen=i;}}
   } else {
    double maxlog=-1e300,total=0;int visits=0;for(int i=0;i<int(q.size());++i){q[i]+=node->edges[i].logit;maxlog=std::max(maxlog,q[i]);visits+=node->edges[i].visits+node->edges[i].pending;}
    for(auto& x:q){x=std::exp(x-maxlog);total+=x;}
    for(int i=0;i<int(q.size());++i){auto& e=node->edges[i];if(e.child && e.child->pending)continue;double score=q[i]/total-double(e.visits+e.pending)/(1+visits);if(score>best){best=score;chosen=i;}}
   }
   if(chosen<0)return 0;
   auto& edge=node->edges[chosen];if(edge.child && edge.child->pending)return 0;
   path.edges.emplace_back(node,chosen);position.make(edge.action);path.history.push_back(edge.action);
   if(!edge.child){edge.child=std::make_unique<Node>();edge.child->player=position.player;}
   node=edge.child.get();path.leaf=node;
   if(position.winner>=0){for(auto [parent,index]:path.edges)++parent->edges[index].pending;++root->edges[path.edges.front().second].epoch;++started;backup(path,position.winner==node->player?1:-1);return -1;}
  }
  if(node->pending)return 0;
  node->pending=true;node->player=position.player;
  for(auto [parent,index]:path.edges)++parent->edges[index].pending;
  if(!path.edges.empty()){++root->edges[path.edges.front().second].epoch;++started;}
  int id=next_id++;requests.emplace(id,std::move(path));return id;
 }
 void fulfill(int id,const int64_t* actions,const double* logits,const double* values,int count){
  auto found=requests.find(id);if(found==requests.end())throw std::runtime_error("Unknown request");auto& path=found->second;
  Board position;for(auto c:path.history){if(!position.legal(c))throw std::runtime_error("Invalid history");position.make(c);}auto legal=position.legal_moves();
  if(count!=int(legal.size())||count<1)throw std::runtime_error("Incomplete legal actions");
  double maximum=-1e300;for(int i=0;i<count;++i){if(legal[i]!=Cell{actions[2*i],actions[2*i+1]}||!std::isfinite(logits[i])||!std::isfinite(values[i])||std::abs(values[i])>1)throw std::runtime_error("Invalid evaluation");maximum=std::max(maximum,logits[i]);}
  auto& node=*path.leaf;double total=0;for(int i=0;i<count;++i)total+=std::exp(logits[i]-maximum);
  node.value=0;for(int i=0;i<count;++i){Edge edge;edge.action=legal[i];edge.logit=logits[i]-maximum;edge.prior=std::exp(edge.logit)/total;node.value+=edge.prior*values[i];double u=std::generate_canonical<double,53>(rng);edge.gumbel=-std::log(-std::log(std::clamp(u,1e-15,1-1e-15)));node.edges.push_back(std::move(edge));}
  node.expanded=true;node.pending=false;backup(path,node.value);requests.erase(found);
 }
 void cancel(){for(auto& [id,path]:requests){path.leaf->pending=false;for(auto [node,index]:path.edges)--node->edges[index].pending;if(!path.edges.empty()){--root->edges[path.edges.front().second].epoch;--started;}}requests.clear();}
 void advance(Cell action){if(!requests.empty()||!board.legal(action))throw std::runtime_error("Invalid advance");std::unique_ptr<Node> next;
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
HX_API int hxg_begin(void* p,int simulations,int sample){try{static_cast<gumbel::Tree*>(p)->begin(simulations,sample);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxg_next(void* p){try{return static_cast<gumbel::Tree*>(p)->request();}catch(const std::exception& e){gumbel::error=e.what();return -2;}}
HX_API int hxg_history(void* p,int id,int64_t* out){auto& h=static_cast<gumbel::Tree*>(p)->requests.at(id).history;if(out)for(int i=0;i<int(h.size());++i){out[2*i]=h[i].q;out[2*i+1]=h[i].r;}return int(h.size());}
HX_API int hxg_fulfill(void* p,int id,const int64_t* a,const double* logits,const double* q,int n){try{static_cast<gumbel::Tree*>(p)->fulfill(id,a,logits,q,n);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API void hxg_cancel(void* p){static_cast<gumbel::Tree*>(p)->cancel();}
HX_API int hxg_advance(void* p,int64_t q,int64_t r){try{static_cast<gumbel::Tree*>(p)->advance({q,r});return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxg_stats(void* p,int64_t* actions,int* visits,double* values,double* scores){auto& t=*static_cast<gumbel::Tree*>(p);auto& n=*t.root;if(!n.expanded)return 0;auto q=t.transformed(n);int max_epoch=0;for(auto& e:n.edges)max_epoch=std::max(max_epoch,e.epoch);for(int i=0;i<int(n.edges.size());++i){auto& e=n.edges[i];if(actions){actions[2*i]=e.action.q;actions[2*i+1]=e.action.r;visits[i]=e.visits;values[i]=e.visits?e.sum/e.visits:n.value;scores[i]=max_epoch && e.epoch==max_epoch?e.gumbel+e.logit+q[i]:-std::numeric_limits<double>::infinity();}}return int(n.edges.size());}
HX_API int hxg_policy(void* p,double* out){auto& t=*static_cast<gumbel::Tree*>(p);auto& n=*t.root;if(!n.expanded)return 0;auto q=t.transformed(n);double maximum=-1e300,total=0;for(int i=0;i<int(q.size());++i){q[i]+=n.edges[i].logit;maximum=std::max(maximum,q[i]);}for(auto& v:q){v=std::exp(v-maximum);total+=v;}if(out)for(int i=0;i<int(q.size());++i)out[i]=q[i]/total;return int(q.size());}
HX_API int hxg_completed(void* p){return static_cast<gumbel::Tree*>(p)->completed;}
}
