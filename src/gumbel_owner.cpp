// One host phase owns each game graph. Caller-side control and proof delivery
// run between joined selection/installation phases.
#include "gumbel.cpp"
#include <chrono>
#include <deque>
#include <condition_variable>
#include <mutex>
extern "C" int hxgf_begin(void*,void*,const int64_t*,int);
extern "C" void* hxgf_new(int);
extern "C" void hxgf_free(void*);
extern "C" void hxgf_stats(void*,int64_t*);
extern "C" int64_t hxgf_queued(void*);
extern "C" int hxgf_workers(void*,int,void* (*)(void*));
extern "C" int hxgf_parallel(void*,int,void (*)(void*,int),void*,bool (*)(void*,int));
extern "C" int hxgf_gather(void*,void*,int64_t*);
extern "C" void hxgf_detach(void*,void*);
extern "C" int hxgf_root_value(void*,void*,double*);
extern "C" int hxgf_install(void*,const uint64_t*,int,const int64_t*,const int64_t*,const double*,const double*,void**,int);
namespace owner {
using gumbel::Tree;using gumbel::Key;using gumbel::KeyHash;
struct Candidate {std::vector<Cell> history;double relevance=0,cost=1;uint64_t seen=0,last=0,completed=0;int depth=0;bool solver=false,active=false,settled=false;};
struct View {std::unique_ptr<Tree> tree;Key key;std::vector<Cell> history;double relevance=1;uint64_t id=0,generation=0,completed=0,issued=0,cancelled=0;int depth=0,passes=0;bool active=false,discovered=false;};
struct Record {uint64_t id,generation,completed,shared,context_a,context_b;int depth,exact;bool raw_known,estimate_known;double value,net,ms;std::vector<Cell> history;};
struct FeedDeleter {void operator()(void* p)const{if(p)hxgf_free(p);}};
struct Owner {
 std::unique_ptr<void,FeedDeleter> owned_feed;void* feed;std::shared_ptr<gumbel::GameStore> game;std::vector<View> views;std::unordered_map<Key,Candidate,KeyHash> candidates;
 std::deque<Record> records;std::vector<gumbel::RootEdge> last_root;
 std::vector<uint64_t> direct_root_credits;std::vector<int> last_members;
 std::vector<Cell> focus;std::mt19937_64 rng;uint64_t next_id=1,ticks=0,allocations=0,reclaimed=0,completed=0,issued=0,cancelled=0,created=0,retired=0,step_ns=0,discover_ns=0;
 int quantum,max_views,max_depth,sample_limit=16;size_t view_cursor=0;uint64_t work_limit;double time_limit_ms;std::chrono::steady_clock::time_point started;
 bool stopped=false,deadline=false,root_raw_known=false;double exploration=.2,root_raw=0;
 Owner(Tree& source,int capacity,int q,int count,int depth,uint64_t work,double ms,uint64_t seed,void* common=nullptr,int samples=16):owned_feed(common?nullptr:hxgf_new(capacity)),feed(common?common:owned_feed.get()),game(source.state),rng(seed),quantum(q),max_views(count),max_depth(depth),sample_limit(samples),work_limit(work),time_limit_ms(ms),started(std::chrono::steady_clock::now()){
  if(!source.shared || !feed || q<4 || count<1 || count>64 || depth<1 || depth>32 || !std::isfinite(ms) || ms<0 || (!work && !ms))throw std::runtime_error("Invalid native owner limits");
  if(game->scheduler_owner)throw std::runtime_error("Game already has a native owner");
  for(auto& [key,weak]:game->nodes)if(auto n=weak.lock())if(n->pending)throw std::runtime_error("Game has external pending neural work");
  for(auto& u:source.board.history)focus.push_back(u.c);views.reserve(max_views);
  View root;root.tree=std::make_unique<Tree>(rng(),game);root.tree->shared=root.tree->graph=true;
  root.tree->scheduler_owned=true;root.tree->tactics=source.tactics;root.tree->range_floor=source.range_floor;root.tree->root_noise=source.root_noise;root.tree->round_barrier=source.round_barrier;
  game->primary=root.tree.get();root.tree->root_at(focus);root.key=gumbel::keys(focus).second;root.history=focus;root.id=next_id++;views.push_back(std::move(root));++created;
  start(views[0]);game->scheduler_owner=this;
 }
 ~Owner(){stop();if(game->scheduler_owner==this)game->scheduler_owner=nullptr;}
 double elapsed()const{return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();}
 bool expired()const{return time_limit_ms && elapsed()>=time_limit_ms;}
 uint64_t reserved()const{uint64_t n=0;for(auto& v:views)if(v.active)n+=v.tree->budget;return n;}
 void attach(View& v){std::vector<int64_t> h;for(auto c:v.history){h.push_back(c.q);h.push_back(c.r);}if(!hxgf_begin(feed,v.tree.get(),h.data(),int(v.history.size())))throw std::runtime_error(gumbel::error);}
 bool start(View& v){
  uint64_t room=work_limit?work_limit-std::min(work_limit,completed+reserved()):uint64_t(quantum);
  int budget=int(std::min<uint64_t>(quantum,room));if(!budget || stopped || expired())return false;
  v.tree->begin(budget,std::min(sample_limit,budget));v.active=true;v.discovered=false;++v.generation;++allocations;
  candidates[v.key].active=true;attach(v);return true;
 }
 void save_credits(Tree& t){
  if(direct_root_credits.empty())direct_root_credits.resize(t.root_edges.size());
  if(direct_root_credits.size()!=t.root_edges.size())throw std::runtime_error("Focus legal set changed within owner");
  for(size_t i=0;i<t.root_edges.size();++i)direct_root_credits[i]+=t.root_edges[i].credits;
 }
 void record(View& v,uint64_t credits){
  auto& t=*v.tree;auto& node=*t.root;double raw=0;
  bool raw_known=hxgf_root_value(feed,v.tree.get(),&raw)!=0;
  if(!v.depth && raw_known){root_raw=raw;root_raw_known=true;}
  int winner=t.board.winner>=0?t.board.winner:node.exact_winner;
  bool known=winner>=0 || node.expanded || node.n>0;
  double estimate=winner>=0?(winner==node.player?1.:-1.):node.q;
  records.push_back({v.id,v.generation,credits,uint64_t(node.n),v.key.a,v.key.b,v.depth,winner,raw_known,known,estimate,raw,elapsed(),v.history});
  if(records.size()>2048)records.pop_front();
 }
 void finish(View& v){
  auto& t=*v.tree;if(t.root->expanded)t.renew(*t.root);
  uint64_t credits=t.completed;completed+=credits;issued+=t.issued;cancelled+=t.cancelled;
  v.completed+=credits;v.issued+=t.issued;v.cancelled+=t.cancelled;++v.passes;v.active=false;
  if(!v.depth)save_credits(t);
  auto& c=candidates[v.key];c.active=false;c.history=v.history;c.depth=v.depth;c.relevance=std::max(c.relevance,v.relevance);c.completed+=credits;c.last=allocations;c.seen=allocations;
  record(v,credits);
  if(v.depth==0 && t.completed){last_root=t.root_edges;last_members=t.round.members;}
  discover(v);
 }
 void discover(View& v){
  auto begin=std::chrono::steady_clock::now();auto& t=*v.tree;auto& node=*t.root;
  if(!node.expanded || node.exact_winner>=0 || v.depth>=max_depth)return;
  if(candidates.size()>=14336){
   std::vector<std::pair<double,Key>> victims;for(auto& [key,c]:candidates)if(c.depth && !c.active)victims.push_back({c.relevance/std::sqrt(1.+c.completed),key});
   std::sort(victims.begin(),victims.end(),[](const auto& a,const auto& b){if(a.first!=b.first)return a.first<b.first;if(a.second.a!=b.second.a)return a.second.a<b.second.a;return a.second.b<b.second.b;});
   size_t drop=std::min(size_t(2048),victims.size());for(size_t i=0;i<drop;++i){candidates.erase(victims[i].second);++reclaimed;}
  }
  t.current(node);auto q=t.transformed(node);double maximum=-1e300,total=0;int legal=0;
  for(size_t i=0;i<q.size();++i){q[i]=node.edges[i].read().eligible?q[i]+node.edges[i].logit:-std::numeric_limits<double>::infinity();if(node.edges[i].read().eligible){maximum=std::max(maximum,q[i]);++legal;}}
  for(double x:q)total+=std::exp(x-maximum);
  if(!legal || !total)return;
  for(size_t i=0;i<node.edges.size();++i){auto& edge=node.edges[i];if(!edge.read().eligible || (edge.read().child && edge.read().child->exact_winner>=0))continue;
   Key key=gumbel::child_keys(node.position,v.history,edge.action).second;
   if(key==views[0].key)continue;
   if(!candidates.contains(key) && candidates.size()>=16384)continue;
   auto [it,inserted]=candidates.try_emplace(key);auto& c=it->second;
   if(inserted || c.history.empty()){c.history=v.history;c.history.push_back(edge.action);}
   c.depth=v.depth+1;c.seen=allocations;
   double share=(1-exploration)*std::exp(q[i]-maximum)/total+exploration/legal;
   double relevance=v.relevance*share*.85;
   // Evidence under another incoming path may improve relevance; no legal move is deleted by this ranking.
   c.relevance=std::max(c.relevance,relevance);
   if(edge.read().child && edge.read().child->n)c.cost=std::max(1.,double(edge.read().child->edges.size())/256.);
  }
  v.discovered=true;discover_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();
 }
 bool live(const Key& key)const {auto c=candidates.find(key);return c!=candidates.end() && c->second.active;}
 // Solver paths enter the same bounded queue as internally discovered views.
 // Every intermediate position remains searchable; a quiet endpoint is not a proof.
 size_t solver_path(const std::vector<Cell>& history,size_t begin,double relevance){
  if(stopped || expired() || begin>=history.size())return 0;
  size_t added=0;std::vector<Cell> prefix(history.begin(),history.begin()+begin);
  for(size_t i=begin;i<history.size();++i){
   prefix.push_back(history[i]);if(prefix.size()<=focus.size())continue;
   // Automatic neural discovery keeps max_depth. A visited CPU line can end
   // deeper; cap this separate, explicit handoff at 128 placements from focus.
   int depth=int(prefix.size()-focus.size());if(depth>128)break;
   auto keys=gumbel::keys(prefix);Key key=keys.second;
   if(game->outcomes.contains(keys.first))break;
   if(key==views[0].key || live(key))continue;
   if(auto found=candidates.find(key);found!=candidates.end()){
    found->second.solver=true;found->second.relevance=std::max(found->second.relevance,relevance/(1.+.05*double(i-begin)));continue;
   }
   if(candidates.size()>=16384)break;
   auto found=game->nodes.find(key);if(found!=game->nodes.end())if(auto node=found->second.lock())if(node->exact_winner>=0)break;
   Candidate c;c.history=prefix;c.depth=depth;c.solver=true;c.relevance=relevance/(1.+.05*double(i-begin));c.seen=allocations;
   candidates.emplace(key,std::move(c));++added;
  }
  return added;
 }
 Candidate* choose(Key& selected){
  const bool explore=allocations%5==4;
  bool refresh=false;
  for(;;){
   Candidate* best=nullptr;double score=-1;
   for(auto& [key,c]:candidates){if(c.depth==0 || (c.depth>max_depth && !c.solver) || c.active || c.settled)continue;
    // A newly exact leader triggers one complete refresh, not one rescan for
    // each proven candidate. Ordinary allocations need no per-candidate lookup.
    if(refresh)if(auto n=game->nodes.find(key);n!=game->nodes.end())if(auto node=n->second.lock())if(node->exact_winner>=0){c.settled=true;continue;}
    double age=double(allocations-c.last+1);double x=explore?age/std::sqrt(1.+c.completed):c.relevance*(1.+std::min(16.,age/16.))/std::sqrt(c.cost*(1.+double(c.completed)/quantum));
    // Stable key breaks ties; pointer/map iteration is not the schedule's ordering.
    if(x>score || (x==score && (key.a<selected.a || (key.a==selected.a && key.b<selected.b)))){score=x;best=&c;selected=key;}
   }
   if(!best || refresh)return best;
   // Only admission needs graph evidence. An exact verdict cannot become
   // unknown in this owner's fixed game; estimates and failed proofs do not
   // settle a candidate. Keep every unresolved candidate available.
   if(auto n=game->nodes.find(selected);n!=game->nodes.end())if(auto node=n->second.lock())if(node->exact_winner>=0){best->settled=true;refresh=true;continue;}
   return best;
  }
 }
 int allocate(){
  if(stopped || expired())return -1;size_t slot=1;
  while(slot<views.size() && views[slot].active)++slot;
  if(slot==views.size() && views.size()>=size_t(max_views))return -1;
  Key key{};auto* c=choose(key);if(!c)return -1;
  if(work_limit && completed+reserved()>=work_limit)return -1;
  auto history=c->history;double relevance=c->relevance;int depth=c->depth;c->last=allocations;
  if(slot<views.size()){hxgf_detach(feed,views[slot].tree.get());views[slot].tree->cancel();++retired;}
  View v;v.tree=std::make_unique<Tree>(rng(),game);v.tree->shared=v.tree->graph=true;
  v.tree->scheduler_owned=true;v.tree->tactics=views[0].tree->tactics;v.tree->range_floor=views[0].tree->range_floor;v.tree->root_noise=views[0].tree->root_noise;v.tree->round_barrier=views[0].tree->round_barrier;
  v.tree->root_at(history);v.history=std::move(history);v.key=key;v.depth=depth;v.relevance=relevance;v.id=next_id++;++created;
  if(slot==views.size())views.push_back(std::move(v));else views[slot]=std::move(v);
  return start(views[slot])?int(slot):-1;
 }
 int step(int ready_limit=0,bool admitted=false){
  if(stopped)return 0;auto begin=std::chrono::steady_clock::now();++ticks;
  if(expired()){deadline=true;stop();return 0;}
  auto* root=views[0].tree.get();
  for(auto& v:views){if(v.active && v.tree->done())finish(v);else if(v.active && !v.discovered && v.tree->root->expanded)discover(v);if(expired()){deadline=true;stop();return 0;}}
  if(root->board.winner>=0 || (root->root->expanded && root->root->exact_winner>=0)){stop();return 0;}
  if(!views[0].active)start(views[0]);
  // Keep one continuation eligible even under a tiny watermark. Beyond that,
  // existing views get the first chance to supply work before opening more.
  if(std::none_of(views.begin()+1,views.end(),[](const auto& v){return v.active;}))allocate();
  int progress=0;
  auto gather=[&](View& v){
   if(expired()){deadline=true;stop();return;}
   int64_t out[4];int status=hxgf_gather(feed,v.tree.get(),out);
   if(status==-2)throw std::runtime_error(gumbel::error);
   progress+=int(out[0])+int(out[3]);
   if(expired()){deadline=true;stop();return;}
   if(root->board.winner>=0 || (root->root->expanded && root->root->exact_winner>=0))stop();
  };
  size_t visited=0,first=ready_limit?view_cursor:0;
  while(visited<views.size()){
   if(ready_limit && !admitted && hxgf_queued(feed)>=ready_limit)break;
   auto& v=views[(first+visited++)%views.size()];if(!v.active)continue;
   admitted=false;gather(v);
   // Cached fulfillments and exact edges advance search without adding NN rows.
   if(stopped)break;
  }
  // Pause between whole gathers, never inside a root visit layer. Resume with
  // the next view so deeper work remains eligible when the queue has space.
  if(ready_limit)view_cursor=(first+visited)%views.size();
  while(!stopped && (!ready_limit || hxgf_queued(feed)<ready_limit)){
   int slot=allocate();if(slot<0)break;gather(views[slot]);
  }
  bool active=false;for(auto& v:views)active|=v.active;
  if(!active)stop();
  views[0].tree->trim_archive(false,false);
  step_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();return progress;
 }
 void stop(){
  if(stopped)return;stopped=true;
  for(auto& v:views){bool active=v.active;if(active){
    auto& t=*v.tree;if(t.root->expanded)t.renew(*t.root);completed+=t.completed;issued+=t.issued;v.completed+=t.completed;v.issued+=t.issued;
    if(!v.depth)save_credits(t);
    // The last install may complete a lease before step() calls finish().
    if(t.done()){++v.passes;if(!v.depth && t.completed){last_root=t.root_edges;last_members=t.round.members;}}
    record(v,t.completed);
   }
   hxgf_detach(feed,v.tree.get());v.tree->cancel();if(active){cancelled+=v.tree->cancelled;v.cancelled+=v.tree->cancelled;}v.active=false;
   if(auto c=candidates.find(v.key);c!=candidates.end())c->second.active=false;
  }
  views[0].tree->trim_archive(false,false);
 }
 int install(const uint64_t* ids,int count,const int64_t* offsets,const int64_t* actions,const double* logits,const double* values){
  std::vector<void*> failures(views.size());int stopped_count=hxgf_install(feed,ids,count,offsets,actions,logits,values,failures.data(),int(failures.size()));
  if(stopped_count<0)throw std::runtime_error(gumbel::error);if(stopped_count){stop();throw std::runtime_error("Inference failed for native owner view");}return 1;
 }
 int choice(int64_t* action){
  auto& t=*views[0].tree;t.proof_root();auto& n=*t.root;if(!n.expanded)return 0;t.current(n);auto q=t.transformed(n);t.trim_archive(false,false);
  // Deeper same-position values may refresh the choice, but an incomplete new
  // comparison does not replace the last completed root sampling credits.
  const auto& epochs=last_root.empty()?t.root_edges:last_root;int maximum=0,seen=0;
  for(size_t i=0;i<n.edges.size() && i<epochs.size();++i){seen=std::max(seen,epochs[i].epoch);if(n.edges[i].read().eligible)maximum=std::max(maximum,epochs[i].epoch);}
  const auto& members=last_root.empty()?t.round.members:last_members;
  bool round_survivor=t.round_barrier && std::any_of(members.begin(),members.end(),[&](int i){return i>=0 && n.edges[i].read().eligible;});
  double best=-1e300;int selected=-1;
  for(size_t i=0;i<n.edges.size();++i){auto& e=n.edges[i];if(!e.read().eligible)continue;
   bool finalist=round_survivor?std::find(members.begin(),members.end(),int(i))!=members.end():i<epochs.size() && epochs[i].epoch==maximum;
    if(n.exact_winner<0 && seen && !finalist)continue;
   double score=e.logit+q[i]+t.bonus(e);if(seen && i<epochs.size())score+=epochs[i].gumbel;
   if(score>best){best=score;selected=int(i);}
  }if(selected<0)return 0;action[0]=n.edges[selected].action.q;action[1]=n.edges[selected].action.r;return 1;
 }
};
}
extern "C" HX_API void* hxgo_new(void* tree,int capacity,int quantum,int views,int depth,uint64_t work,double ms,uint64_t seed){try{if(!tree)throw std::runtime_error("Missing source tree");return new owner::Owner(*static_cast<gumbel::Tree*>(tree),capacity,quantum,views,depth,work,ms,seed);}catch(const std::exception& e){gumbel::error=e.what();return nullptr;}}
extern "C" HX_API int hxgo_free(void* p){auto& o=*static_cast<owner::Owner*>(p);if(!o.owned_feed){gumbel::error="Borrowed owner belongs to its pool";return 0;}o.stop();int64_t stats[6];hxgf_stats(o.feed,stats);if(stats[4]){gumbel::error="Drain or abandon submitted batches after their GPU fence before freeing owner";return 0;}delete &o;return 1;}
extern "C" HX_API int hxgo_step(void* p){try{return static_cast<owner::Owner*>(p)->step();}catch(const std::exception& e){gumbel::error=e.what();return -1;}}
extern "C" HX_API int hxgo_step_ready(void* p,int limit){try{if(limit<1)throw std::runtime_error("Positive native ready limit required");return static_cast<owner::Owner*>(p)->step(limit);}catch(const std::exception& e){gumbel::error=e.what();return -1;}}
extern "C" HX_API void hxgo_cancel(void* p){static_cast<owner::Owner*>(p)->stop();}
extern "C" HX_API int hxgo_done(void* p){return static_cast<owner::Owner*>(p)->stopped;}
extern "C" HX_API void* hxgo_feed(void* p){return static_cast<owner::Owner*>(p)->feed;}
extern "C" HX_API int hxgo_admit(void* p){auto& o=*static_cast<owner::Owner*>(p);if(o.expired()){o.deadline=true;o.stop();}return !o.stopped;}
extern "C" HX_API int hxgo_clock(void* p,double ms){auto& o=*static_cast<owner::Owner*>(p);if(o.ticks || o.stopped || !(ms>0) || !std::isfinite(ms)){gumbel::error="Arm clock before first step";return 0;}o.time_limit_ms=ms;o.work_limit=0;o.started=std::chrono::steady_clock::now();return 1;}
extern "C" HX_API void* hxgo_root(void* p){return static_cast<owner::Owner*>(p)->views[0].tree.get();}
extern "C" HX_API int hxgo_install(void* p,const uint64_t* ids,int count,const int64_t* offsets,const int64_t* actions,const double* logits,const double* values){try{return static_cast<owner::Owner*>(p)->install(ids,count,offsets,actions,logits,values);}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
extern "C" HX_API int hxgo_choice(void* p,int64_t* action){try{return static_cast<owner::Owner*>(p)->choice(action);}catch(const std::exception& e){gumbel::error=e.what();return -1;}}
extern "C" HX_API void hxgo_stats(void* p,uint64_t* out){auto& o=*static_cast<owner::Owner*>(p);uint64_t live=0,pending=0,depth=0,root_completed=o.views[0].completed,total_completed=o.completed,total_issued=o.issued,total_cancelled=o.cancelled;
 for(auto& v:o.views){live+=v.active;if(v.active){total_completed+=v.tree->completed;total_issued+=v.tree->issued;total_cancelled+=v.tree->cancelled;}pending+=v.tree->requests.size();depth=std::max(depth,uint64_t(v.depth));}if(o.views[0].active)root_completed+=o.views[0].tree->completed;
 uint64_t last_credits=0;for(auto& edge:o.last_root)last_credits+=edge.credits;
 std::array<uint64_t,20> a{o.ticks,total_completed,total_issued,total_cancelled,o.created,o.retired,uint64_t(o.candidates.size()),live,pending,depth,root_completed,uint64_t(o.deadline),o.step_ns,o.discover_ns,uint64_t(o.records.size()),uint64_t(o.views.size()),last_credits,uint64_t(o.views[0].passes),o.allocations,o.reclaimed};std::copy(a.begin(),a.end(),out);}
extern "C" HX_API int hxgo_records(void* p,uint64_t* metadata,double* values){auto& o=*static_cast<owner::Owner*>(p);if(metadata && values)for(size_t i=0;i<o.records.size();++i){auto& r=o.records[i];std::array<uint64_t,10> row{r.id,r.generation,r.completed,r.shared,r.context_a,r.context_b,uint64_t(r.depth),uint64_t(r.exact+1),uint64_t(r.raw_known),uint64_t(r.estimate_known)};std::copy(row.begin(),row.end(),metadata+10*i);values[3*i]=r.value;values[3*i+1]=r.net;values[3*i+2]=r.ms;}return int(o.records.size());}
// Diagnostic only: preserve shared Q while exposing which count drives its temperature.
extern "C" HX_API int hxgo_policy_audit(void* p,double* out){auto& o=*static_cast<owner::Owner*>(p);auto& t=*o.views[0].tree;auto& node=*t.root;if(!node.expanded)return 0;t.current(node);int maximum=0;auto raw=t.completed_q(node,maximum);
 if(out)for(size_t i=0;i<node.edges.size();++i){auto& e=node.edges[i];uint64_t direct=i<o.direct_root_credits.size()?o.direct_root_credits[i]:0;if(o.views[0].active && i<t.root_edges.size())direct+=t.root_edges[i].credits;
  std::array<double,9> row{double(e.action.q),double(e.action.r),e.logit,raw[i],double(e.read().visits),double(i<t.root_edges.size()?t.root_edges[i].credits:0),double(i<o.last_root.size()?o.last_root[i].credits:0),e.read().eligible?1.:0.,double(direct)};std::copy(row.begin(),row.end(),out+9*i);}return int(node.edges.size());}

// One writer per game and one neural queue across independent games of one model.
namespace owner {
// A worker can retain this signal after its broker detaches. It never owns or
// dereferences a graph, producer, or broker through the notification.
struct Signal {
 std::mutex mutex;std::condition_variable wake;
 void notify(){std::lock_guard lock(mutex);wake.notify_all();}
};
struct Pool {
 std::unique_ptr<void,FeedDeleter> owned_feed;void* feed;
 std::vector<std::unique_ptr<Owner>> games;std::vector<bool> failed;
 std::string model;size_t cursor=0;int ready_limit=0,host_workers=1;uint64_t steps=0,retargets=0;bool stopped=false;
 void* proof_owner=nullptr;void (*proof_step)(void*)=nullptr;uint64_t (*proof_collect)(void*)=nullptr;void (*proof_retarget)(void*,int)=nullptr;
 void (*proof_bind)(void*,int)=nullptr;
 bool (*proof_ready)(void*)=nullptr;void (*proof_listen)(void*,std::shared_ptr<Signal>)=nullptr;
 void* inference_owner=nullptr;
 Pool(void** sources,int count,int capacity,int quantum,int views,int depth,uint64_t work,const char* version,uint64_t seed):owned_feed(hxgf_new(capacity)),feed(owned_feed.get()),model(version?version:""){
  if(!sources || count<1 || count>1024 || !feed || model.empty())throw std::runtime_error("Invalid multi-game pool");
  games.reserve(count);failed.resize(count);
  for(int i=0;i<count;++i){
   if(!sources[i])throw std::runtime_error("Missing pool source");
   games.push_back(std::make_unique<Owner>(*static_cast<Tree*>(sources[i]),capacity,quantum,views,depth,work,0,seed+i,feed));
  }
 }
 ~Pool(){stop();
#ifdef HEXO_RECLAIM_PROFILE
  using Clock=std::chrono::steady_clock;
  auto ns=[](auto start){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count());};
  uint64_t views=0,view_ns=0,aux_ns=0,graph_ns=0;auto count=games.size();
  for(auto& owner:games){
   auto graph=owner->game;views+=owner->views.size();
   auto start=Clock::now();owner->views.clear();view_ns+=ns(start);
   start=Clock::now();owner.reset();aux_ns+=ns(start);
   start=Clock::now();graph.reset();graph_ns+=ns(start);
  }
  auto start=Clock::now();owned_feed.reset();auto feed_ns=ns(start);
  std::fprintf(stderr,"HEXO_RECLAIM {\"component\":\"pool\",\"model\":\"%s\",\"games\":%llu,\"views\":%llu,\"views_ns\":%llu,\"aux_ns\":%llu,\"graph_ns\":%llu,\"feed_ns\":%llu}\n",model.c_str(),
   (unsigned long long)count,(unsigned long long)views,(unsigned long long)view_ns,(unsigned long long)aux_ns,(unsigned long long)graph_ns,(unsigned long long)feed_ns);
#endif
 }
 void stop(){for(auto& o:games)o->stop();stopped=true;}
 int step(bool neural=true){
  if(proof_step)proof_step(proof_owner);
  if(stopped)return 0;int progress=0;++steps;
  // Backpressure pauses new neural selection, not deadlines or proof delivery.
  for(auto& o:games)if(!o->stopped){
   auto& root=*o->views[0].tree;
   if(o->expired()){o->deadline=true;o->stop();}
   else {root.proof_root();if(root.board.winner>=0 || (root.root->expanded && root.root->exact_winner>=0))o->stop();}
  }
  if(!neural){stopped=std::all_of(games.begin(),games.end(),[](const auto& o){return o->stopped;});return 0;}
  size_t visited=0;
  if(host_workers>1 && (!ready_limit || hxgf_queued(feed)<ready_limit)){
   struct Phase {Pool* pool;std::vector<int> progress;size_t admitted=0;bool closed=false;} phase{this,std::vector<int>(games.size())};
   if(!hxgf_parallel(feed,int(games.size()),[](void* data,int index){
    auto& phase=*static_cast<Phase*>(data);auto& p=*phase.pool;
    auto& o=*p.games[(p.cursor+index)%p.games.size()];
    if(!o.stopped)phase.progress[index]=o.step(p.ready_limit,true);
   },&phase,[](void* data,int){
    auto& phase=*static_cast<Phase*>(data);auto& p=*phase.pool;
    if(phase.closed)return false;
    if(p.ready_limit && hxgf_queued(p.feed)>=p.ready_limit){phase.closed=true;return false;}
    ++phase.admitted;return true;
   }))throw std::runtime_error(gumbel::error);
   visited=phase.admitted;
   for(int value:phase.progress)progress+=value;
  }else while(visited<games.size()){
   if(ready_limit && hxgf_queued(feed)>=ready_limit)break;
   auto& o=*games[(cursor+visited++)%games.size()];if(!o.stopped)progress+=o.step(ready_limit);
  }
  // A full pass keeps normal rotation; a paused pass resumes at the first game
  // not visited. Enabling a watermark alone must not pin the same first game.
  cursor=(cursor+(ready_limit && visited<games.size()?visited:1))%games.size();stopped=std::all_of(games.begin(),games.end(),[](const auto& o){return o->stopped;});return progress;
 }
 bool admit(){
  if(proof_step)proof_step(proof_owner);
  bool active=false;for(auto& o:games){if(o->expired()){o->deadline=true;o->stop();}active|=!o->stopped;}
  stopped=!active;return active;
 }
 void clock(double ms){
  if(steps || stopped || !(ms>0) || !std::isfinite(ms))throw std::runtime_error("Arm pool clock before first step");
  auto now=std::chrono::steady_clock::now();for(auto& o:games){o->time_limit_ms=ms;o->work_limit=0;o->started=now;}
 }
 void retarget(int index,const int64_t* cells,int count,uint64_t work,double ms){
  if(index<0 || index>=int(games.size()) || count<0 || (count && !cells) || !std::isfinite(ms) || ms<0 || (!work && !ms))throw std::runtime_error("Invalid retarget");
  std::vector<Cell> history;history.reserve(count);Board checked;
  for(int i=0;i<count;++i){Cell cell{cells[2*i],cells[2*i+1]};if(!checked.legal(cell))throw std::runtime_error("Illegal retarget history");checked.make(cell);history.push_back(cell);}
  if(proof_retarget)proof_retarget(proof_owner,index);
  auto& o=*games[index];o.stop();o.views.resize(1);auto& root=o.views[0];
  // Keep this Tree address alive for outstanding solver DTOs. Packed neural rows
  // already own their encoding snapshot; cancelled subscribers never install late.
  root.tree->root_at(history);root.history=history;root.key=gumbel::keys(history).second;
  root.completed=root.issued=root.cancelled=0;root.passes=0;root.active=root.discovered=false;
  o.focus=std::move(history);o.candidates.clear();o.last_root.clear();o.last_members.clear();o.direct_root_credits.clear();
  o.root_raw_known=false;o.root_raw=0;
  o.completed=o.issued=o.cancelled=o.ticks=o.allocations=o.reclaimed=o.retired=o.step_ns=o.discover_ns=0;o.created=1;o.view_cursor=0;
  o.work_limit=work;o.time_limit_ms=ms;o.started=std::chrono::steady_clock::now();o.stopped=o.deadline=false;
  failed[index]=false;stopped=false;++retargets;o.start(root);
 }
 std::unique_ptr<Owner> replace(int index,Tree& source,int samples,int views,uint64_t work,double ms,double noise,uint64_t seed){
  auto& old=*games[index];int quantum=old.quantum,depth=old.max_depth;
  // Retirement already drained this slot's proof jobs and detached neural
  // subscribers. Device snapshots own copied encodings, not the old store.
  source.root_noise=noise;
  auto replacement=std::make_unique<Owner>(source,0,quantum,views,depth,(work || ms)?work:uint64_t(quantum),ms,seed,feed,samples);
  auto retired=std::move(games[index]);games[index]=std::move(replacement);
  if(!work && !ms)games[index]->stop();
  if(proof_bind)proof_bind(proof_owner,index);
  failed[index]=false;stopped=false;++retargets;return retired;
 }
 int install(const uint64_t* ids,int count,const int64_t* offsets,const int64_t* actions,const double* logits,const double* values){
  size_t capacity=0;for(auto& o:games)capacity+=o->views.size();std::vector<void*> failures(capacity);
  int num=hxgf_install(feed,ids,count,offsets,actions,logits,values,failures.data(),int(capacity));
  if(num<0)throw std::runtime_error(gumbel::error);
  // An unencodable leaf stops its game. Other games and shared subscribers continue.
  for(int j=0;j<num;++j)for(size_t i=0;i<games.size();++i)for(auto& v:games[i]->views)if(v.tree.get()==failures[j]){failed[i]=true;games[i]->stop();}
  return 1;
 }
};
Pool& caller(void* p){auto& pool=*static_cast<Pool*>(p);if(pool.inference_owner)throw std::runtime_error("Pool belongs to its native inference producer");return pool;}
}
extern "C" HX_API void* hxgm_new(void** sources,int count,int capacity,int quantum,int views,int depth,uint64_t work,const char* version,uint64_t seed){try{return new owner::Pool(sources,count,capacity,quantum,views,depth,work,version,seed);}catch(const std::exception& e){gumbel::error=e.what();return nullptr;}}
extern "C" HX_API int hxgm_free(void* p){auto& pool=*static_cast<owner::Pool*>(p);if(pool.inference_owner){gumbel::error="Close the inference service before freeing its search pool";return 0;}if(pool.proof_owner){gumbel::error="Close the proof loop before freeing its search pool";return 0;}pool.stop();int64_t stats[6];hxgf_stats(pool.feed,stats);if(stats[4]){gumbel::error="Drain or fenced-abandon global tasks before freeing pool";return 0;}delete &pool;return 1;}
extern "C" HX_API int hxgm_step(void* p){try{return owner::caller(p).step();}catch(const std::exception& e){gumbel::error=e.what();return -1;}}
extern "C" HX_API int hxgm_ready_limit(void* p,int rows){try{if(rows<0 || rows>16384)throw std::runtime_error("Invalid neural ready limit");owner::caller(p).ready_limit=rows;return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
extern "C" HX_API int hxgm_workers(void* p,int count){try{
 auto& pool=owner::caller(p);if(pool.steps)throw std::runtime_error("Configure native host workers before the first phase");
 if(!hxgf_workers(pool.feed,count,[](void* tree)->void*{return static_cast<gumbel::Tree*>(tree)->state.get();}))throw std::runtime_error(gumbel::error);
 pool.host_workers=count;return 1;
}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
extern "C" HX_API int hxgm_cancel(void* p){try{auto& pool=owner::caller(p);pool.stop();if(pool.proof_step)pool.proof_step(pool.proof_owner);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
extern "C" HX_API int hxgm_cancel_game(void* p,int i){try{auto& pool=owner::caller(p);if(i<0 || i>=int(pool.games.size()))return 0;pool.games[i]->stop();pool.stopped=std::all_of(pool.games.begin(),pool.games.end(),[](const auto& o){return o->stopped;});if(pool.proof_step)pool.proof_step(pool.proof_owner);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
extern "C" HX_API int hxgm_clock(void* p,double ms){try{owner::caller(p).clock(ms);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
extern "C" HX_API int hxgm_retarget(void* p,int i,const int64_t* history,int count,uint64_t work,double ms){try{owner::caller(p).retarget(i,history,count,work,ms);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
extern "C" HX_API int hxgm_admit(void* p){try{return owner::caller(p).admit();}catch(const std::exception& e){gumbel::error=e.what();return -1;}}
extern "C" HX_API int hxgm_done(void* p){return static_cast<owner::Pool*>(p)->stopped;}
extern "C" HX_API void* hxgm_feed(void* p){return static_cast<owner::Pool*>(p)->feed;}
extern "C" HX_API void* hxgm_owner(void* p,int i){auto& pool=*static_cast<owner::Pool*>(p);return i<0 || i>=int(pool.games.size())?nullptr:pool.games[i].get();}
extern "C" HX_API const char* hxgm_model(void* p){return static_cast<owner::Pool*>(p)->model.c_str();}
extern "C" HX_API int hxgm_install(void* p,const uint64_t* ids,int count,const int64_t* offsets,const int64_t* actions,const double* logits,const double* values){try{return owner::caller(p).install(ids,count,offsets,actions,logits,values);}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
extern "C" HX_API void hxgm_stats(void* p,uint64_t* out){auto& pool=*static_cast<owner::Pool*>(p);uint64_t active=0,failed=0;for(size_t i=0;i<pool.games.size();++i){active+=!pool.games[i]->stopped;failed+=pool.failed[i];}std::array<uint64_t,5> stats{pool.steps,uint64_t(pool.games.size()),active,failed,pool.retargets};std::copy(stats.begin(),stats.end(),out);}
extern "C" HX_API int hxgm_history(void* p,int i,int64_t* out){auto& pool=*static_cast<owner::Pool*>(p);if(i<0 || i>=int(pool.games.size()))return -1;auto& h=pool.games[i]->focus;if(out)for(size_t j=0;j<h.size();++j){out[2*j]=h[j].q;out[2*j+1]=h[j].r;}return int(h.size());}
// Feed pruning runs between graph phases, before encoded snapshots are taken.
extern "C" HX_API int hxg_retire(void* p,int id){try{
 auto& tree=*static_cast<gumbel::Tree*>(p);auto found=tree.requests.find(id);
 if(found==tree.requests.end())throw std::runtime_error("Unknown queued neural request");
 tree.proof_root();if(!tree.proof_closed(found->second))return 0;
 tree.requests.erase(found);return 1;
}catch(const std::exception& e){gumbel::error=e.what();return -1;}}
extern "C" HX_API int hxgm_record_history(void* p,int game,int record,int64_t* out){auto& pool=*static_cast<owner::Pool*>(p);if(game<0 || game>=int(pool.games.size()) || record<0 || record>=int(pool.games[game]->records.size()))return -1;auto& h=pool.games[game]->records[record].history;if(out)for(size_t j=0;j<h.size();++j){out[2*j]=h[j].q;out[2*j+1]=h[j].r;}return int(h.size());}

extern "C" HX_API int hxgo_history(void* p,int64_t* out){auto& h=static_cast<owner::Owner*>(p)->focus;if(out)for(size_t i=0;i<h.size();++i){out[2*i]=h[i].q;out[2*i+1]=h[i].r;}return int(h.size());}

#ifndef __EMSCRIPTEN__
#include "gumbel_broker.hpp"
#endif
extern "C" HX_API int hxgo_record_history(void* p,int record,int64_t* out){auto& o=*static_cast<owner::Owner*>(p);if(record<0 || record>=int(o.records.size()))return -1;auto& h=o.records[record].history;if(out)for(size_t i=0;i<h.size();++i){out[2*i]=h[i].q;out[2*i+1]=h[i].r;}return int(h.size());}
