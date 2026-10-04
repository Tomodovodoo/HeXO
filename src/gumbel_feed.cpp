#include "hexo.hpp"
#include "gumbel_parallel.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <deque>
#include <list>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <atomic>

namespace gumbel { extern thread_local std::string error; }
extern "C" {
int hxg_next(void*);
int hxg_history(void*,int,int64_t*);
int hxg_fulfill(void*,int,const int64_t*,const double*,const double*,int);
int hxg_retire(void*,int);
void hxg_cancel(void*);
}

namespace feeding {
struct Timer {
 uint64_t* total;
 std::chrono::steady_clock::time_point start;
 explicit Timer(uint64_t* target):total(target){if(total)start=std::chrono::steady_clock::now();}
 ~Timer(){if(total)*total+=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count());}
};
using Key=std::vector<int64_t>;
struct Hash {
 size_t operator()(const Key& key)const {
  uint64_t h=0xcbf29ce484222325ULL;
  for(auto v:key){h^=uint64_t(v);h*=0x100000001b3ULL;h^=h>>32;}
  return size_t(h);
 }
};
// Hashing only indexes the full key. Equality compares every coordinate and context field.
// Rule-equivalent boards with a different first stone or prior turn do not share predictions.
Key context(const int64_t* history,int n) {
 std::vector<std::array<int64_t,3>> colored;
 colored.reserve(n);
 for(int i=0;i<n;++i)colored.push_back({((i+1)/2)%2,history[2*i],history[2*i+1]});
 std::sort(colored.begin(),colored.end());
 Key key;key.reserve(3*n+8);key.push_back(n);
 for(auto row:colored)key.insert(key.end(),row.begin(),row.end());
 int start=n==0||n%2?n:n-1;
 if(start<n){key.push_back(history[2*start]);key.push_back(history[2*start+1]);}
 std::vector<std::array<int64_t,2>> previous;
 for(int i=std::max(0,start-2);i<start;++i)previous.push_back({history[2*i],history[2*i+1]});
 std::sort(previous.begin(),previous.end());
 for(auto row:previous)key.insert(key.end(),row.begin(),row.end());
 return key;
}
struct Subscriber { void* tree;int request; };
struct Task { Key key;std::vector<int64_t> history;std::vector<Subscriber> subscribers;bool submitted=false; };
struct Prediction {
 std::vector<int64_t> actions;
 std::vector<double> logits,values;
};
struct Cached { std::shared_ptr<const Prediction> prediction;std::list<Key>::iterator recency; };
struct Root { Key key;double value=0;bool known=false; };
struct Feed {
 std::mutex mutex;
 std::unique_ptr<Workers> workers=std::make_unique<Workers>();
 void* (*group)(void*)=nullptr;
 size_t capacity;
 uint64_t next=1;
 std::map<uint64_t,Task> tasks;
 std::unordered_map<Key,uint64_t,Hash> pending;
 std::deque<uint64_t> ready;
 std::unordered_map<Key,Cached,Hash> cache;
 std::list<Key> recency;
 std::unordered_map<void*,Root> roots;
 int64_t new_rows=0,joins=0,hits=0,installed=0;
 uint64_t proof_requests=0,proof_rows=0,prune_ns=0;
 std::atomic<int64_t> queued=0;
 bool profile=false;
 uint64_t select_ns=0,identity_ns=0,cached_ns=0,install_ns=0;
 explicit Feed(int limit):capacity(limit) { if(limit<0)throw std::runtime_error("Negative feed cache capacity"); }
 std::shared_ptr<const Prediction> get(const Key& key) {
  auto found=cache.find(key);if(found==cache.end())return {};
  recency.splice(recency.begin(),recency,found->second.recency);return found->second.prediction;
 }
 void root_value(void* tree,const Key& key,const Prediction& p) {
  if(auto root=roots.find(tree);root!=roots.end() && root->second.key==key){
   root->second.value=p.values.front();root->second.known=true;
  }
 }
 void fulfill(Subscriber subscriber,const Key& key,const Prediction& p) {
  if(!hxg_fulfill(subscriber.tree,subscriber.request,p.actions.data(),p.logits.data(),p.values.data(),int(p.values.size())))
   throw std::runtime_error(gumbel::error);
  std::lock_guard lock(mutex);
  root_value(subscriber.tree,key,p);++installed;
 }
 // Cache payloads are immutable; a producer keeps its snapshot through a
 // concurrent eviction without copying every legal action and prediction.
 void put(Key key,std::shared_ptr<const Prediction> p) {
  if(!capacity)return;
  if(auto found=cache.find(key);found!=cache.end()){
   found->second.prediction=std::move(p);recency.splice(recency.begin(),recency,found->second.recency);return;
  }
  recency.push_front(key);cache.emplace(std::move(key),Cached{std::move(p),recency.begin()});
  while(cache.size()>capacity){cache.erase(recency.back());recency.pop_back();}
 }
 void begin(void* tree,const int64_t* history,int n) {
  if(!tree || n<0 || (n&&!history))throw std::runtime_error("Invalid feed root");
  Root root{context(history,n)};
  std::lock_guard lock(mutex);
  if(auto p=get(root.key)){root.value=p->values.front();root.known=true;}
  roots[tree]=std::move(root);
 }
 // Drain to the existing tree barrier. Splitting a visit layer here changes interior selection.
 int gather(void* tree,int64_t* out) {
  {std::lock_guard lock(mutex);if(!roots.contains(tree) || !out)throw std::runtime_error("Invalid feed gather");}
  int added=0,joined=0,cached_hits=0;bool progress=false;
  uint64_t selection=0,identity=0,cached_work=0;
  int status=0;
  while(true){
   int request;
   {Timer clock(profile?&selection:nullptr);request=hxg_next(tree);}
   if(request==-2)throw std::runtime_error(gumbel::error);
   if(request==-1){progress=true;continue;}
   if(request<=0){status=request;break;}
   std::vector<int64_t> history;Key key;std::shared_ptr<const Prediction> cached;
   {Timer clock(profile?&identity:nullptr);
    int n=hxg_history(tree,request,nullptr);history.resize(2*n);hxg_history(tree,request,history.data());
    key=context(history.data(),n);
    std::lock_guard lock(mutex);cached=get(key);
    if(!cached){
     if(auto existing=pending.find(key);existing!=pending.end()){
      tasks.at(existing->second).subscribers.push_back({tree,request});++joins;++joined;
     }else{
      if(next==0)throw std::runtime_error("Feed task identity exhausted");
      uint64_t id=next++;pending.emplace(key,id);
      tasks.emplace(id,Task{std::move(key),std::move(history),{{tree,request}}});ready.push_back(id);
      ++added;++new_rows;++queued;
     }
    }else {++hits;++cached_hits;}
   }
   if(cached){Timer clock(profile?&cached_work:nullptr);fulfill({tree,request},key,*cached);progress=true;}
  }
  {std::lock_guard lock(mutex);select_ns+=selection;identity_ns+=identity;cached_ns+=cached_work;}
  out[0]=added;out[1]=cached_hits;out[2]=joined;out[3]=progress;
  return status;
 }
 std::vector<uint64_t> batch(int limit)const {
  std::vector<uint64_t> ids;
  for(auto id:ready)if(auto t=tasks.find(id);t!=tasks.end() && !t->second.submitted){
   ids.push_back(id);if(int(ids.size())==limit)break;
  }
  return ids;
 }
 int prune(){
  Timer clock(&prune_ns);int retired=0;
  for(auto id:ready){
   auto found=tasks.find(id);if(found==tasks.end() || found->second.submitted)continue;
   auto& task=found->second;
   std::erase_if(task.subscribers,[&](Subscriber subscriber){
    int result=hxg_retire(subscriber.tree,subscriber.request);
    if(result<0)throw std::runtime_error(gumbel::error);
    if(result){++retired;++proof_requests;}return result!=0;
   });
   if(task.subscribers.empty()){forget(task.key,id);tasks.erase(found);--queued;++proof_rows;}
  }
  // A cancelled/settled row leaves no queued identity to inspect next time.
  std::erase_if(ready,[&](uint64_t id){auto it=tasks.find(id);return it==tasks.end() || it->second.submitted;});
  return retired;
 }
 int retire(uint64_t id){
  auto found=tasks.find(id);if(found==tasks.end() || !found->second.submitted)throw std::runtime_error("Unknown snapshot task");
  auto& subscribers=found->second.subscribers;
  std::erase_if(subscribers,[&](Subscriber subscriber){
   int result=hxg_retire(subscriber.tree,subscriber.request);
   if(result<0)throw std::runtime_error(gumbel::error);if(result)++proof_requests;return result!=0;
  });
  // Keep the submitted ID/snapshot, but do not let a new generation subscribe
  // after an empty completion has already been queued for this task.
  if(subscribers.empty())forget(found->second.key,id);
  return int(subscribers.size());
 }
 void forget(const Key& key,uint64_t id){auto it=pending.find(key);if(it!=pending.end() && it->second==id)pending.erase(it);}
 void detach(void* tree) {
  // Detach before freeing or advancing a cancelled tree. Submitted rows retain no tree ownership.
  bool attached;{std::lock_guard lock(mutex);attached=roots.erase(tree)!=0;}
  if(attached)hxg_cancel(tree);
  std::lock_guard lock(mutex);
  for(auto it=tasks.begin();it!=tasks.end();){
   auto& t=it->second;
   std::erase_if(t.subscribers,[&](Subscriber s){return s.tree==tree;});
   if(t.subscribers.empty())forget(t.key,it->first);
   if(t.subscribers.empty()&&!t.submitted){--queued;it=tasks.erase(it);}else ++it;
  }
 }
};
}

extern "C" {
HX_API void* hxgf_new(int capacity){try{return new feeding::Feed(capacity);}catch(const std::exception& e){gumbel::error=e.what();return nullptr;}}
// Trees must be detached while still alive. No tree is dereferenced by the destructor.
HX_API void hxgf_free(void* p){delete static_cast<feeding::Feed*>(p);}
HX_API int hxgf_workers(void* p,int count,void* (*group)(void*)){try{
 auto& f=*static_cast<feeding::Feed*>(p);auto workers=std::make_unique<feeding::Workers>(count);
 f.workers=std::move(workers);f.group=group;return 1;
}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxgf_parallel(void* p,int count,void (*work)(void*,int),void* data,bool (*admit)(void*,int)){try{
 static_cast<feeding::Feed*>(p)->workers->run(count,work,data,admit);return 1;
}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxgf_begin(void* p,void* tree,const int64_t* history,int n){try{static_cast<feeding::Feed*>(p)->begin(tree,history,n);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxgf_seed(void* p,void* tree,const int64_t* actions,const double* logits,const double* values,int count){try{
 auto& f=*static_cast<feeding::Feed*>(p);std::lock_guard lock(f.mutex);auto root=f.roots.find(tree);
 if(root==f.roots.end() || count<1 || !actions || !logits || !values)throw std::runtime_error("Invalid feed root prediction");
 for(int i=0;i<count;++i)if(!std::isfinite(logits[i]) || !std::isfinite(values[i]) || std::abs(values[i])>1)
  throw std::runtime_error("Invalid feed root prediction");
 auto prediction=std::make_shared<feeding::Prediction>(feeding::Prediction{{actions,actions+2*count},{logits,logits+count},{values,values+count}});
 f.root_value(tree,root->second.key,*prediction);f.put(root->second.key,std::move(prediction));return 1;
}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxgf_gather(void* p,void* tree,int64_t* out){try{return static_cast<feeding::Feed*>(p)->gather(tree,out);}catch(const std::exception& e){gumbel::error=e.what();return -2;}}
// Layout query and take are serialized on the graph-owner thread; no tree is re-rooted between them.
HX_API int hxgf_layout(void* p,int limit,int64_t* out){try{
 if(limit<1 || !out)throw std::runtime_error("Invalid feed batch limit");
 auto& f=*static_cast<feeding::Feed*>(p);f.prune();auto ids=f.batch(limit);int64_t size=0;
 for(auto id:ids)size+=f.tasks.at(id).history.size()/2;
 out[0]=int64_t(ids.size());out[1]=size;return 1;
}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxgf_take(void* p,int count,uint64_t* ids,void** trees,int* requests,int64_t* offsets,int64_t* history,int64_t capacity){try{
 bool handles_only=!offsets && !history && capacity==0;
 if(count<1 || !ids || !trees || !requests || (!handles_only&&!offsets) || capacity<0 || (capacity&&!history))throw std::runtime_error("Invalid feed batch buffers");
 auto& f=*static_cast<feeding::Feed*>(p);auto batch=f.batch(count);
 if(int(batch.size())!=count)throw std::runtime_error("Changed feed batch layout");
 int64_t size=0;for(auto id:batch)size+=f.tasks.at(id).history.size()/2;
 if(!handles_only && size>capacity)throw std::runtime_error("Feed history buffer too small");
 size=0;
 for(int i=0;i<count;++i){
  auto& task=f.tasks.at(batch[i]);auto subscriber=task.subscribers.front();
  ids[i]=batch[i];trees[i]=subscriber.tree;requests[i]=subscriber.request;
  if(!handles_only){offsets[i]=size;if(!task.history.empty())std::copy(task.history.begin(),task.history.end(),history+2*size);}
  size+=task.history.size()/2;task.submitted=true;--f.queued;
 }
 if(!handles_only)offsets[count]=size;
 while(!f.ready.empty()){
  auto it=f.tasks.find(f.ready.front());if(it!=f.tasks.end()&&!it->second.submitted)break;f.ready.pop_front();
 }
 return 1;
}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxgf_install(void* p,const uint64_t* ids,int count,const int64_t* offsets,const int64_t* actions,
 const double* logits,const double* values,void** stopped,int stopped_capacity){try{
 if(count<1 || !ids || !offsets || offsets[0]!=0 || stopped_capacity<0 || (stopped_capacity&&!stopped))throw std::runtime_error("Invalid feed result buffers");
 auto& f=*static_cast<feeding::Feed*>(p);std::set<void*> failures;std::set<uint64_t> seen;
 feeding::Timer clock(f.profile?&f.install_ns:nullptr);
 for(int i=0;i<count;++i){
  auto found=f.tasks.find(ids[i]);
  if(found==f.tasks.end() || !found->second.submitted || !seen.insert(ids[i]).second || offsets[i+1]<offsets[i])
   throw std::runtime_error("Invalid feed completion");
  if(offsets[i+1]==offsets[i])for(auto s:found->second.subscribers)failures.insert(s.tree);
 }
 if(offsets[count] && (!actions||!logits||!values))throw std::runtime_error("Missing feed predictions");
 for(int64_t i=0;i<offsets[count];++i)
  if(!std::isfinite(logits[i]) || !std::isfinite(values[i]) || std::abs(values[i])>1)
   throw std::runtime_error("Invalid feed prediction");
 if(int(failures.size())>stopped_capacity)throw std::runtime_error("Feed stopped-tree buffer too small");
 int output=0;for(auto tree:failures){stopped[output++]=tree;f.detach(tree);}
 struct Completion {feeding::Subscriber subscriber;const feeding::Key* key;std::shared_ptr<const feeding::Prediction> prediction;};
 std::vector<std::vector<Completion>> groups;std::map<void*,size_t> by_game;
 std::vector<std::shared_ptr<const feeding::Prediction>> predictions(count);
 for(int i=0;i<count;++i){
  auto found=f.tasks.find(ids[i]);auto& task=found->second;int64_t first=offsets[i],last=offsets[i+1];
  if(last>first){
   auto prediction=std::make_shared<feeding::Prediction>(feeding::Prediction{{actions+2*first,actions+2*last},{logits+first,logits+last},{values+first,values+last}});
   if(f.workers->size()==1){
    for(auto subscriber:task.subscribers)f.fulfill(subscriber,task.key,*prediction);
    f.put(task.key,std::move(prediction));
   }else{
    predictions[i]=prediction;
    for(auto subscriber:task.subscribers){
     void* identity=f.group?f.group(subscriber.tree):subscriber.tree;
     auto [entry,inserted]=by_game.emplace(identity,groups.size());if(inserted)groups.emplace_back();
     groups[entry->second].push_back({subscriber,&task.key,prediction});
    }
   }
  }
 }
 // Each group owns one graph. Shared cache/task maps are changed only after
 // all graph workers return; immutable predictions may serve several games.
 struct Work {feeding::Feed* feed;std::vector<std::vector<Completion>>* groups;} work{&f,&groups};
 f.workers->run(int(groups.size()),[](void* data,int index){auto& w=*static_cast<Work*>(data);
  for(auto& result:(*w.groups)[index])w.feed->fulfill(result.subscriber,*result.key,*result.prediction);
 },&work);
 for(int i=0;i<count;++i){
  auto found=f.tasks.find(ids[i]);auto& task=found->second;
  if(predictions[i])f.put(task.key,std::move(predictions[i]));
  f.forget(task.key,ids[i]);f.tasks.erase(found);
 }
 return output;
}catch(const std::exception& e){gumbel::error=e.what();return -1;}}
HX_API void hxgf_detach(void* p,void* tree){static_cast<feeding::Feed*>(p)->detach(tree);}
HX_API int hxgf_root_value(void* p,void* tree,double* value){
 auto& f=*static_cast<feeding::Feed*>(p);std::lock_guard lock(f.mutex);auto root=f.roots.find(tree);
 if(root!=f.roots.end() && !root->second.known)if(auto cached=f.get(root->second.key)){
  root->second.value=cached->values.front();root->second.known=true;
 }
 if(root==f.roots.end()||!root->second.known)return 0;
 *value=root->second.value;return 1;
}
HX_API void hxgf_stats(void* p,int64_t* out){
 auto& f=*static_cast<feeding::Feed*>(p);int64_t subscribers=0;for(auto& [id,t]:f.tasks)subscribers+=t.subscribers.size();
 out[0]=f.new_rows;out[1]=f.joins;out[2]=f.hits;out[3]=f.installed;out[4]=int64_t(f.tasks.size());out[5]=subscribers;
}
// Distinct neural rows ready to launch, excluding submitted work and subscribers.
HX_API int64_t hxgf_queued(void* p){return static_cast<feeding::Feed*>(p)->queued;}
HX_API int hxgf_prune(void* p){try{return static_cast<feeding::Feed*>(p)->prune();}catch(const std::exception& e){gumbel::error=e.what();return -1;}}
HX_API int hxgf_retire(void* p,uint64_t id){try{return static_cast<feeding::Feed*>(p)->retire(id);}catch(const std::exception& e){gumbel::error=e.what();return -1;}}
HX_API const int64_t* hxgf_key(void* p,uint64_t id,int64_t* count){try{
 auto& task=static_cast<feeding::Feed*>(p)->tasks.at(id);*count=int64_t(task.key.size());return task.key.data();
}catch(const std::exception& e){gumbel::error=e.what();return nullptr;}}
HX_API void hxgf_pruning(void* p,uint64_t* out){auto& f=*static_cast<feeding::Feed*>(p);out[0]=f.proof_requests;out[1]=f.proof_rows;out[2]=f.prune_ns;}
HX_API void hxgf_profile(void* p,int enabled){static_cast<feeding::Feed*>(p)->profile=enabled!=0;}
HX_API void hxgf_times(void* p,uint64_t* out){auto& f=*static_cast<feeding::Feed*>(p);out[0]=f.select_ns;out[1]=f.identity_ns;out[2]=f.cached_ns;out[3]=f.install_ns;}
}

// Cancelled subscribers are detached before this call. Caller has fenced any GPU reads.
extern "C" HX_API int hxgf_abandon_all(void* p){try{
 auto& f=*static_cast<feeding::Feed*>(p);for(auto& [id,t]:f.tasks)if(!t.subscribers.empty())throw std::runtime_error("Detach all subscribers before abandoning tasks");
 f.tasks.clear();f.pending.clear();f.ready.clear();f.queued=0;return 1;
}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
