#include "hexo.hpp"
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

namespace gumbel { extern thread_local std::string error; }
extern "C" {
int hxg_next(void*);
int hxg_history(void*,int,int64_t*);
int hxg_fulfill(void*,int,const int64_t*,const double*,const double*,int);
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
struct Cached { Prediction prediction;std::list<Key>::iterator recency; };
struct Root { Key key;double value=0;bool known=false; };
struct Feed {
 size_t capacity;
 uint64_t next=1;
 std::map<uint64_t,Task> tasks;
 std::unordered_map<Key,uint64_t,Hash> pending;
 std::deque<uint64_t> ready;
 std::unordered_map<Key,Cached,Hash> cache;
 std::list<Key> recency;
 std::unordered_map<void*,Root> roots;
 int64_t new_rows=0,joins=0,hits=0,installed=0;
 bool profile=false;
 uint64_t select_ns=0,identity_ns=0,cached_ns=0,install_ns=0;
 explicit Feed(int limit):capacity(limit) { if(limit<0)throw std::runtime_error("Negative feed cache capacity"); }
 Cached* get(const Key& key) {
  auto found=cache.find(key);if(found==cache.end())return nullptr;
  recency.splice(recency.begin(),recency,found->second.recency);return &found->second;
 }
 void root_value(void* tree,const Key& key,const Prediction& p) {
  if(auto root=roots.find(tree);root!=roots.end() && root->second.key==key){
   root->second.value=p.values.front();root->second.known=true;
  }
 }
 void fulfill(Subscriber subscriber,const Key& key,const Prediction& p) {
  if(!hxg_fulfill(subscriber.tree,subscriber.request,p.actions.data(),p.logits.data(),p.values.data(),int(p.values.size())))
   throw std::runtime_error(gumbel::error);
  root_value(subscriber.tree,key,p);++installed;
 }
 void put(Key key,Prediction p) {
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
  if(auto p=get(root.key)){root.value=p->prediction.values.front();root.known=true;}
  roots[tree]=std::move(root);
 }
 // Drain to the existing tree barrier. Splitting a visit layer here changes interior selection.
 int gather(void* tree,int64_t* out) {
  if(!roots.contains(tree) || !out)throw std::runtime_error("Invalid feed gather");
  int added=0;auto initial_hits=hits,initial_joins=joins;bool progress=false;
  int status=0;
  while(true){
   int request;
   {Timer clock(profile?&select_ns:nullptr);request=hxg_next(tree);}
   if(request==-2)throw std::runtime_error(gumbel::error);
   if(request==-1){progress=true;continue;}
   if(request<=0){status=request;break;}
   std::vector<int64_t> history;Key key;Cached* cached;
   {Timer clock(profile?&identity_ns:nullptr);
    int n=hxg_history(tree,request,nullptr);history.resize(2*n);hxg_history(tree,request,history.data());
    key=context(history.data(),n);cached=get(key);
   }
   if(cached){Timer clock(profile?&cached_ns:nullptr);fulfill({tree,request},key,cached->prediction);++hits;progress=true;continue;}
   if(auto existing=pending.find(key);existing!=pending.end()){
    tasks.at(existing->second).subscribers.push_back({tree,request});++joins;
   }else{
    if(next==0)throw std::runtime_error("Feed task identity exhausted");
    uint64_t id=next++;pending.emplace(key,id);
    tasks.emplace(id,Task{std::move(key),std::move(history),{{tree,request}}});ready.push_back(id);
    ++added;++new_rows;
   }
  }
  out[0]=added;out[1]=hits-initial_hits;out[2]=joins-initial_joins;out[3]=progress;
  return status;
 }
 std::vector<uint64_t> batch(int limit)const {
  std::vector<uint64_t> ids;
  for(auto id:ready)if(auto t=tasks.find(id);t!=tasks.end() && !t->second.submitted){
   ids.push_back(id);if(int(ids.size())==limit)break;
  }
  return ids;
 }
 void detach(void* tree) {
  // Detach before freeing or advancing a cancelled tree. Submitted rows retain no tree ownership.
  if(roots.erase(tree))hxg_cancel(tree);
  for(auto it=tasks.begin();it!=tasks.end();){
   auto& t=it->second;
   std::erase_if(t.subscribers,[&](Subscriber s){return s.tree==tree;});
   if(t.subscribers.empty()&&!t.submitted){pending.erase(t.key);it=tasks.erase(it);}else ++it;
  }
 }
};
}

extern "C" {
HX_API void* hxgf_new(int capacity){try{return new feeding::Feed(capacity);}catch(const std::exception& e){gumbel::error=e.what();return nullptr;}}
// Trees must be detached while still alive. No tree is dereferenced by the destructor.
HX_API void hxgf_free(void* p){delete static_cast<feeding::Feed*>(p);}
HX_API int hxgf_begin(void* p,void* tree,const int64_t* history,int n){try{static_cast<feeding::Feed*>(p)->begin(tree,history,n);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxgf_seed(void* p,void* tree,const int64_t* actions,const double* logits,const double* values,int count){try{
 auto& f=*static_cast<feeding::Feed*>(p);auto root=f.roots.find(tree);
 if(root==f.roots.end() || count<1 || !actions || !logits || !values)throw std::runtime_error("Invalid feed root prediction");
 for(int i=0;i<count;++i)if(!std::isfinite(logits[i]) || !std::isfinite(values[i]) || std::abs(values[i])>1)
  throw std::runtime_error("Invalid feed root prediction");
 feeding::Prediction prediction{{actions,actions+2*count},{logits,logits+count},{values,values+count}};
 f.root_value(tree,root->second.key,prediction);f.put(root->second.key,std::move(prediction));return 1;
}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxgf_gather(void* p,void* tree,int64_t* out){try{return static_cast<feeding::Feed*>(p)->gather(tree,out);}catch(const std::exception& e){gumbel::error=e.what();return -2;}}
// Layout query and take are serialized on the graph-owner thread; no tree is re-rooted between them.
HX_API int hxgf_layout(void* p,int limit,int64_t* out){try{
 if(limit<1 || !out)throw std::runtime_error("Invalid feed batch limit");
 auto& f=*static_cast<feeding::Feed*>(p);auto ids=f.batch(limit);int64_t size=0;
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
  size+=task.history.size()/2;task.submitted=true;
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
 for(int i=0;i<count;++i){
  auto found=f.tasks.find(ids[i]);auto& task=found->second;int64_t first=offsets[i],last=offsets[i+1];
  if(last>first){
   feeding::Prediction prediction{{actions+2*first,actions+2*last},{logits+first,logits+last},{values+first,values+last}};
   for(auto subscriber:task.subscribers)f.fulfill(subscriber,task.key,prediction);
   f.put(task.key,std::move(prediction));
  }
  f.pending.erase(task.key);f.tasks.erase(found);
 }
 return output;
}catch(const std::exception& e){gumbel::error=e.what();return -1;}}
HX_API void hxgf_detach(void* p,void* tree){static_cast<feeding::Feed*>(p)->detach(tree);}
HX_API int hxgf_root_value(void* p,void* tree,double* value){
 auto& f=*static_cast<feeding::Feed*>(p);auto root=f.roots.find(tree);
 if(root!=f.roots.end() && !root->second.known)if(auto cached=f.get(root->second.key)){
  root->second.value=cached->prediction.values.front();root->second.known=true;
 }
 if(root==f.roots.end()||!root->second.known)return 0;
 *value=root->second.value;return 1;
}
HX_API void hxgf_stats(void* p,int64_t* out){
 auto& f=*static_cast<feeding::Feed*>(p);int64_t subscribers=0;for(auto& [id,t]:f.tasks)subscribers+=t.subscribers.size();
 out[0]=f.new_rows;out[1]=f.joins;out[2]=f.hits;out[3]=f.installed;out[4]=int64_t(f.tasks.size());out[5]=subscribers;
}
HX_API void hxgf_profile(void* p,int enabled){static_cast<feeding::Feed*>(p)->profile=enabled!=0;}
HX_API void hxgf_times(void* p,uint64_t* out){auto& f=*static_cast<feeding::Feed*>(p);out[0]=f.select_ns;out[1]=f.identity_ns;out[2]=f.cached_ns;out[3]=f.install_ns;}
}

// Cancelled subscribers are detached before this call. Caller has fenced any GPU reads.
extern "C" HX_API int hxgf_abandon_all(void* p){try{
 auto& f=*static_cast<feeding::Feed*>(p);for(auto& [id,t]:f.tasks)if(!t.subscribers.empty())throw std::runtime_error("Detach all subscribers before abandoning tasks");
 f.tasks.clear();f.pending.clear();f.ready.clear();return 1;
}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
