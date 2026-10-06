// One graph writer consumes proof completions; workers only own immutable jobs.
#include "gumbel_owner.cpp"
#include <thread>
#include <mutex>
#include <condition_variable>
#include <bitset>
#include <atomic>
namespace proving {
using Clock=std::chrono::steady_clock;
using gumbel::Node;using gumbel::Tree;using gumbel::Key;using gumbel::KeyHash;
struct API {
 using New=void*(*)();using Free=void(*)(void*);using Query=void*(*)(void*,const char*);
 using Info=bool(*)(void*,uint64_t*);using Moves=int(*)(void*,int64_t*,size_t);using JSON=void*(*)(void*);
 using Prepare=uint64_t(*)();using Cancel=bool(*)(uint64_t);using Release=void(*)(uint64_t);using Busy=bool(*)(void*);
  New make=nullptr;Free worker_free=nullptr,answer_free=nullptr,buffer_free=nullptr;Query query=nullptr;Info info=nullptr;Moves moves=nullptr;JSON json=nullptr;Prepare prepare=nullptr;Cancel cancel=nullptr;Release release=nullptr;Busy busy=nullptr;
  explicit API(const uint64_t* f) {
   if(!f)return;
   make=reinterpret_cast<New>(f[0]);worker_free=reinterpret_cast<Free>(f[1]);query=reinterpret_cast<Query>(f[2]);info=reinterpret_cast<Info>(f[3]);moves=reinterpret_cast<Moves>(f[4]);json=reinterpret_cast<JSON>(f[5]);answer_free=reinterpret_cast<Free>(f[6]);buffer_free=reinterpret_cast<Free>(f[7]);prepare=reinterpret_cast<Prepare>(f[8]);cancel=reinterpret_cast<Cancel>(f[9]);release=reinterpret_cast<Release>(f[10]);busy=reinterpret_cast<Busy>(f[11]);
  for(int i=0;i<12;++i)if(!f[i])throw std::runtime_error("Missing native proof callback");
 }
};
struct Dependency {Key key;int distance;bool operator==(const Dependency&)const=default;};
using Premises=std::bitset<256>;
struct Task {
  Key key;std::vector<Cell> history;std::weak_ptr<Node> node;uint64_t generation=0,born=0,facts=0;double impact=0,cost=.2,change=0,observed_change=0,attempted_change=0;
 int side=0,closed=0,worker=-1;std::array<unsigned,2> attempts{};bool flight=false,scoped=false;Clock::time_point ready{};
 std::vector<Dependency> scope;
};
struct Fact {std::vector<Cell> history;int winner,distance,slot;};
struct Frontier {
 struct Effort {uint64_t fresh=0,queries=0,missing=0;};
 std::map<uint64_t,Effort> effort;
 std::unordered_map<Key,std::shared_ptr<Task>,KeyHash> tasks;std::unordered_map<Key,Fact,KeyHash> facts;
 std::deque<Key> fact_order;uint64_t generation=1,revision=0,next=0,offers=0;size_t capacity;
 Premises occupied;std::unordered_map<Cell,std::array<Premises,2>,CellHash> members;
 uint64_t scope_checks=0,scope_changed=0,scope_unchanged=0,scope_ns=0;
 explicit Frontier(size_t cap):capacity(cap){}
 void forget(Key key){
  auto& fact=facts.at(key);
  for(size_t n=0;n<fact.history.size();++n){auto it=members.find(fact.history[n]);
   it->second[(n+1)/2%2].reset(fact.slot);if(it->second[0].none() && it->second[1].none())members.erase(it);}
  occupied.reset(fact.slot);facts.erase(key);
 }
 void remember(const std::vector<Cell>& history,const Node& node){
  if(node.exact_winner<0 || node.distance<1 || node.distance>10000)return;
  auto key=gumbel::keys(history).first;auto old=facts.find(key);
  if(old!=facts.end()){
   if(old->second.winner!=node.exact_winner)throw std::runtime_error("Conflicting exact frontier facts");
   if(old->second.distance<=node.distance)return;old->second.distance=node.distance;
  }else{
    if(facts.size()==occupied.size()){forget(fact_order.front());fact_order.pop_front();}
    size_t slot=0;while(occupied[slot])++slot;occupied.set(slot);
    facts.emplace(key,Fact{history,node.exact_winner,node.distance,int(slot)});fact_order.push_back(key);
    for(size_t n=0;n<history.size();++n)members[history[n]][(n+1)/2%2].set(slot);
   }
   ++revision;
 }
 void scope(Task& task,bool stamps){
  if(task.scoped && task.facts==revision)return;
  auto began=Clock::now();++scope_checks;auto mask=occupied;
  // A forward proof walk only adds colored stones. A fact missing any current
  // stone cannot occur below this task. Start with recent, usually rare cells.
  for(size_t n=task.history.size();n && mask.any();--n){auto it=members.find(task.history[n-1]);
   if(it==members.end()){mask.reset();break;}mask&=it->second[n/2%2];}
  std::vector<Dependency> selected;size_t words=0;
  if(mask.any())for(auto key:fact_order){auto& fact=facts.at(key);if(!mask[fact.slot])continue;
   size_t cost=fact.history.size()*2+3;if(words+cost>32768)continue;
   words+=cost;selected.push_back({key,fact.distance});}
  // Generalized stamps can apply beyond exact-board containment. Their opt-in
  // path keeps conservative global invalidation until it has a library epoch.
  if(!task.scoped || selected!=task.scope || stamps){task.closed=0;task.ready={};++scope_changed;}
  else ++scope_unchanged;
  task.scope=std::move(selected);task.facts=revision;task.scoped=true;
  scope_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-began).count();
 }
 void offer(const std::shared_ptr<Node>& node,const std::vector<Cell>& history,double impact,double change){
  ++offers;remember(history,*node);if(node->exact_winner>=0)return;
  auto key=gumbel::keys(history).first;auto found=tasks.find(key);
  if(found!=tasks.end()){
    auto& t=*found->second;t.impact=std::max(t.impact,impact);t.observed_change=change;
    const double moved=std::abs(change-t.attempted_change);t.change=std::max(t.change,moved);
    if(!t.flight){t.node=node;t.history=history;if(moved>.1)t.ready=Clock::time_point{};}return;
  }
  if(tasks.size()>=capacity){
   auto victim=tasks.end();double weakest=std::numeric_limits<double>::infinity();
   for(auto i=tasks.begin();i!=tasks.end();++i)if(!i->second->flight){auto n=i->second->node.lock();
    double score=n && n->exact_winner<0?i->second->impact/std::max(.05,i->second->cost):-1;
    if(score<weakest){weakest=score;victim=i;}}
   if(victim==tasks.end() || (weakest>=impact/.2 && offers%5!=0))return;tasks.erase(victim);
  }
  auto task=std::make_shared<Task>();task->key=key;task->node=node;task->history=history;task->generation=generation;
   task->impact=impact;task->change=task->observed_change=change;task->born=++next;task->facts=revision;tasks.emplace(key,std::move(task));
 }
};
struct Job {
 uint64_t id=0,token=0;size_t game=0;std::shared_ptr<Task> task;std::shared_ptr<Node> pin;
  std::vector<Cell> history;std::vector<Dependency> scope;std::string context,request,result,error;int side=0,worker=-1,preferred=-1,quantum=0;uint64_t generation=0,facts=0;
 Clock::time_point queued,started,deadline{};double elapsed=0,wait=0;bool cancelled=false,pruned=false,wake_owner=false;
 std::array<uint64_t,13> info{};std::array<int64_t,4> moves{};int move_count=0;std::vector<int64_t> neural;
};
struct Worker {void* native=nullptr;
#ifndef __EMSCRIPTEN__
 std::thread thread;
#endif
 std::shared_ptr<Job> active;uint64_t token=0;double service=0,idle=0;Clock::time_point idle_since=Clock::now();};
struct Loop {
 API::Moves endpoint=nullptr;int endpoint_limit=0;uint64_t endpoint_paths=0,endpoint_candidates=0,endpoint_rejected=0,endpoint_bytes=0,endpoint_ns=0;
 std::deque<std::string> endpoint_records;
 owner::Pool& pool;API api;std::vector<Frontier> frontiers;std::vector<std::unique_ptr<Worker>> workers;
 std::mutex mutex;
 // Idle workers sleep on wake and each queued job wakes one of them, so the
 // graph owner's admission never queues behind every idle worker. Drain waits on answered.
 std::condition_variable wake,answered;std::deque<std::shared_ptr<Job>> queued,done;
 std::shared_ptr<owner::Signal> listener;std::atomic<bool> completion_ready=false;size_t urgent=0;
  std::unordered_map<uint64_t,std::shared_ptr<Job>> live;size_t capacity,cursor=0;int slice,table,external_limit=64;bool stopping=false,enabled=true,stamps=false,external=false;
 uint64_t next=0,ticks=0,submitted=0,started=0,finished=0,installed=0,cancelled=0,pruned=0,unknown=0,fresh=0,missing_fresh=0,snapshot_ns=0,install_ns=0;
 uint64_t available_facts=0,sent_facts=0,empty_scope_jobs=0,quantum_ms=0;
 std::deque<std::string> records;
 std::string released_effort;
  Loop(owner::Pool& source,const uint64_t* functions,int count,int queue,int ms,int mb,int tasks,bool use_stamps):pool(source),api(functions),capacity(queue),slice(ms),table(mb),stamps(use_stamps),external(!functions){
  if(pool.proof_owner || count<1 || count>16 || queue<count || queue>128 || ms<1 || ms>1000 || mb<1 || mb>64 || tasks<8 || tasks>4096)throw std::runtime_error("Invalid native proof loop limits");
  frontiers.reserve(pool.games.size());for(size_t i=0;i<pool.games.size();++i)frontiers.emplace_back(tasks);
  try {
    for(int i=0;i<count;++i){auto worker=std::make_unique<Worker>();if(!external){worker->native=api.make();if(!worker->native)throw std::runtime_error("Could not create native proof worker");}workers.push_back(std::move(worker));}
#ifndef __EMSCRIPTEN__
    if(!external)for(size_t i=0;i<workers.size();++i)workers[i]->thread=std::thread([this,i]{run(i);});
#else
    if(!external)throw std::runtime_error("Browser proofs require external solver workers");
#endif
  }catch(...){shutdown();throw;}
  for(size_t i=0;i<pool.games.size();++i)bind(int(i));
  pool.proof_owner=this;pool.proof_step=[](void* p){static_cast<Loop*>(p)->step();};pool.proof_retarget=[](void* p,int i){static_cast<Loop*>(p)->retarget(i);};
  pool.proof_collect=[](void* p){return static_cast<Loop*>(p)->feedback();};
  pool.proof_bind=[](void* p,int i){static_cast<Loop*>(p)->bind(i);};
  pool.proof_ready=[](void* p){return static_cast<Loop*>(p)->completion_ready.load(std::memory_order_acquire);};
  pool.proof_listen=[](void* p,std::shared_ptr<owner::Signal> signal){auto& loop=*static_cast<Loop*>(p);std::lock_guard lock(loop.mutex);loop.listener=std::move(signal);};
 }
 ~Loop(){shutdown();for(auto& game:pool.games){game->game->evidence=nullptr;game->game->evidence_owner=nullptr;}pool.proof_owner=nullptr;pool.proof_step=nullptr;pool.proof_collect=nullptr;pool.proof_retarget=nullptr;pool.proof_bind=nullptr;pool.proof_ready=nullptr;pool.proof_listen=nullptr;}
 void bind(int game){auto& store=*pool.games[game]->game;store.evidence_owner=this;store.evidence=[](void* p,Tree& t,const gumbel::Path& path){static_cast<Loop*>(p)->observe(t,path);};}
 void completed(const std::shared_ptr<Job>& job){
  // Quiet UNKNOWN results still drain on the bounded owner tick. Waking for
  // every tiny query interrupts neural feeding without adding graph evidence.
  job->wake_owner=job->info[0] || !job->neural.empty() || !job->error.empty();done.push_back(job);
  if(job->wake_owner){++urgent;completion_ready.store(true,std::memory_order_release);}
 }
 void mark(Job& job,bool obsolete=false){
  if(!job.cancelled){job.cancelled=true;++cancelled;}if(obsolete && !job.pruned){job.pruned=true;++pruned;}
  for(auto& worker:workers)if(worker->active.get()==&job && worker->token)api.cancel(worker->token);
 }
 void shutdown(){
  {std::lock_guard lock(mutex);stopping=true;for(auto& [id,job]:live)mark(*job);}wake.notify_all();
   for(auto& worker:workers){
#ifndef __EMSCRIPTEN__
    if(worker->thread.joinable())worker->thread.join();
#endif
    if(worker->native){api.worker_free(worker->native);worker->native=nullptr;}}
 }
 void observe(Tree& t,const gumbel::Path& path){
  size_t i=0;while(i<pool.games.size() && pool.games[i]->game.get()!=t.state.get())++i;if(i==pool.games.size() || pool.games[i]->stopped)return;
  auto& f=frontiers[i];double relevance=1;for(auto& v:pool.games[i]->views)if(v.tree.get()==&t){relevance=v.relevance;break;}
  double share=1;for(auto [parent,index]:path.edges){const auto& edge=parent->edges[index];share*=std::max(.001,parent->prior(edge));}
  double forcing=1+std::min(size_t(8),path.own.size()+path.threats.size());
  f.offer(path.leaf->shared_from_this(),path.history,std::max(.0001,relevance*share)*forcing,std::abs(path.leaf->q-path.leaf->value));
  for(auto [parent,index]:path.edges)if(parent->exact_winner>=0 && parent->stones<=int(path.history.size()))f.remember(std::vector<Cell>(path.history.begin(),path.history.begin()+parent->stones),*parent);
 }
 std::string context(size_t i,const Task& task){
  auto start=Clock::now();auto& f=frontiers[i];std::string text="{\"history\":";
  auto cells=[&](const std::vector<Cell>& h){text+='[';for(size_t n=0;n<h.size();++n){if(n)text+=',';text+='[';text+=std::to_string(h[n].q);text+=',';text+=std::to_string(h[n].r);text+=']';}text+=']';};
  cells(task.history);text+=",\"known\":[";size_t count=0;
  for(auto dependency:task.scope){auto& fact=f.facts.at(dependency.key);if(count++)text+=',';
   text+="{\"history\":";cells(fact.history);text+=",\"winner\":"+std::to_string(fact.winner)+",\"plies\":"+std::to_string(fact.distance)+'}';
  }text+="]}";snapshot_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();return text;
 }
 struct Ready {
  struct Choice {std::shared_ptr<Task> task;double priority=0,age=0;bool picked=false;};
  Choice only;bool single=false;std::vector<Choice> choices;std::array<std::vector<size_t>,2> heaps;
  Choice& choice(size_t i){return single?only:choices[i];}
  double score(size_t i,bool explore)const{const auto& c=single?only:choices[i];return explore?c.age:c.priority;}
  auto lower(bool explore){return [this,explore](size_t a,size_t b){
   double x=score(a,explore),y=score(b,explore);return x==y?a>b:x<y;
  };}
  void build(){
   if(single)return;
   for(auto& heap:heaps){heap.reserve(choices.size());for(size_t i=0;i<choices.size();++i)heap.push_back(i);}
   std::make_heap(heaps[0].begin(),heaps[0].end(),lower(false));
   std::make_heap(heaps[1].begin(),heaps[1].end(),lower(true));
  }
  size_t best(bool explore){
   if(single)return only.task && !only.picked?0:SIZE_MAX;
   auto& heap=heaps[explore];while(!heap.empty() && choices[heap.front()].picked){
    std::pop_heap(heap.begin(),heap.end(),lower(explore));heap.pop_back();
   }return heap.empty()?SIZE_MAX:heap.front();
  }
 };
 std::vector<Ready> prepare(size_t available,Clock::time_point& retry){
  retry={};
  std::vector<Ready> ready(frontiers.size());auto now=Clock::now();bool explore=submitted%5==4;
  for(size_t n=0;n<frontiers.size();++n){size_t i=(cursor+n)%frontiers.size();auto& o=*pool.games[i];auto& f=frontiers[i];if(o.stopped)continue;
   auto& candidates=ready[i];candidates.single=available==1;if(!candidates.single)candidates.choices.reserve(f.tasks.size());
   for(auto it=f.tasks.begin();it!=f.tasks.end();){auto task=it->second;auto node=task->node.lock();
    if(!task->flight && (!node || node->exact_winner>=0)){it=f.tasks.erase(it);continue;}++it;
    if(task->flight || !node || node->dormant)continue;
    f.scope(*task,stamps);
    if(task->closed==3)continue;
    if(task->ready>now){if(retry==Clock::time_point{} || task->ready<retry)retry=task->ready;continue;}
    bool pending=node->pending;if(auto peers=o.game->positions.find(task->key);peers!=o.game->positions.end())for(auto& weak:peers->second)if(auto peer=weak.lock())pending|=peer->pending;
    if(pending)continue;double age=double(f.next-task->born+1);
    double score=task->impact*(1+task->change)/std::max(.05,task->cost)+.0001*age;
    if(candidates.single){if(!candidates.only.task || (explore?age:score)>candidates.score(0,explore))candidates.only={task,score,age};}
    else candidates.choices.push_back({task,score,age});
   }
   candidates.build();
  }return ready;
 }
 std::shared_ptr<Task> take(std::vector<Ready>& ready,size_t& game){
  double best=-1;size_t selected=SIZE_MAX;bool explore=submitted%5==4;
  for(size_t n=0;n<ready.size();++n){size_t i=(cursor+n)%ready.size();auto& candidates=ready[i];auto next=candidates.best(explore);
   if(next==SIZE_MAX)continue;double score=candidates.score(next,explore);
   if(score>best){best=score;selected=next;game=i;}
  }
  if(selected==SIZE_MAX)return {};
  auto& chosen=ready[game].choice(selected);chosen.picked=true;return chosen.task;
 }
 void admit(){
  // The graph owner cannot change candidates while this refill is dispatching.
  // Rank eligibility once, then consume priority/age heaps instead of rescanning
  // every position for each job. Worker completions install on the next step.
  std::vector<Ready> ready;Clock::time_point retry{};
  for(;;){
   size_t available;
   {std::lock_guard lock(mutex);if(stopping || !enabled || live.size()>=capacity)return;available=capacity-live.size();}
   if(ready.empty())ready=prepare(available,retry);
   size_t i=0;auto task=take(ready,i);
   if(!task){
    // Ranking/dispatch can outlast a retry delay. Refresh only when an excluded
    // cooldown has elapsed, without waiting or rescanning for every job.
    if(retry==Clock::time_point{} || retry>Clock::now())return;
    ready.clear();continue;
   }
   auto node=task->node.lock();auto& o=*pool.games[i];
   auto job=std::make_shared<Job>();job->id=++next;job->game=i;job->task=task;job->pin=node;job->history=task->history;
   job->side=task->side;job->preferred=task->worker;job->quantum=std::min(1000,slice*int(uint64_t(1)<<std::min(6u,task->attempts[job->side])));
    // Without a shared cancel flag, bound the synchronous WASM call by a short
    // slice. Cooperative adapters may continue longer on the same resident table.
   if(external)job->quantum=std::min(external_limit,job->quantum);
   job->generation=task->generation;job->facts=frontiers[i].revision;job->scope=task->scope;job->queued=Clock::now();job->context=context(i,*task);
   available_facts+=frontiers[i].facts.size();sent_facts+=task->scope.size();empty_scope_jobs+=task->scope.empty();quantum_ms+=job->quantum;
   if(o.time_limit_ms)job->deadline=o.started+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double,std::milli>(o.time_limit_ms));
   task->flight=true;task->attempted_change=task->observed_change;o.game->pins[task.get()]={node.get()};
   {std::lock_guard lock(mutex);live.emplace(job->id,job);queued.push_back(job);++submitted;}
   cursor=(i+1)%frontiers.size();wake.notify_one();
  }
 }
  std::string request(const Job& job,int ms,uint64_t token=0)const{
   return job.context.substr(0,job.context.size()-1)+",\"ms\":"+std::to_string(ms)+",\"nodes\":10000000,\"idtt_nodes\":0,\"depth\":8,\"attacker\":\""+(job.side?"defender":"mover")+"\",\"table_mb\":"+std::to_string(table)+",\"bounds\":true,\"resume\":true,\"stamps\":"+(stamps?"true":"false")+(endpoint_limit && !job.side?",\"neural_frontier\":"+std::to_string(endpoint_limit):"")+(token?",\"request_id\":"+std::to_string(token):"")+'}';
  }
  uint64_t external_take(int index){
   if(!external || index<0 || index>=int(workers.size()))throw std::runtime_error("Invalid external proof worker");
   auto& worker=*workers[index];if(worker.active)throw std::runtime_error("External proof worker is busy");
   while(enabled && !stopping && !queued.empty()){
    auto it=std::find_if(queued.begin(),queued.end(),[&](const auto& j){return j->preferred<0 || j->preferred==index;});
    if(it==queued.end())it=queued.begin();auto job=*it;queued.erase(it);
    int ms=job->quantum;if(job->deadline!=Clock::time_point{})ms=std::min(ms,int(std::chrono::duration_cast<std::chrono::milliseconds>(job->deadline-Clock::now()).count()));
    if(ms<1)mark(*job);
    if(job->cancelled){job->info[4]=1;completed(job);++finished;continue;}
    job->worker=index;job->started=Clock::now();job->wait=std::chrono::duration<double,std::milli>(job->started-job->queued).count();
    worker.idle+=std::chrono::duration<double,std::milli>(job->started-worker.idle_since).count();
    job->request=request(*job,ms);worker.active=job;++started;return job->id;
   }return 0;
  }
  void external_complete(int index,uint64_t id,const uint64_t* info,const int64_t* moves,int count,const char* result,const char* error){
   if(!external || index<0 || index>=int(workers.size()))throw std::runtime_error("Invalid external proof completion");
   auto& worker=*workers[index];auto job=worker.active;
   if(!job || job->id!=id || !info || count<0 || count>2 || (count && !moves))throw std::runtime_error("Unknown external proof completion");
   std::copy_n(info,13,job->info.begin());job->move_count=count;if(count)std::copy_n(moves,2*count,job->moves.begin());
   if(result)job->result=result;if(error)job->error=error;
   job->elapsed=std::chrono::duration<double,std::milli>(Clock::now()-job->started).count();worker.service+=job->elapsed;
   worker.idle_since=Clock::now();worker.active.reset();completed(job);++finished;
  }
#ifndef __EMSCRIPTEN__
  void run(size_t i) noexcept {
  auto& worker=*workers[i];auto idle_start=Clock::now();
  for(;;){std::unique_lock lock(mutex);wake.wait(lock,[&]{return stopping || !queued.empty();});if(stopping && queued.empty())return;
   // Prefer a continuation's resident table, but steal work rather than idle.
   auto next=std::find_if(queued.begin(),queued.end(),[&](const auto& j){return j->preferred<0 || j->preferred==int(i);});
   if(next==queued.end())next=queued.begin();auto job=*next;queued.erase(next);worker.active=job;job->worker=int(i);job->started=Clock::now();++started;
   worker.idle+=std::chrono::duration<double,std::milli>(job->started-idle_start).count();job->wait=std::chrono::duration<double,std::milli>(job->started-job->queued).count();
   job->info[4]=1; // Every pre-dispatch exit has confirmed zero fresh work.
   void* answer=nullptr;void* raw=nullptr;
   try {
    int ms=job->quantum;if(job->deadline!=Clock::time_point{})ms=std::min(ms,int(std::chrono::duration_cast<std::chrono::milliseconds>(job->deadline-Clock::now()).count()));
    if(ms<1)mark(*job);
    if(!job->cancelled){worker.token=api.prepare();if(!worker.token)job->error="cancellation token limit";
     else{job->token=worker.token;
       std::string payload=request(*job,ms,worker.token);
      job->info[4]=0;lock.unlock();answer=api.query(worker.native,payload.c_str());
      if(!answer || !api.info(answer,job->info.data()))job->error="missing typed native answer";
      else{job->move_count=api.moves(answer,job->moves.data(),2);if(job->move_count<0)throw std::runtime_error("Invalid typed proof witness");
       if(endpoint && !job->side){int count=endpoint(answer,nullptr,0);if(count<0 || count>endpoint_limit*130)throw std::runtime_error("Invalid neural frontier size");job->neural.resize(count);if(count && endpoint(answer,job->neural.data(),count)!=count)throw std::runtime_error("Invalid neural frontier payload");}
       if(job->info[0] || !job->neural.empty()){raw=api.json(answer);if(raw){job->result=static_cast<char*>(raw);api.buffer_free(raw);raw=nullptr;}}
      }
      if(answer){api.answer_free(answer);answer=nullptr;}
      // A deadline may return before cooperative background cancellation finishes.
      while(api.busy(worker.native))std::this_thread::sleep_for(std::chrono::microseconds(100));
      lock.lock();
     }
    }
   }catch(...){if(answer)api.answer_free(answer);if(raw)api.buffer_free(raw);if(!lock.owns_lock())lock.lock();job->error="native proof worker failure";}
   if(worker.token){api.release(worker.token);worker.token=0;}job->elapsed=std::chrono::duration<double,std::milli>(Clock::now()-job->started).count();
   worker.service+=job->elapsed;++finished;completed(job);worker.active.reset();idle_start=Clock::now();answered.notify_all();
   // Do not acquire the broker's mutex while holding the proof mutex. Retain
   // only its independent signal so detach/free cannot invalidate this wake.
   auto signal=job->wake_owner?listener:nullptr;lock.unlock();if(signal)signal->notify();
  }
 }
#endif
 void publish(Tree& view,const Board& board,const gumbel::Outcome& outcome){
  auto key=gumbel::keys(board).first;
  if(auto old=view.outcomes.find(key);old!=view.outcomes.end() && old->second.winner!=outcome.winner)throw std::runtime_error("Conflicting verified graph proof");
  view.record(key,outcome);
  if(auto peers=view.positions.find(key);peers!=view.positions.end())for(auto& weak:std::vector(peers->second))if(auto node=weak.lock())if(view.apply(view.outcomes.at(key),*node))view.revise(*node);
 }
 size_t neural(Job& job){
  if(job.neural.empty())return 0;
  auto start=Clock::now();auto& o=*pool.games[job.game];
  if(!endpoint_limit || job.side || !job.info[12] || job.neural.size()>size_t(endpoint_limit*130))throw std::runtime_error("Unexpected neural frontier");
  Board base;for(Cell c:job.history){if(!base.legal(c))throw std::runtime_error("Invalid frontier root");base.make(c);}
  size_t cursor=0,paths=0,added=0;
  while(cursor<job.neural.size()){
   if(job.neural.size()-cursor<2 || ++paths>size_t(endpoint_limit))throw std::runtime_error("Malformed neural frontier");
   int64_t count=job.neural[cursor++],reason=job.neural[cursor++];
   if(count<1 || count>64 || reason<0 || reason>1 || job.neural.size()-cursor<size_t(2*count))throw std::runtime_error("Malformed neural path");
   Board board=base;auto history=job.history;bool valid=board.winner<0;
   for(int i=0;i<count;++i){Cell c{job.neural[cursor],job.neural[cursor+1]};cursor+=2;
    if(!valid || c.q<-1000000 || c.q>1000000 || c.r<-1000000 || c.r>1000000 || !board.legal(c)){valid=false;continue;}
    board.make(c);history.push_back(c);if(board.winner>=0)valid=false;
   }
   for(auto [cell,player]:o.views[0].tree->board.cells){auto found=board.cells.find(cell);if(found==board.cells.end() || found->second!=player){valid=false;break;}}
   ++endpoint_paths;
   if(valid){added+=o.solver_path(history,o.focus.size(),job.task->impact);}
   else ++endpoint_rejected;
  }
  endpoint_candidates+=added;endpoint_bytes+=job.neural.size()*sizeof(int64_t);
  endpoint_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();return added;
 }
 void install(Job& job){
  auto start=Clock::now();auto& o=*pool.games[job.game];auto& f=frontiers[job.game];auto& task=*job.task;
  auto& effort=f.effort[job.generation];++effort.queries;
  if(job.info[4]){fresh+=job.info[3];effort.fresh+=job.info[3];}else{++missing_fresh;++effort.missing;}
  o.game->pins.erase(job.task.get());task.flight=false;
  if(job.cancelled || job.generation!=f.generation){task.ready=Clock::now()+std::chrono::milliseconds(slice);return;}
  if(!job.error.empty())throw std::runtime_error(job.error);
  int player=int((job.history.size()+1)/2%2),remaining=job.history.empty() || job.history.size()%2==0?1:2;
  if(job.info[12] && (job.info[9]!=uint64_t(player) || job.info[10]!=uint64_t(remaining) || job.info[11]!=uint64_t(job.side)))throw std::runtime_error("Native proof scope mismatch");
  int verdict=int(job.info[0]);
  if(external && (!job.info[12] || (verdict!=0 && verdict!=1 && verdict!=3) || (verdict && (job.info[2]<1 || job.info[1]!=uint64_t((verdict==1?player:1-player)+1)))))throw std::runtime_error("Invalid external proof outcome");
  if(verdict){
   if((verdict==1 && job.side!=0) || (verdict==3 && job.side!=1) || job.info[2]>10000)throw std::runtime_error("Invalid typed exact proof");
   Tree view(0,o.game);view.shared=view.graph=view.scheduler_owned=true;view.root_at(job.history);
   if(verdict==1){
    if(job.move_count<1 || job.move_count>remaining)throw std::runtime_error("Incomplete winning witness");
    Board after=view.board;for(int i=0;i<job.move_count;++i){Cell c{job.moves[2*i],job.moves[2*i+1]};if(!after.legal(c))throw std::runtime_error("Illegal winning witness");after.make(c);}
    if(after.winner<0 && after.player==player)throw std::runtime_error("Incomplete winning turn");
    int distance=job.move_count+4*(int(job.info[2])-1);Cell first{job.moves[0],job.moves[1]};
    if(job.move_count==2){Board second=view.board;second.make(first);publish(view,second,{player,player,distance-1,int(second.cells.size()),true,{{Cell{job.moves[2],job.moves[3]},player,distance-1,true}}});}
    publish(view,view.board,{player,player,distance,int(job.history.size()),true,{{first,player,distance,true}}});view.proof_root();
   }else if(!hxg_prove_loss(&view,1-player,4*int(job.info[2])+2))throw std::runtime_error(gumbel::error);
   view.trim_archive();f.remember(job.history,*view.root);f.tasks.erase(task.key);++installed;
   if(!job.result.empty()){records.push_back("{\"id\":"+std::to_string(job.id)+",\"game\":"+std::to_string(job.game)+",\"generation\":"+std::to_string(job.generation)+",\"request\":"+job.context+",\"result\":"+job.result+'}');if(records.size()>512)records.pop_front();}
  }else{
   size_t added=neural(job);
   if(!job.neural.empty()){
    endpoint_records.push_back("{\"id\":"+std::to_string(job.id)+",\"game\":"+std::to_string(job.game)+",\"generation\":"+std::to_string(job.generation)+",\"frontier_candidates\":"+std::to_string(added)+",\"request\":"+job.context+",\"result\":"+(job.result.empty()?"null":job.result)+'}');if(endpoint_records.size()>512)endpoint_records.pop_front();
   }
   ++unknown;++task.attempts[job.side];task.worker=job.worker;task.cost=.5*task.cost+.5*job.elapsed;task.change=0;
   f.scope(task,stamps);bool same_scope=job.scope==task.scope && (!stamps || job.facts==f.revision);
   if(same_scope && job.info[8] && job.info[6]>=1073741824 && job.info[7]==0)task.closed|=1<<job.side;
   int next_side=1-job.side;task.side=(task.closed&(1<<next_side))?job.side:next_side;
   task.ready=same_scope?Clock::now()+std::chrono::milliseconds(slice*int(uint64_t(1)<<std::min(6u,task.attempts[job.side]))):Clock::time_point{};
  }
  install_ns+=std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count();
 }
 void collect(){for(;;){std::shared_ptr<Job> job;{std::lock_guard lock(mutex);if(done.empty())return;job=done.front();done.pop_front();if(job->wake_owner){--urgent;completion_ready.store(urgent!=0,std::memory_order_release);}live.erase(job->id);}install(*job);}}
 void prune(){
  {std::lock_guard lock(mutex);for(auto& [id,job]:live){auto& o=*pool.games[job->game];if(o.stopped || job->generation!=frontiers[job->game].generation || job->pin->exact_winner>=0)mark(*job,true);}}
 }
 uint64_t feedback(){auto before=installed;prune();collect();prune();return installed-before;}
  void step(){
   ++ticks;feedback();
  for(size_t i=0;i<pool.games.size();++i){auto& o=*pool.games[i];if(o.stopped)continue;
   for(auto& v:o.views)if(v.active && v.tree->root->expanded)frontiers[i].offer(v.tree->root,v.history,v.relevance,std::abs(v.tree->root->q-v.tree->root->value));
  }
  admit();
 }
 void retarget(int game){auto& f=frontiers[game];++f.generation;f.tasks.clear();std::lock_guard lock(mutex);for(auto& [id,job]:live)if(job->game==size_t(game))mark(*job,true);}
 const char* release(int game){
  {std::lock_guard lock(mutex);for(auto& [id,job]:live)if(job->game==size_t(game))return nullptr;}
  auto& f=frontiers[game];released_effort="[";int count=0;
  for(auto [generation,e]:f.effort){if(count++)released_effort+=',';released_effort+='['+std::to_string(generation)+','+std::to_string(e.fresh)+','+std::to_string(e.queries)+','+std::to_string(e.missing)+']';}
  released_effort+=']';f.effort.clear();f.tasks.clear();f.facts.clear();f.fact_order.clear();f.members.clear();f.occupied.reset();f.next=f.offers=0;++f.revision;
  auto& store=*pool.games[game]->game;store.evidence=nullptr;store.evidence_owner=nullptr;
  return released_effort.c_str();
 }
  void cancel_all(){std::lock_guard lock(mutex);enabled=false;for(auto& [id,job]:live)mark(*job);if(external){for(auto& job:queued){job->info[4]=1;completed(job);++finished;}queued.clear();}wake.notify_all();}
 void resume(){std::lock_guard lock(mutex);enabled=true;}
  void drain(){cancel_all();if(external){collect();if(!live.empty())throw std::runtime_error("Complete external proof slices before freeing their graph");return;}
#ifndef __EMSCRIPTEN__
   for(;;){collect();std::unique_lock lock(mutex);if(live.empty())break;answered.wait(lock,[&]{return !done.empty();});}
#endif
  }
};
}
extern "C" HX_API void* hxp_new(void* pool,const uint64_t* functions,int workers,int capacity,int slice,int table,int tasks,int stamps){try{if(!pool || !functions)throw std::runtime_error("Missing proof pool");return new proving::Loop(*static_cast<owner::Pool*>(pool),functions,workers,capacity,slice,table,tasks,stamps!=0);}catch(const std::exception& e){gumbel::error=e.what();return nullptr;}}
extern "C" HX_API void* hxpe_new(void* pool,int workers,int capacity,int slice,int table,int tasks,int stamps,int maximum){try{if(!pool || maximum<1 || maximum>1000)throw std::runtime_error("Invalid external proof limits");auto* loop=new proving::Loop(*static_cast<owner::Pool*>(pool),nullptr,workers,capacity,slice,table,tasks,stamps!=0);loop->external_limit=maximum;return loop;}catch(const std::exception& e){gumbel::error=e.what();return nullptr;}}
extern "C" HX_API int hxp_neural(void* p,uint64_t callback,int limit){try{
 auto& loop=*static_cast<proving::Loop*>(p);std::lock_guard lock(loop.mutex);
 if(limit<0 || limit>8 || !loop.live.empty() || (!loop.external && limit && !callback))throw std::runtime_error("Invalid neural frontier configuration");
 loop.endpoint=reinterpret_cast<proving::API::Moves>(callback);loop.endpoint_limit=limit;return 1;
}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
extern "C" HX_API int hxpe_neural(void* p,int worker,uint64_t id,const int64_t* values,int count){try{
 auto& loop=*static_cast<proving::Loop*>(p);
 if(!loop.external || worker<0 || worker>=int(loop.workers.size()) || count<0 || count>loop.endpoint_limit*130 || (count && !values))throw std::runtime_error("Invalid external neural frontier");
 auto job=loop.workers[worker]->active;if(!job || job->id!=id)throw std::runtime_error("Unknown external neural frontier");
 if(count)job->neural.assign(values,values+count);return 1;
}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
extern "C" HX_API void hxp_neural_stats(void* p,uint64_t* out){auto& loop=*static_cast<proving::Loop*>(p);std::array<uint64_t,6> values{loop.endpoint_paths,loop.endpoint_candidates,loop.endpoint_rejected,loop.endpoint_bytes,loop.endpoint_ns,uint64_t(loop.endpoint_records.size())};std::copy(values.begin(),values.end(),out);}
extern "C" HX_API const char* hxp_neural_record(void* p,int i){auto& records=static_cast<proving::Loop*>(p)->endpoint_records;return i<0 || i>=int(records.size())?nullptr:records[i].c_str();}
extern "C" HX_API int hxpe_cancelled(void* p,int worker){auto& loop=*static_cast<proving::Loop*>(p);if(worker<0 || worker>=int(loop.workers.size()))return 0;auto& active=loop.workers[worker]->active;return active && active->cancelled;}
extern "C" HX_API uint64_t hxpe_take(void* p,int worker){try{return static_cast<proving::Loop*>(p)->external_take(worker);}catch(const std::exception& e){gumbel::error=e.what();return UINT64_MAX;}}
extern "C" HX_API const char* hxpe_request(void* p,int worker){auto& loop=*static_cast<proving::Loop*>(p);if(!loop.external || worker<0 || worker>=int(loop.workers.size()) || !loop.workers[worker]->active)return nullptr;return loop.workers[worker]->active->request.c_str();}
extern "C" HX_API int hxpe_complete(void* p,int worker,uint64_t id,const uint64_t* info,const int64_t* moves,int count,const char* result,const char* error){try{static_cast<proving::Loop*>(p)->external_complete(worker,id,info,moves,count,result,error);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
extern "C" HX_API int hxp_step(void* p){try{static_cast<proving::Loop*>(p)->step();return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
extern "C" HX_API void hxp_cancel(void* p){static_cast<proving::Loop*>(p)->cancel_all();}
extern "C" HX_API void hxp_resume(void* p){static_cast<proving::Loop*>(p)->resume();}
extern "C" HX_API int hxp_drain(void* p){try{static_cast<proving::Loop*>(p)->drain();return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
extern "C" HX_API int hxp_free(void* p){auto& loop=*static_cast<proving::Loop*>(p);if(loop.pool.inference_owner){gumbel::error="Detach the native inference service before freeing proof loop";return 0;}{std::lock_guard lock(loop.mutex);if(!loop.live.empty()){gumbel::error="Drain proof jobs before freeing loop";return 0;}}delete &loop;return 1;}
extern "C" HX_API int hxp_offer(void* p,int game,const int64_t* cells,int count,double relevance){try{
 auto& loop=*static_cast<proving::Loop*>(p);if(game<0 || game>=int(loop.pool.games.size()) || count<0 || (count && !cells) || !std::isfinite(relevance) || relevance<0)throw std::runtime_error("Invalid proof frontier position");
 std::vector<Cell> history;for(int i=0;i<count;++i)history.push_back({cells[2*i],cells[2*i+1]});auto& owner=*loop.pool.games[game];
 gumbel::Tree view(0,owner.game);view.shared=view.graph=view.scheduler_owned=true;view.root_at(history);loop.frontiers[game].offer(view.root,history,relevance,0);return 1;
}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
extern "C" HX_API void hxp_stats(void* p,uint64_t* out,double* times){auto& loop=*static_cast<proving::Loop*>(p);std::lock_guard lock(loop.mutex);uint64_t active=0,tasks=0,facts=0;double service=0,idle=0;for(auto& w:loop.workers){active+=bool(w->active);service+=w->service;idle+=w->idle;}for(auto& f:loop.frontiers){tasks+=f.tasks.size();facts+=f.facts.size();}
 std::array<uint64_t,16> values{loop.ticks,loop.submitted,loop.started,loop.finished,loop.installed,loop.cancelled,loop.pruned,loop.unknown,loop.fresh,loop.missing_fresh,uint64_t(loop.queued.size()),active,uint64_t(loop.done.size()),tasks,facts,uint64_t(loop.records.size())};std::copy(values.begin(),values.end(),out);times[0]=service;times[1]=idle;times[2]=loop.snapshot_ns/1e6;times[3]=loop.install_ns/1e6;}
extern "C" HX_API const char* hxp_record(void* p,int i){auto& records=static_cast<proving::Loop*>(p)->records;return i<0 || i>=int(records.size())?nullptr:records[i].c_str();}
extern "C" HX_API void hxp_scope_stats(void* p,uint64_t* out){auto& loop=*static_cast<proving::Loop*>(p);uint64_t checks=0,changed=0,unchanged=0,ns=0,cells=0,closed=0;
 for(auto& f:loop.frontiers){checks+=f.scope_checks;changed+=f.scope_changed;unchanged+=f.scope_unchanged;ns+=f.scope_ns;cells+=f.members.size();for(auto& [key,task]:f.tasks)closed+=bool(task->closed&1)+bool(task->closed&2);}
 std::array<uint64_t,10> values{checks,changed,unchanged,ns,loop.available_facts,loop.sent_facts,loop.empty_scope_jobs,loop.quantum_ms,cells,closed};std::copy(values.begin(),values.end(),out);}
extern "C" HX_API uint64_t hxp_generation(void* p,int game){return static_cast<proving::Loop*>(p)->frontiers.at(game).generation;}
extern "C" HX_API const char* hxp_release(void* p,int game){return static_cast<proving::Loop*>(p)->release(game);}
extern "C" HX_API int hxp_effort(void* p,int game,uint64_t* out){auto& f=static_cast<proving::Loop*>(p)->frontiers.at(game);int i=0;for(auto [generation,e]:f.effort){if(out){std::array<uint64_t,4> row{generation,e.fresh,e.queries,e.missing};std::copy(row.begin(),row.end(),out+4*i);}++i;}return i;}
