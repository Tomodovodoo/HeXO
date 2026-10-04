#pragma once
#include <condition_variable>
#include <atomic>
#include <mutex>
#include <thread>
#include <sstream>
#include <iomanip>

extern "C" {
void* hxgp_new(void* const*,const int*,int,int);
void* hxgp_combine(void* const*,const int*,int,int);
void hxgp_free(void*);
int hxgp_outputs(void*,void**);
int hxgf_layout(void*,int,int64_t*);
int hxgf_take(void*,int,uint64_t*,void**,int*,int64_t*,int64_t*,int64_t);
const int64_t* hxgf_key(void*,uint64_t,int64_t*);
int hxgf_retire(void*,uint64_t);
int hxgf_abandon_all(void*);
void hxp_cancel(void*);
void hxp_resume(void*);
int hxp_drain(void*);
uint64_t hxp_generation(void*,int);
const char* hxp_release(void*,int);
}

namespace inference {
using Clock=std::chrono::steady_clock;
using Key=std::vector<int64_t>;
struct Hash {
 size_t operator()(const Key& key)const {uint64_t h=0xcbf29ce484222325ULL;for(auto v:key){h^=uint64_t(v);h*=0x100000001b3ULL;h^=h>>32;}return size_t(h);}
};
struct SnapshotDeleter {void operator()(void* p)const{if(p)hxgp_free(p);}};
struct Prediction {std::vector<int64_t> actions;std::vector<double> logits,values;};
struct Producer;
struct Job {
 std::shared_ptr<Producer> owner;std::unique_ptr<void,SnapshotDeleter> snapshot;
 std::vector<uint64_t> ids;std::vector<Key> keys;
 std::vector<std::shared_ptr<const Prediction>> results;
 std::vector<bool> installed;
 int remaining=0;
};
struct Completion {std::shared_ptr<Job> job;std::vector<int> rows;bool finished;};
struct Subscriber {std::shared_ptr<Job> job;int row;};
struct Task {
 Key key;int model,row;std::shared_ptr<Job> representative;
 std::vector<Subscriber> subscribers;Clock::time_point queued;bool flight=false,live=true;
};
struct Broker;
struct Command {int game,samples,views;uint64_t token,work;double ms,noise;std::vector<int64_t> cells;
 int kind=0;std::shared_ptr<gumbel::GameStore> replacement;bool tactics=false;double range=0;uint64_t seed=0;};
struct Producer:std::enable_shared_from_this<Producer> {
 Broker& broker;owner::Pool& pool;int model,index,worker_request=0;
 std::atomic<bool> retiring=false;
 std::thread thread;std::deque<std::shared_ptr<Job>> completed;
 std::vector<std::shared_ptr<Job>> outstanding;bool done=false;uint64_t pause_ack=0;
 std::deque<Command> commands;std::vector<uint64_t> epochs;
 std::vector<bool> requested,reported,released;
 Producer(Broker& b,owner::Pool& p,int m,int i):broker(b),pool(p),model(m),index(i),epochs(p.games.size()),requested(p.games.size()),reported(p.games.size(),true),released(p.games.size()){}
 void run()noexcept;
 std::string result(int index,uint64_t token);
};
// The service owns only immutable snapshots and prediction messages. Producers
// exclusively mutate their pools. No GPU callback dereferences a graph node.
struct Broker {
 std::mutex mutex;std::condition_variable wake;
 // Stable registry slots; old immutable jobs can retain a joined producer.
 // Model IDs never change identity, even after its last producer retires.
 std::vector<std::shared_ptr<Producer>> producers{16};std::map<int,std::string> models;
 std::unordered_map<Key,std::shared_ptr<Task>,Hash> tasks;
 std::deque<std::shared_ptr<Task>> ready;
 std::map<uint64_t,std::vector<std::shared_ptr<Task>>> flights;
 int quantum,pending,merge_cells;double latency_ms;
 std::atomic<bool> cancelled=false;bool started=false,joined=false,continuous=false,paused=false;uint64_t pause_epoch=0;
 uint64_t next=0,created=0,coalesced=0,launched=0,delivered=0,installed_messages=0,withdrawn=0,batches=0,high_water=0;
 std::string error;
 struct Event {Producer* producer;int game;std::string text;};
 std::deque<Event> events;std::string last_event;
 Broker(int q,int p,int merge,double latency):quantum(q),pending(p),merge_cells(merge),latency_ms(latency){
  if(q<1 || q>128 || p<1 || p>4 || merge<0 || !std::isfinite(latency) || latency<0 || latency>20)
   throw std::runtime_error("Invalid native inference service limits");
 }
 int attach(owner::Pool& pool,int model){
  std::lock_guard lock(mutex);
  if((started && !continuous) || cancelled || joined || model<0 || pool.inference_owner || pool.stopped)
   throw std::runtime_error("Attach an idle unused pool to the inference service");
  auto vacant=std::find(producers.begin(),producers.end(),nullptr);
  if(vacant==producers.end())throw std::runtime_error("Native inference producer capacity reached");
  int index=int(vacant-producers.begin());
  int64_t feed[6];hxgf_stats(pool.feed,feed);if(feed[4])throw std::runtime_error("Drain prior neural work before attaching a producer");
  if(auto found=models.find(model);found!=models.end() && found->second!=pool.model)
   throw std::runtime_error("Inference model identity conflict");
  auto producer=std::make_shared<Producer>(*this,pool,model,index);
  if(started){pool.ready_limit=2*quantum;pool.stop();if(pool.proof_owner)hxp_resume(pool.proof_owner);}
  models[model]=pool.model;*vacant=producer;pool.inference_owner=this;
  try{if(started)producer->thread=std::thread([producer]{producer->run();});}
  catch(...){vacant->reset();pool.inference_owner=nullptr;throw;}
  wake.notify_all();return index;
 }
 void detach(int index){
  std::shared_ptr<Producer> producer;
  {std::lock_guard lock(mutex);
   if(!continuous || !started || cancelled || index<0 || index>=int(producers.size()) || !producers[index])
    throw std::runtime_error("Invalid producer retirement");
   producer=producers[index];
   if(producer->retiring || producer->worker_request || !producer->commands.empty() ||
      std::any_of(producer->requested.begin(),producer->requested.end(),[](bool b){return b;}) ||
      std::any_of(producer->released.begin(),producer->released.end(),[](bool b){return !b;}) ||
      std::any_of(producer->reported.begin(),producer->reported.end(),[](bool b){return !b;}))
    throw std::runtime_error("Consume every game retirement before detaching its producer");
   for(auto& game:producer->pool.games)if(std::any_of(game->game->pins.begin(),game->game->pins.end(),[&](const auto& pin){
       return std::none_of(game->views.begin(),game->views.end(),[&](const auto& view){return pin.first==view.tree.get();});}))
    throw std::runtime_error("Close retired caller trees before detaching their producer");
   producer->retiring=true;wake.notify_all();
  }
  producer->thread.join();
  std::lock_guard lock(mutex);producers[index].reset();producer->pool.inference_owner=nullptr;
 }
 bool model_pending(int model){
  std::lock_guard lock(mutex);
  return std::any_of(producers.begin(),producers.end(),[&](const auto& p){return p && p->model==model;}) ||
         std::any_of(tasks.begin(),tasks.end(),[&](const auto& task){return task.second->model==model;});
 }
 void workers(int index,int count){
  std::unique_lock lock(mutex);
  if(!continuous || !started || cancelled || index<0 || index>=int(producers.size()) ||
     !producers[index] || count<1 || count>16)throw std::runtime_error("Invalid native host allocation");
  auto p=producers[index];
  if(p->done || p->retiring || p->worker_request)throw std::runtime_error("Producer is unavailable for host allocation");
  p->worker_request=count;wake.notify_all();
  wake.wait(lock,[&]{return !p->worker_request || cancelled || p->done;});
  if(!error.empty())throw std::runtime_error(error);
  if(p->worker_request)throw std::runtime_error("Host allocation cancelled before acknowledgement");
 }
 int worker_request(Producer& p){std::lock_guard lock(mutex);return p.worker_request;}
 void workers_ready(Producer& p){std::lock_guard lock(mutex);p.worker_request=0;wake.notify_all();}
 void fail(const std::string& message){
  {std::lock_guard lock(mutex);if(error.empty())error=message;}cancel();
 }
 void cancel(){
  cancelled=true;
  {std::lock_guard lock(mutex);for(auto& p:producers)if(p && p->pool.proof_owner)hxp_cancel(p->pool.proof_owner);}
  wake.notify_all();
 }
 void start(double ms){
  std::unique_lock lock(mutex);
  if(started || std::none_of(producers.begin(),producers.end(),[](const auto& p){return bool(p);}) || !std::isfinite(ms) || ms<0)throw std::runtime_error("Invalid inference service start");
  started=true;auto now=Clock::now();
  try{
   for(auto& p:producers)if(p){auto& pool=p->pool;pool.ready_limit=2*quantum;
    if(continuous)pool.stop();
    if(ms)for(auto& game:pool.games){game->time_limit_ms=ms;game->work_limit=0;game->started=now;game->deadline=false;}
    if(pool.proof_owner)hxp_resume(pool.proof_owner);
   }
   for(auto& p:producers)if(p)p->thread=std::thread([source=p]{source->run();});
  }catch(...){lock.unlock();cancel();for(auto& p:producers)if(p && p->thread.joinable())p->thread.join();throw;}
 }
 void retarget(int producer,int game,uint64_t expected,const int64_t* cells,int count,uint64_t work,double ms,int samples,int views,double noise,gumbel::Tree* source=nullptr,const char* version=nullptr,uint64_t seed=0){
  if(!continuous || !started || cancelled || producer<0 || producer>=int(producers.size()) || count<0 || (count && !cells) ||
     (!work && !(ms>0) && !source) || !std::isfinite(ms) || ms<0 || samples<1 || samples>1024 || views<1 || views>64 || !std::isfinite(noise) || noise<0 || noise>1)
   throw std::runtime_error("Invalid continuous search command");
  std::lock_guard lock(mutex);if(!producers[producer])throw std::runtime_error("Producer is retired");auto& p=*producers[producer];
  if(game<0 || game>=int(p.epochs.size()) || p.done || p.retiring || p.requested[game] || expected!=p.epochs[game] || !p.reported[game])
   throw std::runtime_error("Consume the matching game completion before retargeting");
  if(p.released[game]!=(source!=nullptr))throw std::runtime_error("Release the previous game before replacing its slot");
  Command command{game,samples,views,expected+1,work,ms,noise,{}};if(count)command.cells.assign(cells,cells+2*count);
  if(source){
   auto& old=*p.pool.games[game];
   if(std::any_of(old.game->pins.begin(),old.game->pins.end(),[&](const auto& pin){
       return std::none_of(old.views.begin(),old.views.end(),[&](const auto& view){return pin.first==view.tree.get();});}))
    throw std::runtime_error("Close retired caller trees before replacing their slot");
   if(!version || p.pool.model!=version || !source->shared || source->state->scheduler_owner || !source->requests.empty() || count!=int(source->board.history.size()) ||
      std::any_of(source->state->pins.begin(),source->state->pins.end(),[&](const auto& pin){return pin.first!=source;}))
    throw std::runtime_error("Replacement requires a fresh graph of the slot's frozen model");
   for(int i=0;i<count;++i)if(source->board.history[i].c!=Cell{cells[2*i],cells[2*i+1]})throw std::runtime_error("Replacement history mismatch");
   command.kind=2;command.replacement=source->state;command.tactics=source->tactics;command.range=source->range_floor;command.seed=seed;
  }
  p.commands.push_back(std::move(command));p.requested[game]=true;
  // A successful replacement transfers the source Tree. Destroy its caller
  // pin before the producer can mutate the retained store; shared ownership
  // alone would not prevent concurrent unordered_map mutation on close().
  if(source)delete source;
  wake.notify_all();
 }
 void release(int producer,int game,uint64_t expected){
  std::lock_guard lock(mutex);
  if(!continuous || !started || cancelled || producer<0 || producer>=int(producers.size()))throw std::runtime_error("Invalid continuous release");
  if(!producers[producer])throw std::runtime_error("Producer is retired");auto& p=*producers[producer];
  if(game<0 || game>=int(p.epochs.size()) || p.done || p.retiring || p.requested[game] || !p.reported[game] || p.released[game] || expected!=p.epochs[game])
   throw std::runtime_error("Consume the matching root completion before releasing its game");
  Command command{};command.game=game;command.token=expected+1;command.kind=1;
  p.requested[game]=true;p.commands.push_back(std::move(command));wake.notify_all();
 }
 void pause(bool value){
  std::lock_guard lock(mutex);
  if(!continuous || !started || cancelled)throw std::runtime_error("Resumable pause requires a running continuous inference service");
  if(paused!=value){paused=value;++pause_epoch;}wake.notify_all();
 }
 bool admission(uint64_t& epoch){std::lock_guard lock(mutex);epoch=paused?pause_epoch:0;return !paused;}
 void acknowledge(Producer& p,uint64_t epoch){
  std::lock_guard lock(mutex);if(!paused || epoch!=pause_epoch || !p.completed.empty())return;
  for(auto& job:p.outstanding)for(size_t row=0;row<job->results.size();++row)if(job->results[row] && !job->installed[row])return;
  p.pause_ack=epoch;
 }
 bool fenced(){
  std::lock_guard lock(mutex);if(!error.empty())throw std::runtime_error(error);
  return paused && flights.empty() && std::all_of(producers.begin(),producers.end(),[&](const auto& p){return !p || p->done || (p->pause_ack==pause_epoch && p->completed.empty());});
 }
 std::vector<Command> commands(Producer& p){std::lock_guard lock(mutex);std::vector<Command> out(p.commands.begin(),p.commands.end());p.commands.clear();return out;}
 void publish(Producer& p,int game,uint64_t token,std::string event,bool released=false){
  std::lock_guard lock(mutex);p.epochs[game]=token;p.reported[game]=false;p.requested[game]=false;p.released[game]=released;events.push_back({&p,game,std::move(event)});wake.notify_all();
 }
 const char* event(){
  std::lock_guard lock(mutex);if(!error.empty())throw std::runtime_error(error);if(events.empty())return nullptr;
  auto event=std::move(events.front());events.pop_front();event.producer->reported[event.game]=true;last_event=std::move(event.text);return last_event.c_str();
 }
 void finish_row(Subscriber s,std::shared_ptr<const Prediction> prediction){
  if(s.job->results[s.row])return;
  s.job->results[s.row]=std::move(prediction);
  s.job->owner->pause_ack=0;
  if(!--s.job->remaining)s.job->owner->completed.push_back(s.job);
 }
 void enqueue(const std::shared_ptr<Job>& job){
  std::lock_guard lock(mutex);
  for(size_t i=0;i<job->keys.size();++i){auto& key=job->keys[i];
   if(auto found=tasks.find(key);found!=tasks.end()){
    found->second->subscribers.push_back({job,int(i)});++coalesced;
   }else{
    auto task=std::make_shared<Task>();task->key=key;task->model=job->owner->model;task->row=int(i);
    task->representative=job;task->subscribers.push_back({job,int(i)});task->queued=Clock::now();
    tasks.emplace(key,task);ready.push_back(std::move(task));++created;
   }
  }
  high_water=std::max(high_water,uint64_t(tasks.size()));wake.notify_all();
 }
 void withdraw(const std::shared_ptr<Job>& job,int row){
  std::lock_guard lock(mutex);if(job->results[row])return;
  auto found=tasks.find(job->keys[row]);if(found==tasks.end()){
   if(!cancelled)throw std::runtime_error("Missing inference subscriber");
   finish_row({job,row},std::make_shared<Prediction>());return;
  }
  auto task=found->second;
  std::erase_if(task->subscribers,[&](const Subscriber& s){return s.job==job && s.row==row;});
  finish_row({job,row},std::make_shared<Prediction>());
  if(task->subscribers.empty() && !task->flight){task->live=false;tasks.erase(found);++withdrawn;}
  wake.notify_all();
 }
 std::vector<Completion> completions(Producer& producer){
  std::lock_guard lock(mutex);std::vector<std::shared_ptr<Job>> jobs(producer.completed.begin(),producer.completed.end());
  producer.completed.clear();
  // A producer snapshot may span several device batches. Pausing must install
  // the rows already returned without launching the snapshot's remaining rows.
  if(paused)for(auto& job:producer.outstanding)if(std::find(jobs.begin(),jobs.end(),job)==jobs.end())jobs.push_back(job);
  std::vector<Completion> out;
  for(auto& job:jobs){Completion c{job,{},job->remaining==0};
   for(size_t row=0;row<job->results.size();++row)if(job->results[row] && !job->installed[row]){c.rows.push_back(int(row));job->installed[row]=true;}
   if(!c.rows.empty() || c.finished)out.push_back(std::move(c));
  }
  if(!out.empty())producer.pause_ack=0;return out;
 }
 void wait(Producer& producer){
  std::unique_lock lock(mutex);wake.wait_for(lock,std::chrono::milliseconds(1),[&]{return cancelled || producer.worker_request || !producer.completed.empty() || !producer.commands.empty();});
 }
 bool done_locked()const{return started && std::all_of(producers.begin(),producers.end(),[](const auto& p){return !p || p->done;}) && flights.empty();}
 int take(int limit,double wait_ms,uint64_t* token,int* model,void** snapshot){
  if(limit<1 || limit>1024 || !std::isfinite(wait_ms) || wait_ms<0 || wait_ms>1000)throw std::runtime_error("Invalid service batch request");
  auto until=Clock::now()+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double,std::milli>(wait_ms));
  std::unique_lock lock(mutex);
  for(;;){
   if(!error.empty())throw std::runtime_error(error);
   if(cancelled || paused)return 0;
   if(flights.size()>=2){if(!wait_ms || Clock::now()>=until)return 0;wake.wait_until(lock,until);continue;}
   std::erase_if(ready,[](const auto& t){return !t->live || t->flight;});
   if(!ready.empty()){
    int selected=ready.front()->model;int count=int(std::count_if(ready.begin(),ready.end(),[&](const auto& t){return t->model==selected;}));
    auto due=ready.front()->queued+std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double,std::milli>(latency_ms));
    if(count>=limit || Clock::now()>=due || (wait_ms>0 && Clock::now()>=until)){
     std::vector<std::shared_ptr<Task>> batch;
     for(auto it=ready.begin();it!=ready.end() && int(batch.size())<limit;){
      auto task=*it;if(task->model!=selected){++it;continue;}
      task->flight=true;batch.push_back(task);it=ready.erase(it);
     }
     uint64_t id=++next;flights.emplace(id,batch);launched+=batch.size();++batches;
     lock.unlock();std::vector<void*> sources;std::vector<int> rows;
     for(auto& task:batch){sources.push_back(task->representative->snapshot.get());rows.push_back(task->row);}
     auto packed=hxgp_combine(sources.data(),rows.data(),int(rows.size()),merge_cells);
     if(!packed){std::string message=gumbel::error;lock.lock();flights.erase(id);
      for(auto& task:batch){task->flight=false;if(task->subscribers.empty()){task->live=false;tasks.erase(task->key);}else ready.push_back(task);}
      throw std::runtime_error(message);
     }
     *token=id;*model=selected;*snapshot=packed;return int(rows.size());
    }
    if(!wait_ms)return 0;wake.wait_until(lock,std::min(until,due));
   }else{
    if(done_locked() || Clock::now()>=until)return 0;wake.wait_until(lock,until);
   }
  }
 }
 void complete(uint64_t id,const int64_t* offsets,const int64_t* actions,const double* logits,const double* values){
  std::lock_guard lock(mutex);auto found=flights.find(id);
  if(found==flights.end() || !offsets || offsets[0]!=0)throw std::runtime_error("Unknown inference completion");
  auto& batch=found->second;std::vector<std::shared_ptr<const Prediction>> predictions;predictions.reserve(batch.size());
  for(size_t i=0;i<batch.size();++i){int64_t first=offsets[i],last=offsets[i+1];
   if(last<first || (last>first && (!actions || !logits || !values)))throw std::runtime_error("Invalid inference outputs");
   for(int64_t j=first;j<last;++j)if(!std::isfinite(logits[j]) || !std::isfinite(values[j]) || std::abs(values[j])>1)
    throw std::runtime_error("Invalid inference prediction");
   auto prediction=std::make_shared<Prediction>();
   if(last>first){prediction->actions.assign(actions+2*first,actions+2*last);prediction->logits.assign(logits+first,logits+last);prediction->values.assign(values+first,values+last);}
   predictions.push_back(std::move(prediction));
  }
  for(size_t i=0;i<batch.size();++i){auto& task=batch[i];for(auto s:task->subscribers){finish_row(s,predictions[i]);++delivered;}
   task->live=false;tasks.erase(task->key);
  }
  flights.erase(found);wake.notify_all();
 }
 void abort(uint64_t id){
  std::lock_guard lock(mutex);auto found=flights.find(id);if(found==flights.end())throw std::runtime_error("Unknown abandoned service batch");
  for(auto& task:found->second){for(auto s:task->subscribers)finish_row(s,std::make_shared<Prediction>());task->live=false;tasks.erase(task->key);}
  flights.erase(found);wake.notify_all();
 }
 void join(){
  cancel();for(auto& p:producers)if(p && p->thread.joinable())p->thread.join();
  std::lock_guard lock(mutex);if(!flights.empty())throw std::runtime_error("Fence and complete service batches before joining");joined=true;
 }
};
inline void Producer::run()noexcept{
 try{
  std::vector<uint64_t> active(pool.games.size()),retiring(pool.games.size());
  while(!broker.cancelled && (broker.continuous || pool.admit())){
   if(int count=broker.worker_request(*this)){
    // The preceding graph/feed phase has joined. Caller threads never resize
    // worker storage while selection or installation is using it.
    if(!hxgf_workers(pool.feed,count,[](void* tree)->void*{return static_cast<gumbel::Tree*>(tree)->state.get();}))throw std::runtime_error(gumbel::error);
    pool.host_workers=count;broker.workers_ready(*this);
   }
   for(auto& completion:broker.completions(*this)){
    auto& job=completion.job;std::vector<uint64_t> ids;
    std::vector<int64_t> offsets(1,0),actions;std::vector<double> logits,values;
    for(int row:completion.rows){auto& p=job->results[row];ids.push_back(job->ids[row]);actions.insert(actions.end(),p->actions.begin(),p->actions.end());logits.insert(logits.end(),p->logits.begin(),p->logits.end());values.insert(values.end(),p->values.begin(),p->values.end());offsets.push_back(int64_t(logits.size()));}
    if(!ids.empty()){
     pool.install(ids.data(),int(ids.size()),offsets.data(),actions.data(),logits.data(),values.data());
     std::lock_guard lock(broker.mutex);broker.installed_messages+=ids.size();
    }
    if(completion.finished)std::erase(outstanding,job);
   }
   for(auto& command:broker.commands(*this)){
    if(command.kind==1){
     pool.games[command.game]->stop();
     if(pool.proof_retarget)pool.proof_retarget(pool.proof_owner,command.game);
     retiring[command.game]=command.token;continue;
    }
    if(command.kind==2){
     gumbel::Tree source(command.seed,command.replacement);source.shared=source.graph=true;
     source.tactics=command.tactics;source.range_floor=command.range;
     std::vector<Cell> history;for(size_t i=0;i<command.cells.size();i+=2)history.push_back({command.cells[i],command.cells[i+1]});source.root_at(history);
     pool.replace(command.game,source,command.samples,command.views,command.work,command.ms,command.noise,command.seed);
     if(!command.work && !command.ms){
      int producer=index;
      broker.publish(*this,command.game,command.token,"{\"kind\":\"replaced\",\"producer\":"+std::to_string(producer)+",\"model\":"+std::to_string(model)+",\"game\":"+std::to_string(command.game)+",\"token\":"+std::to_string(command.token)+'}');
     }else active[command.game]=command.token;
     continue;
    }
    auto& o=*pool.games[command.game];o.sample_limit=command.samples;o.max_views=command.views;
    o.views[0].tree->root_noise=command.noise;
    pool.retarget(command.game,command.cells.data(),int(command.cells.size()/2),command.work,command.ms);
    active[command.game]=command.token;
   }
   uint64_t pause_token=0;bool neural=broker.admission(pause_token);
   int progress=pool.step(neural);
   for(auto& job:outstanding)for(size_t row=0;row<job->ids.size();++row){
    if(job->installed[row])continue;
    int remaining=hxgf_retire(pool.feed,job->ids[row]);if(remaining<0)throw std::runtime_error(gumbel::error);
    if(!remaining)broker.withdraw(job,int(row));
   }
   for(size_t i=0;i<retiring.size();++i)if(retiring[i]){
    const char* effort=pool.proof_owner?hxp_release(pool.proof_owner,int(i)):"[]";
    if(!effort)continue;
    int producer=index;
    std::string event="{\"kind\":\"released\",\"producer\":"+std::to_string(producer)+",\"model\":"+std::to_string(model)+",\"game\":"+std::to_string(i)+",\"token\":"+std::to_string(retiring[i])+",\"effort\":"+effort+'}';
    broker.publish(*this,int(i),retiring[i],std::move(event),true);retiring[i]=0;progress=1;
   }
   if(broker.continuous)for(size_t i=0;i<active.size();++i)if(active[i] && pool.games[i]->stopped){
    auto token=active[i];auto event=result(int(i),token);active[i]=0;broker.publish(*this,int(i),token,std::move(event));progress=1;
   }
   if(neural && int(outstanding.size())<broker.pending && !pool.stopped){
    int64_t layout[2];if(!hxgf_layout(pool.feed,broker.quantum,layout))throw std::runtime_error(gumbel::error);
    int count=int(layout[0]);if(count){
     auto job=std::make_shared<Job>();job->owner=shared_from_this();job->ids.resize(count);job->results.resize(count);job->installed.resize(count);job->remaining=count;
     std::vector<void*> trees(count);std::vector<int> requests(count);
     if(!hxgf_take(pool.feed,count,job->ids.data(),trees.data(),requests.data(),nullptr,nullptr,0))throw std::runtime_error(gumbel::error);
     job->snapshot.reset(hxgp_new(trees.data(),requests.data(),count,0));if(!job->snapshot)throw std::runtime_error(gumbel::error);
     for(auto id:job->ids){int64_t size=0;auto key=hxgf_key(pool.feed,id,&size);if(!key)throw std::runtime_error(gumbel::error);
      job->keys.emplace_back(1,int64_t(model));job->keys.back().insert(job->keys.back().end(),key,key+size);
     }
     outstanding.push_back(job);broker.enqueue(job);progress=1;
    }
   }
   broker.acknowledge(*this,pause_token);
   if(this->retiring && outstanding.empty())break;
   if(!progress)broker.wait(*this);
  }
 }catch(const std::exception& e){broker.fail(e.what());}catch(...){broker.fail("Native inference producer failed");}
 try{
  pool.stop();if(pool.proof_owner){hxp_cancel(pool.proof_owner);if(!hxp_drain(pool.proof_owner))throw std::runtime_error(gumbel::error);}
  for(auto& job:outstanding)for(size_t row=0;row<job->ids.size();++row)broker.withdraw(job,int(row));
  // GPU readers own copied snapshots, never this feed's trees or request table.
  if(!hxgf_abandon_all(pool.feed))throw std::runtime_error(gumbel::error);
 }catch(const std::exception& e){broker.fail(e.what());}catch(...){broker.fail("Native producer teardown failed");}
 {std::lock_guard lock(broker.mutex);completed.clear();outstanding.clear();done=true;}broker.wake.notify_all();
}
inline std::string Producer::result(int index,uint64_t token){
 auto& o=*pool.games[index];auto& t=*o.views[0].tree;t.proof_root();auto& n=*t.root;
 if(pool.failed[index] || (!n.expanded && o.deadline)){
  int producer=this->index;
  std::string reason=pool.failed[index]?"span":"deadline";
  std::string out="{\"producer\":"+std::to_string(producer)+",\"model\":"+std::to_string(model)+",\"game\":"+std::to_string(index)+",\"token\":"+std::to_string(token)+",\"error\":\""+reason+"\",\"history\":[";
  for(size_t i=0;i<o.focus.size();++i){if(i)out+=',';out+='['+std::to_string(o.focus[i].q)+','+std::to_string(o.focus[i].r)+']';}return out+"]}";
 }
 if(!n.expanded || n.edges.empty())throw std::runtime_error("Continuous root has no legal search result");
 t.current(n);int ignored=0;auto q=t.completed_q(n,ignored);
 uint64_t maximum=0;double lo=1e300,hi=-1e300;
 for(size_t i=0;i<n.edges.size();++i)if(n.edges[i].eligible){maximum=std::max(maximum,i<o.direct_root_credits.size()?o.direct_root_credits[i]:0);lo=std::min(lo,q[i]);hi=std::max(hi,q[i]);}
 if(lo>hi)throw std::runtime_error("No eligible continuous root action");
 double range=std::max({1e-8,t.range_floor,hi-lo}),highest=-1e300,total=0;
 std::vector<double> weights(n.edges.size());
 for(size_t i=0;i<n.edges.size();++i)if(n.edges[i].eligible)highest=std::max(highest,n.edges[i].logit+(q[i]-lo)/range*(50.+maximum)*.1+t.bonus(n.edges[i]));
 for(size_t i=0;i<n.edges.size();++i)if(n.edges[i].eligible)total+=weights[i]=std::exp(n.edges[i].logit+(q[i]-lo)/range*(50.+maximum)*.1+t.bonus(n.edges[i])-highest);
 int64_t action[2];if(!o.choice(action))throw std::runtime_error("Continuous root cannot select a move");
 double raw=o.root_raw;bool raw_known=o.root_raw_known;auto key=gumbel::keys(o.focus).second;
 int producer=this->index;
 std::ostringstream out;out<<std::setprecision(17);
 auto cells=[&](const std::vector<Cell>& h){out<<'[';for(size_t j=0;j<h.size();++j){if(j)out<<',';out<<'['<<h[j].q<<','<<h[j].r<<']';}out<<']';};
 out<<"{\"producer\":"<<producer<<",\"model\":"<<model<<",\"game\":"<<index<<",\"token\":"<<token<<",\"history\":";cells(o.focus);
 out<<",\"action\":["<<action[0]<<','<<action[1]<<"],\"exact_winner\":"<<n.exact_winner<<",\"proof_plies\":"<<n.distance<<",\"completed\":"<<o.completed<<",\"issued\":"<<o.issued<<",\"root_completed\":"<<o.views[0].completed<<",\"elapsed_ms\":"<<o.elapsed()<<",\"node_value\":"<<n.q<<",\"network_value\":";
 if(raw_known)out<<raw;else out<<"null";
 out<<",\"solver_generation\":"<<(pool.proof_owner?hxp_generation(pool.proof_owner,index):0)<<",\"context\":["<<key.a<<','<<key.b<<"],\"edges\":[";
 for(size_t i=0;i<n.edges.size();++i){if(i)out<<',';auto& e=n.edges[i];out<<'['<<e.action.q<<','<<e.action.r<<','<<e.logit<<','<<q[i]<<','<<t.value(n,e)<<','<<weights[i]/total<<','<<e.visits<<','<<(i<o.direct_root_credits.size()?o.direct_root_credits[i]:0)<<','<<(e.eligible?1:0)<<']';}
 out<<"],\"exact_prefixes\":[";Board prefix;size_t count=0;
 for(size_t ply=0;ply<o.focus.size();++ply){auto fact=o.game->outcomes.find(gumbel::keys(prefix).first);if(fact!=o.game->outcomes.end()){
  if(count++)out<<',';out<<'['<<ply<<','<<fact->second.winner<<','<<fact->second.distance<<",[";size_t actions=0;
  for(auto& edge:fact->second.edges)if(edge.winner==fact->second.winner && edge.distance==fact->second.distance){if(actions++)out<<',';out<<'['<<edge.action.q<<','<<edge.action.r<<']';}out<<"]]";}prefix.make(o.focus[ply]);}
 out<<"]}";return out.str();
}
}
extern "C" {
HX_API void* hxb_new(int quantum,int pending,int merge,double latency){try{return new inference::Broker(quantum,pending,merge,latency);}catch(const std::exception& e){gumbel::error=e.what();return nullptr;}}
HX_API int hxb_attach(void* p,void* pool,int model){try{return static_cast<inference::Broker*>(p)->attach(*static_cast<owner::Pool*>(pool),model)+1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxb_detach(void* p,int producer){try{static_cast<inference::Broker*>(p)->detach(producer);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxb_model_pending(void* p,int model){return static_cast<inference::Broker*>(p)->model_pending(model);}
HX_API int hxb_workers(void* p,int producer,int count){try{static_cast<inference::Broker*>(p)->workers(producer,count);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxb_start(void* p,double ms){try{static_cast<inference::Broker*>(p)->start(ms);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxb_continuous(void* p){auto& b=*static_cast<inference::Broker*>(p);if(b.started){gumbel::error="Configure continuous mode before starting";return 0;}b.continuous=true;return 1;}
HX_API int hxb_retarget(void* p,int producer,int game,uint64_t expected,const int64_t* cells,int count,uint64_t work,double ms,int samples,int views,double noise){try{static_cast<inference::Broker*>(p)->retarget(producer,game,expected,cells,count,work,ms,samples,views,noise);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxb_release(void* p,int producer,int game,uint64_t expected){try{static_cast<inference::Broker*>(p)->release(producer,game,expected);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxb_replace(void* p,int producer,int game,uint64_t expected,void* source,const char* version,const int64_t* cells,int count,uint64_t work,double ms,int samples,int views,double noise,uint64_t seed){try{if(!source)throw std::runtime_error("Missing replacement graph");static_cast<inference::Broker*>(p)->retarget(producer,game,expected,cells,count,work,ms,samples,views,noise,static_cast<gumbel::Tree*>(source),version,seed);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API const char* hxb_event(void* p){try{return static_cast<inference::Broker*>(p)->event();}catch(const std::exception& e){gumbel::error=e.what();return nullptr;}}
HX_API int hxb_pause(void* p,int paused){try{static_cast<inference::Broker*>(p)->pause(paused!=0);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxb_paused(void* p){try{return static_cast<inference::Broker*>(p)->fenced();}catch(const std::exception& e){gumbel::error=e.what();return -1;}}
HX_API uint64_t hxb_installed(void* p){auto& b=*static_cast<inference::Broker*>(p);std::lock_guard lock(b.mutex);return b.installed_messages;}
HX_API void hxb_cancel(void* p){static_cast<inference::Broker*>(p)->cancel();}
HX_API int hxb_take(void* p,int limit,double wait,uint64_t* token,int* model,void** snapshot){try{return static_cast<inference::Broker*>(p)->take(limit,wait,token,model,snapshot);}catch(const std::exception& e){gumbel::error=e.what();return -1;}}
HX_API int hxb_complete(void* p,uint64_t token,const int64_t* offsets,const int64_t* actions,const double* logits,const double* values){try{static_cast<inference::Broker*>(p)->complete(token,offsets,actions,logits,values);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxb_abort(void* p,uint64_t token){try{static_cast<inference::Broker*>(p)->abort(token);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxb_done(void* p){auto& b=*static_cast<inference::Broker*>(p);std::lock_guard lock(b.mutex);if(!b.error.empty()){gumbel::error=b.error;return -1;}return b.done_locked();}
HX_API int hxb_join(void* p){try{static_cast<inference::Broker*>(p)->join();return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxb_free(void* p){auto& b=*static_cast<inference::Broker*>(p);if(b.started && !b.joined){gumbel::error="Join the native inference service before freeing it";return 0;}for(auto& producer:b.producers)if(producer)producer->pool.inference_owner=nullptr;delete &b;return 1;}
HX_API void hxb_stats(void* p,uint64_t* out){auto& b=*static_cast<inference::Broker*>(p);std::lock_guard lock(b.mutex);
 std::array<uint64_t,10> v{b.created,b.coalesced,b.launched,b.delivered,b.withdrawn,b.batches,b.high_water,uint64_t(b.tasks.size()),uint64_t(b.flights.size()),uint64_t(std::count_if(b.producers.begin(),b.producers.end(),[](const auto& p){return p && !p->done;}))};std::copy(v.begin(),v.end(),out);
}
}
