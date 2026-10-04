#pragma once
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace feeding {
// Persistent host workers. Each phase assigns disjoint game stores; public
// control, proof dispatch and GPU admission run only after the phase joins.
class Workers {
 using Work=void(*)(void*,int);
 std::mutex mutex;
 std::condition_variable wake,done;
 std::vector<std::thread> threads;
 Work work=nullptr;void* context=nullptr;
 size_t next=0,count=0,left=0;uint64_t generation=0;
 bool stopping=false,active=false;
 std::exception_ptr error;
 void worker(){
  uint64_t seen=0;std::unique_lock lock(mutex);
  for(;;){
   wake.wait(lock,[&]{return stopping || generation!=seen;});if(stopping)return;
   seen=generation;
   while(next<count){
    int index=int(next++);auto call=work;auto data=context;lock.unlock();
    std::exception_ptr failed;try{call(data,index);}catch(...){failed=std::current_exception();}
    lock.lock();if(failed && !error)error=failed;
   }
   if(!--left){active=false;done.notify_one();}
  }
 }
 void stop(){
  {std::lock_guard lock(mutex);stopping=true;}wake.notify_all();
  for(auto& thread:threads)if(thread.joinable())thread.join();
 }
public:
 explicit Workers(int count=1){
  if(count<1 || count>16)throw std::runtime_error("Invalid native host worker count");
  if(count==1)return;
  try{for(int i=0;i<count;++i)threads.emplace_back([this]{worker();});}
  catch(...){stop();throw;}
 }
 ~Workers(){stop();}
 int size()const{return threads.empty()?1:int(threads.size());}
 void run(int jobs,Work call,void* data){
  if(!jobs)return;
  if(threads.empty()){for(int i=0;i<jobs;++i)call(data,i);return;}
  std::unique_lock lock(mutex);
  if(active || stopping)throw std::runtime_error("Native host phases must be serialized");
  context=data;work=call;count=jobs;next=0;left=threads.size();error=nullptr;active=true;++generation;
  wake.notify_all();done.wait(lock,[&]{return !active;});if(error)std::rethrow_exception(error);
 }
};
}
