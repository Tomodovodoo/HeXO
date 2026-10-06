#include "hexo.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <map>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace gumbel { extern thread_local std::string error; }
extern "C" int hxg_encode(void*,int,uint8_t*,int,int64_t*,int64_t*);
extern "C" int hxg_encode_rect(void*,int,uint8_t*,int,int64_t*,int64_t*);
extern "C" int hxg_legal(void*,int,int64_t*);

// Two-pass pending-leaf encoder. Each info row holds the crop side, hxg_encode's nine fields,
// then offsets into the contiguous plane and legal buffers. Null planes query the layout.
// The caller owns every pending request for both passes; this does not select or fulfill leaves.
static int encode_many(void* const* trees,const int* ids,int count,int64_t* info,
 uint8_t* planes,int64_t plane_capacity,int64_t* cells,int64_t* actions,int64_t legal_capacity,bool rectangular){
 try {
  if(count<0 || (count && (!trees || !ids || !info)) || plane_capacity<0 || legal_capacity<0)
   throw std::runtime_error("Invalid leaf batch buffers");
  if(planes && (!cells || !actions))throw std::runtime_error("Missing leaf batch outputs");
  int64_t bytes=0,legal=0;int fields=rectangular?13:12;auto encode=rectangular?hxg_encode_rect:hxg_encode;
  for(int i=0;i<count;++i){
   if(!trees[i])throw std::runtime_error("Missing leaf batch tree");
   int64_t* row=info+fields*int64_t(i);
   if(!planes){
    std::array<int64_t,10> metadata{};
    row[0]=encode(trees[i],ids[i],nullptr,0,nullptr,metadata.data());
    std::copy(metadata.begin(),metadata.begin()+9,row+1);if(rectangular)row[12]=metadata[9];
    if(!row[0])return 0;
    row[10]=bytes;row[11]=legal;
   } else if(row[10]!=bytes || row[11]!=legal)throw std::runtime_error("Invalid leaf batch offsets");
   if(row[0]==-2){
    if(planes && encode(trees[i],ids[i],nullptr,0,nullptr,nullptr)!=-2)
     throw std::runtime_error("Changed leaf batch layout");
    continue;
   }
   if(row[0]<1 || row[0]>256 || row[1]<1)throw std::runtime_error("Invalid leaf batch layout");
   if(planes && hxg_legal(trees[i],ids[i],nullptr)!=row[1])
    throw std::runtime_error("Changed leaf batch legal list");
   int64_t width=rectangular?row[12]:row[0];
   if(width<1 || width>256)throw std::runtime_error("Invalid leaf batch width");
   bytes+=8*row[0]*width;legal+=row[1];
   if(planes && (bytes>plane_capacity || legal>legal_capacity))
    throw std::runtime_error("Leaf batch buffer too small");
  }
  if(planes)for(int i=0;i<count;++i){
   int64_t* row=info+fields*int64_t(i);
   if(row[0]==-2)continue;
   std::array<int64_t,10> metadata{};int64_t width=rectangular?row[12]:row[0];
   int side=encode(trees[i],ids[i],planes+row[10],int(8*row[0]*width),cells+row[11],metadata.data());
   if(!side)return 0;
   if(side!=row[0] || (rectangular && metadata[9]!=row[12]))throw std::runtime_error("Changed leaf batch crop");
   hxg_legal(trees[i],ids[i],actions+2*row[11]);
  }
  return 1;
 } catch(const std::exception& e){gumbel::error=e.what();return 0;}
}

extern "C" HX_API int hxg_encode_many(void* const* trees,const int* ids,int count,int64_t* info, uint8_t* planes,int64_t plane_capacity,int64_t* cells,int64_t* actions,int64_t legal_capacity){return encode_many(trees,ids,count,info,planes,plane_capacity,cells,actions,legal_capacity,false);}
extern "C" HX_API int hxg_encode_many_rect(void* const* trees,const int* ids,int count,int64_t* info, uint8_t* planes,int64_t plane_capacity,int64_t* cells,int64_t* actions,int64_t legal_capacity){return encode_many(trees,ids,count,info,planes,plane_capacity,cells,actions,legal_capacity,true);}

namespace packing {
struct Group { int side,width;std::vector<int> rows; };
// Recent warmed GPU service observations fit milliseconds per executed cell
// and per launch. The bounded window adapts to the model/device/load without
// making a first-use graph capture look like ordinary inference cost.
struct Costs {
 std::array<std::array<double,3>,64> samples{};int count=0,next=0;
 double cc=0,cl=0,ll=0,cy=0,ly=0,cell=0,launch=0;bool fitted=false;
 void learn(const double* rows,int size){
  if(size<0 || (size && !rows))throw std::runtime_error("Missing inference cost observations");
  for(int i=0;i<size;++i)for(int j=0;j<3;++j)if(!std::isfinite(rows[3*i+j]) || rows[3*i+j]<=0)
   throw std::runtime_error("Invalid inference cost observation");
  auto add=[&](const std::array<double,3>& s,double sign){double c=s[0],l=s[1],y=s[2];cc+=sign*c*c;cl+=sign*c*l;ll+=sign*l*l;cy+=sign*c*y;ly+=sign*l*y;};
  for(int i=0;i<size;++i){if(count==int(samples.size()))add(samples[next],-1);else ++count;
   samples[next]={rows[3*i],rows[3*i+1],rows[3*i+2]};add(samples[next],1);next=(next+1)%samples.size();}
  double det=cc*ll-cl*cl;if(count<4 || det<=1e-12*cc*ll)return;
  double a=(cy*ll-ly*cl)/det,b=(ly*cc-cy*cl)/det;
  if(b<0){b=0;a=cy/cc;}
  if(a>0 && std::isfinite(a) && std::isfinite(b)){cell=a;launch=b;fitted=true;}
 }
};
// Snapshot the pending requests while their trees are alive. Afterwards this object owns
// all encoding and output mappings, so a cancelled subscriber need not keep a tree alive.
struct Batch {
 struct Snapshot {};
 struct Row { Batch* storage;int index; };
 std::atomic<unsigned> owners{1};
 std::vector<Row> sources;
 std::vector<Batch*> retained;
 std::vector<int64_t> info,offsets,cells,actions;
 std::vector<uint8_t> planes,decoded;
 std::vector<double> logits,values;
 std::vector<Group> groups;bool sealed=false,mixed=false;
 void retain(){owners.fetch_add(1,std::memory_order_relaxed);}
 void release(){if(owners.fetch_sub(1,std::memory_order_acq_rel)==1)delete this;}
 ~Batch(){for(auto* storage:retained)storage->release();}
 Row source(int id){return sources.empty()?Row{this,id}:sources[id];}
 const uint8_t* row_planes(int id){auto row=source(id);return row.storage->planes.data()+row.storage->info[13*int64_t(row.index)+10];}
 const int64_t* row_cells(int id){auto row=source(id);return row.storage->cells.data()+row.storage->offsets[row.index];}
 Batch(void* const* trees,const int* requests,int count,int merge_cells,bool rectangular=false):info(13*int64_t(count)),offsets(count+1),decoded(count) {
  if(count<1 || merge_cells<0)throw std::runtime_error("Invalid packed batch size");
  auto encode=rectangular?hxg_encode_rect:hxg_encode;
  for(int i=0;i<count;++i){auto* row=info.data()+13*int64_t(i);std::array<int64_t,10> metadata{};
   if(!trees[i])throw std::runtime_error("Missing leaf batch tree");
   row[0]=encode(trees[i],requests[i],nullptr,0,nullptr,metadata.data());
   if(!row[0])throw std::runtime_error(gumbel::error);
   std::copy(metadata.begin(),metadata.begin()+9,row+1);row[12]=rectangular?metadata[9]:row[0];
  }
  int64_t bytes=0,legal=0;std::map<std::pair<int,int>,std::vector<int>> by_size;
  for(int i=0;i<count;++i){auto* row=info.data()+13*int64_t(i);offsets[i]=legal;
   if(row[0]==-2){decoded[i]=1;continue;}
   row[10]=bytes;row[11]=legal;bytes+=8*row[0]*row[12];legal+=row[1];by_size[{int(row[0]),int(row[12])}].push_back(i);
  }
  offsets[count]=legal;planes.resize(bytes);cells.resize(legal);actions.resize(2*legal);logits.resize(legal);values.resize(legal);
  for(int i=0;i<count;++i){auto* row=info.data()+13*int64_t(i);if(row[0]==-2)continue;
   std::array<int64_t,10> metadata{};
   int height=encode(trees[i],requests[i],planes.data()+row[10],int(8*row[0]*row[12]),cells.data()+row[11],metadata.data());
   if(height!=row[0] || (rectangular && metadata[9]!=row[12]))throw std::runtime_error("Changed leaf batch crop");
   hxg_legal(trees[i],requests[i],actions.data()+2*row[11]);
  }
  // Match the evaluator's adjacent-size merge, retaining every row's original mapping.
  regroup(std::move(by_size),merge_cells);
 }
 // Keep immutable encoding alive instead of recopying it for each forward.
 // Flatten references to original snapshots so repeated combinations do not
 // retain intermediate prediction buffers or create chains of batch owners.
 Batch(void* const* sources,const int* rows,int count,int merge_cells,Snapshot):info(13*int64_t(count)),offsets(count+1),decoded(count){
  if(count<1 || merge_cells<0)throw std::runtime_error("Invalid snapshot combination");
  std::map<std::pair<int,int>,std::vector<int>> by_size;
  this->sources.reserve(count);retained.reserve(count);
  int64_t legal=0;
  for(int i=0;i<count;++i){
   if(!sources[i])throw std::runtime_error("Missing row snapshot");
   auto& source=*static_cast<Batch*>(sources[i]);int id=rows[i];
   if(id<0 || id>=int(source.decoded.size()))throw std::runtime_error("Invalid snapshot row");
   const auto* original=source.info.data()+13*int64_t(id);auto* row=info.data()+13*int64_t(i);
   std::copy(original,original+13,row);row[10]=0;row[11]=legal;offsets[i]=legal;
   auto origin=source.source(id);this->sources.push_back(origin);
   if(std::find(retained.begin(),retained.end(),origin.storage)==retained.end())retained.push_back(origin.storage);
   if(row[0]==-2){decoded[i]=1;continue;}
   int64_t first=source.offsets[id],last=source.offsets[id+1];legal+=last-first;
   by_size[{int(row[0]),int(row[12])}].push_back(i);
  }
  actions.reserve(2*legal);
  for(int i=0;i<count;++i){auto& source=*static_cast<Batch*>(sources[i]);int id=rows[i];
   actions.insert(actions.end(),source.actions.begin()+2*source.offsets[id],source.actions.begin()+2*source.offsets[id+1]);}
  offsets[count]=legal;logits.resize(legal);values.resize(legal);
  regroup(std::move(by_size),merge_cells);
  // Nothing after these increments throws. The caller keeps source handles
  // alive for construction; later owner and device releases may race.
  for(auto* storage:retained)storage->retain();
 }
 void regroup(std::map<std::pair<int,int>,std::vector<int>> by_size,int merge_cells){
  mixed=by_size.size()>1;
  for(auto it=by_size.begin();it!=by_size.end();){auto next=std::next(it);if(next==by_size.end())break;
   if(next->first.first>=it->first.first && next->first.second>=it->first.second && int64_t(it->second.size())*(next->first.first*next->first.second-it->first.first*it->first.second)<merge_cells){
    next->second.insert(next->second.begin(),it->second.begin(),it->second.end());by_size.erase(it);
   }
   it=next;
  }
  for(auto& [shape,rows]:by_size)groups.push_back({shape.first,shape.second,std::move(rows)});
 }
 void plan(const Costs& cost,const int64_t* limits,int size,int step,bool rectangular=false){
  if(sealed)throw std::runtime_error("Packed layout already submitted");
  if(size<0 || (size && !limits) || step<1)throw std::runtime_error("Missing inference capture limits");
  std::map<std::pair<int,int>,int> caps;int fields=rectangular?3:2;
  for(int i=0;i<size;++i){int64_t s=limits[fields*i],w=rectangular?limits[fields*i+1]:s,cap=limits[fields*i+fields-1];
   if(s<1 || s>256 || w<1 || w>256 || cap<1 || cap>128 || (cap&(cap-1)))throw std::runtime_error("Invalid inference capture limit");
   caps[{int(s),int(w)}]=int(cap);}
  std::map<std::pair<int,int>,std::vector<int>> by_size;
  for(int i=0;i<int(decoded.size());++i){int side=int(info[13*int64_t(i)]);if(side<1)continue;
   if(decoded[i])throw std::runtime_error("Packed prediction already completed");
   by_size[{side,int(info[13*int64_t(i)+12])}].push_back(i);}
  groups.clear();for(auto& [shape,rows]:by_size)groups.push_back({shape.first,shape.second,std::move(rows)});
  if(!cost.fitted)return;
  auto units=[&](int rows,int side,int width,int limit){double cells=0;int launches=0;
   while(rows){int chunk=std::min(rows,step);rows-=chunk;
    while(chunk>limit){cells+=double(limit)*side*width;++launches;chunk-=limit;}
    if(limit>=128 && chunk>64 && chunk<=96){cells+=64.*side*width;++launches;chunk-=64;}
    if(limit>=64 && chunk>32 && chunk<=(side==40 && width==40?56:48)){cells+=32.*side*width;++launches;chunk-=32;}
    if(chunk>16 && chunk<=24){cells+=16.*side*width;++launches;chunk-=16;}
    if(chunk){int cap=1;while(cap<chunk)cap*=2;cells+=double(cap)*side*width;++launches;}
   }
   return std::pair{cells,launches};
  };
  if(rectangular && std::any_of(groups.begin(),groups.end(),[](const auto& g){return g.side!=g.width;})){
   auto score=[&](int rows,int height,int width){auto cap=caps.find({height,width});
    if(cap==caps.end())return std::numeric_limits<double>::infinity();
    auto [cells,launches]=units(rows,height,width,cap->second);return cost.cell*cells+cost.launch*launches;};
   // Shapes need not form a containment chain. Merge any two groups whose
   // bounding canvas has lower measured execution cost than separate forwards.
   for(;;){double saving=0;int first=-1,second=-1;
    for(int i=0;i<int(groups.size());++i)for(int j=i+1;j<int(groups.size());++j){
     auto& a=groups[i];auto& b=groups[j];
     double gain=score(int(a.rows.size()),a.side,a.width)+score(int(b.rows.size()),b.side,b.width)
       -score(int(a.rows.size()+b.rows.size()),std::max(a.side,b.side),std::max(a.width,b.width));
     if(gain>saving){saving=gain;first=i;second=j;}
    }
    if(first<0)break;
    auto& a=groups[first];auto& b=groups[second];a.side=std::max(a.side,b.side);a.width=std::max(a.width,b.width);
    a.rows.insert(a.rows.end(),b.rows.begin(),b.rows.end());groups.erase(groups.begin()+second);
   }
   return;
  }
  // A partition may merge adjacent sizes into its largest canvas. Score the
  // actual capture segments, including padding; retain every original row.
  std::vector<double> best(groups.size()+1,std::numeric_limits<double>::infinity());std::vector<int> before(best.size());best[0]=0;
  for(int end=1;end<int(best.size());++end){int rows=0;
   for(int start=end-1;start>=0;--start){rows+=int(groups[start].rows.size());
    auto& target=groups[end-1];auto cap=caps.find({target.side,target.width});if(cap==caps.end()){
     if(start==end-1){best[end]=best[start];before[end]=start;}break;
    }
    if(!caps.contains({groups[start].side,groups[start].width}) || groups[start].side>target.side || groups[start].width>target.width)break;
    auto [cells,launches]=units(rows,target.side,target.width,cap->second);double score=best[start]+cost.cell*cells+cost.launch*launches;
    if(score<best[end]){best[end]=score;before[end]=start;}
   }
  }
  std::vector<Group> planned;
  for(int end=int(groups.size());end;){int start=before[end];Group g{groups[end-1].side,groups[end-1].width,{}};
   for(int i=start;i<end;++i)g.rows.insert(g.rows.end(),groups[i].rows.begin(),groups[i].rows.end());
   planned.push_back(std::move(g));end=start;
  }
  std::reverse(planned.begin(),planned.end());groups=std::move(planned);
 }
 Group& group(int index){if(index<0 || index>=int(groups.size()))throw std::runtime_error("Invalid packed batch group");return groups[index];}
 void pack_range(int index,int start,int count,uint8_t* output,int64_t capacity){auto& g=group(index);int area=g.side*g.width;
  if(start<0 || count<1 || start>int(g.rows.size())-count)throw std::runtime_error("Invalid packed feature rows");
  int64_t bytes=8*int64_t(area)*count;
  if(!output || capacity<bytes)throw std::runtime_error("Packed plane buffer too small");
  sealed=true;
  for(int i=0;i<count;++i){auto* row=info.data()+13*int64_t(g.rows[start+i]);int side=int(row[0]),width=int(row[12]);
   const auto* src=row_planes(g.rows[start+i]);auto* dst=output+8*int64_t(area)*i;
   if(side==g.side && width==g.width)std::memcpy(dst,src,8*size_t(area));
   else {std::memset(dst,0,8*size_t(area));
    for(int channel=0;channel<8;++channel)for(int y=0;y<side;++y)
     std::memcpy(dst+channel*area+y*g.width,src+channel*side*width+y*width,size_t(width));}
  }
 }
 void pack(int index,uint8_t* output,int64_t capacity){pack_range(index,0,int(group(index).rows.size()),output,capacity);}
 // The browser ONNX model consumes LineFeatures rather than the eight raw
 // planes used by the native fused backend. Preserve that input contract in
 // one compiled call, including crop masks, padded canvases and zero channels.
 void features(int index,int start,int count,float* output,int64_t capacity){
  auto& g=group(index);int side=g.side,width=g.width,area=side*width;int64_t stride=20*int64_t(area);
  if(!output || count<1 || capacity<stride*count)throw std::runtime_error("Packed feature buffer too small");
  std::vector<uint8_t> input(8*int64_t(area)*count);pack_range(index,start,count,input.data(),int64_t(input.size()));
  std::fill(output,output+stride*count,0.f);
  std::vector<uint8_t> open(2*area),best(2*area);
  auto inside=[&](int x,int y){return x>=0 && y>=0 && x<width && y<side;};
  constexpr int axes[3][2]={{1,0},{0,1},{1,-1}};
  for(int row=0;row<count;++row){auto* src=input.data()+8*int64_t(area)*row;auto* out=output+stride*row;
   auto* own=src;auto* opp=src+area;auto* mask=src+3*area;std::fill(best.begin(),best.end(),0);
   for(int channel=0;channel<8;++channel)for(int j=0;j<area;++j)out[channel*area+j]=float(src[channel*area+j]*mask[j]);
   for(int axis=0;axis<3;++axis){int dx=axes[axis][0],dy=axes[axis][1];std::fill(open.begin(),open.end(),0);
    for(int y=0;y<side;++y)for(int x=0;x<width;++x){int o=0,p=0,m=0;
     for(int i=0;i<6;++i){int u=x+i*dx,v=y+i*dy;if(!inside(u,v))break;int j=v*width+u;o+=own[j];p+=opp[j];m+=mask[j];}
     if(m==6){int j=y*width+x;if(!p)open[j]=uint8_t(o);if(!o)open[area+j]=uint8_t(p);}
    }
    for(int player=0;player<2;++player)for(int y=0;y<side;++y)for(int x=0;x<width;++x){uint8_t b=0;
     for(int i=0;i<6;++i){int u=x-i*dx,v=y-i*dy;if(inside(u,v))b=std::max(b,open[player*area+v*width+u]);}
     int j=y*width+x;out[(8+3*player+axis)*area+j]=(float(b)/6.f)*float(mask[j]);best[player*area+j]=std::max(best[player*area+j],b);
    }
   }
   for(int j=0;j<area;++j)if(mask[j] && !own[j] && !opp[j]){
    out[14*area+j]=best[j]>=4;out[15*area+j]=best[area+j]>=4;
    out[16*area+j]=best[j]>=5;out[17*area+j]=best[area+j]>=5;
   }
  }
 }
 void decode_row(const Group& g,int id,const float* policy,float far_logit,float value){
  auto* row=info.data()+13*int64_t(id);int width=int(row[12]);double q=std::tanh(double(value)/2);
  double far=double(far_logit)-(row[9]?std::log(double(row[9])):0);
  auto* legal=row_cells(id);
  for(int64_t j=offsets[id];j<offsets[id+1];++j){int64_t cell=legal[j-offsets[id]];
   logits[j]=cell<0?far:double(policy[cell/width*g.width+cell%width]);values[j]=q;
  }decoded[id]=1;
 }
 void decode(int index,int start,int count,const float* output,int64_t capacity){auto& g=group(index);int stride=g.side*g.width+2;
  if(start<0 || count<1 || start>int(g.rows.size())-count || !output || capacity<int64_t(count)*stride)
   throw std::runtime_error("Invalid packed prediction buffers");
  // Validate before writing, including grid cells outside the legal mask.
  for(int64_t i=0;i<int64_t(count)*stride;++i)if(!std::isfinite(output[i]))throw std::runtime_error("Nonfinite dense model predictions");
  for(int i=start;i<start+count;++i)if(decoded[g.rows[i]])throw std::runtime_error("Duplicate packed row completion");
  for(int i=0;i<count;++i){const float* prediction=output+int64_t(i)*stride;decode_row(g,g.rows[start+i],prediction,prediction[stride-2],prediction[stride-1]);}
 }
 void decode_split(int index,int start,int count,const float* policy,const float* far,const float* value){auto& g=group(index);int area=g.side*g.width;
  if(start<0 || count<1 || start>int(g.rows.size())-count || !policy || !far || !value)throw std::runtime_error("Invalid split prediction buffers");
  for(int64_t i=0;i<int64_t(count)*area;++i)if(!std::isfinite(policy[i]))throw std::runtime_error("Nonfinite dense model predictions");
  for(int i=0;i<count;++i){
   if(!std::isfinite(far[i]) || !std::isfinite(value[i]))throw std::runtime_error("Nonfinite dense model predictions");
   if(decoded[g.rows[start+i]])throw std::runtime_error("Duplicate packed row completion");
  }
  for(int i=0;i<count;++i)decode_row(g,g.rows[start+i],policy+int64_t(i)*area,far[i],value[i]);
 }
 void outputs(void** out){if(!out || std::find(decoded.begin(),decoded.end(),0)!=decoded.end())throw std::runtime_error("Incomplete packed batch");
  out[0]=offsets.data();out[1]=actions.data();out[2]=logits.data();out[3]=values.data();
 }
};
}

extern "C" {
HX_API void* hxgp_costs_new(){try{return new packing::Costs;}catch(const std::exception& e){gumbel::error=e.what();return nullptr;}}
HX_API void hxgp_costs_free(void* p){delete static_cast<packing::Costs*>(p);}
HX_API int hxgp_costs_learn(void* p,const double* rows,int count){try{auto& c=*static_cast<packing::Costs*>(p);c.learn(rows,count);return c.fitted?2:1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API void hxgp_costs_stats(void* p,double* out){auto& c=*static_cast<packing::Costs*>(p);out[0]=c.count;out[1]=c.cell;out[2]=c.launch;out[3]=c.fitted;}
HX_API int hxgp_plan(void* p,void* costs,const int64_t* limits,int count,int step){try{static_cast<packing::Batch*>(p)->plan(*static_cast<packing::Costs*>(costs),limits,count,step);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxgp_plan_rect(void* p,void* costs,const int64_t* limits,int count,int step){try{static_cast<packing::Batch*>(p)->plan(*static_cast<packing::Costs*>(costs),limits,count,step,true);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API void* hxgp_new(void* const* trees,const int* requests,int count,int merge_cells){try{
 if(count<1 || !trees || !requests)throw std::runtime_error("Invalid packed batch inputs");
 return new packing::Batch(trees,requests,count,merge_cells);
}catch(const std::exception& e){gumbel::error=e.what();return nullptr;}}
HX_API void* hxgp_new_rect(void* const* trees,const int* requests,int count,int merge_cells){try{
 if(count<1 || !trees || !requests)throw std::runtime_error("Invalid packed batch inputs");
 return new packing::Batch(trees,requests,count,merge_cells,true);
}catch(const std::exception& e){gumbel::error=e.what();return nullptr;}}
HX_API void* hxgp_combine(void* const* sources,const int* rows,int count,int merge_cells){try{
 if(!sources || !rows)throw std::runtime_error("Missing snapshot combination");
 return new packing::Batch(sources,rows,count,merge_cells,packing::Batch::Snapshot{});
}catch(const std::exception& e){gumbel::error=e.what();return nullptr;}}
HX_API void hxgp_free(void* p){if(p)static_cast<packing::Batch*>(p)->release();}
HX_API int hxgp_groups(void* p){return int(static_cast<packing::Batch*>(p)->groups.size());}
HX_API int hxgp_mixed(void* p){return static_cast<packing::Batch*>(p)->mixed;}
HX_API int hxgp_group(void* p,int index,int64_t* out){try{
 if(!out)throw std::runtime_error("Missing packed group output");
 auto& g=static_cast<packing::Batch*>(p)->group(index);if(g.width!=g.side)throw std::runtime_error("Rectangular group requires hxgp_shape");out[0]=g.side;out[1]=int64_t(g.rows.size());return 1;
}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxgp_shape(void* p,int index,int64_t* out){try{
 if(!out)throw std::runtime_error("Missing packed shape output");
 auto& g=static_cast<packing::Batch*>(p)->group(index);out[0]=g.side;out[1]=g.width;out[2]=int64_t(g.rows.size());return 1;
}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxgp_pack(void* p,int index,uint8_t* out,int64_t capacity){try{static_cast<packing::Batch*>(p)->pack(index,out,capacity);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxgp_features(void* p,int index,int start,int count,float* out,int64_t capacity){try{static_cast<packing::Batch*>(p)->features(index,start,count,out,capacity);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxgp_decode_split(void* p,int index,int start,int count,const float* policy,const float* far,const float* value){try{static_cast<packing::Batch*>(p)->decode_split(index,start,count,policy,far,value);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxgp_decode(void* p,int index,int start,int count,const float* out,int64_t capacity){try{static_cast<packing::Batch*>(p)->decode(index,start,count,out,capacity);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
// Borrowed output buffers remain valid until hxgp_free. No tree is accessed by decoding.
HX_API int hxgp_outputs(void* p,void** out){try{static_cast<packing::Batch*>(p)->outputs(out);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
}
