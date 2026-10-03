#include "hexo.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace gumbel { extern thread_local std::string error; }
extern "C" int hxg_encode(void*,int,uint8_t*,int,int64_t*,int64_t*);
extern "C" int hxg_legal(void*,int,int64_t*);

// Two-pass pending-leaf encoder. Each info row holds the crop side, hxg_encode's nine fields,
// then offsets into the contiguous plane and legal buffers. Null planes query the layout.
// The caller owns every pending request for both passes; this does not select or fulfill leaves.
extern "C" HX_API int hxg_encode_many(void* const* trees,const int* ids,int count,int64_t* info,
 uint8_t* planes,int64_t plane_capacity,int64_t* cells,int64_t* actions,int64_t legal_capacity){
 try {
  if(count<0 || (count && (!trees || !ids || !info)) || plane_capacity<0 || legal_capacity<0)
   throw std::runtime_error("Invalid leaf batch buffers");
  if(planes && (!cells || !actions))throw std::runtime_error("Missing leaf batch outputs");
  int64_t bytes=0,legal=0;
  for(int i=0;i<count;++i){
   if(!trees[i])throw std::runtime_error("Missing leaf batch tree");
   int64_t* row=info+12*int64_t(i);
   if(!planes){
    row[0]=hxg_encode(trees[i],ids[i],nullptr,0,nullptr,row+1);
    if(!row[0])return 0;
    row[10]=bytes;row[11]=legal;
   } else if(row[10]!=bytes || row[11]!=legal)throw std::runtime_error("Invalid leaf batch offsets");
   if(row[0]==-2){
    if(planes && hxg_encode(trees[i],ids[i],nullptr,0,nullptr,nullptr)!=-2)
     throw std::runtime_error("Changed leaf batch layout");
    continue;
   }
   if(row[0]<1 || row[0]>256 || row[1]<1)throw std::runtime_error("Invalid leaf batch layout");
   if(planes && hxg_legal(trees[i],ids[i],nullptr)!=row[1])
    throw std::runtime_error("Changed leaf batch legal list");
   bytes+=8*row[0]*row[0];legal+=row[1];
   if(planes && (bytes>plane_capacity || legal>legal_capacity))
    throw std::runtime_error("Leaf batch buffer too small");
  }
  if(planes)for(int i=0;i<count;++i){
   int64_t* row=info+12*int64_t(i);
   if(row[0]==-2)continue;
   int side=hxg_encode(trees[i],ids[i],planes+row[10],int(8*row[0]*row[0]),cells+row[11],row+1);
   if(!side)return 0;
   if(side!=row[0])throw std::runtime_error("Changed leaf batch crop");
   hxg_legal(trees[i],ids[i],actions+2*row[11]);
  }
  return 1;
 } catch(const std::exception& e){gumbel::error=e.what();return 0;}
}

namespace packing {
struct Group { int side;std::vector<int> rows; };
// Snapshot the pending requests while their trees are alive. Afterwards this object owns
// all encoding and output mappings, so a cancelled subscriber need not keep a tree alive.
struct Batch {
 std::vector<int64_t> info,offsets,cells,actions;
 std::vector<uint8_t> planes,decoded;
 std::vector<double> logits,values;
 std::vector<Group> groups;
 Batch(void* const* trees,const int* requests,int count,int merge_cells):info(12*int64_t(count)),offsets(count+1),decoded(count) {
  if(count<1 || merge_cells<0)throw std::runtime_error("Invalid packed batch size");
  if(!hxg_encode_many(trees,requests,count,info.data(),nullptr,0,nullptr,nullptr,0))throw std::runtime_error(gumbel::error);
  int64_t bytes=0,legal=0;std::map<int,std::vector<int>> by_size;
  for(int i=0;i<count;++i){auto* row=info.data()+12*int64_t(i);offsets[i]=legal;
   if(row[0]==-2){decoded[i]=1;continue;}
   bytes+=8*row[0]*row[0];legal+=row[1];by_size[int(row[0])].push_back(i);
  }
  offsets[count]=legal;planes.resize(bytes);cells.resize(legal);actions.resize(2*legal);logits.resize(legal);values.resize(legal);
  if(legal && !hxg_encode_many(trees,requests,count,info.data(),planes.data(),bytes,cells.data(),actions.data(),legal))
   throw std::runtime_error(gumbel::error);
  // Match the evaluator's adjacent-size merge, retaining every row's original mapping.
  for(auto it=by_size.begin();it!=by_size.end();){auto next=std::next(it);if(next==by_size.end())break;
   if(int64_t(it->second.size())*(next->first*next->first-it->first*it->first)<merge_cells){
    next->second.insert(next->second.begin(),it->second.begin(),it->second.end());by_size.erase(it);
   }
   it=next;
  }
  for(auto& [side,rows]:by_size)groups.push_back({side,std::move(rows)});
 }
 Group& group(int index){if(index<0 || index>=int(groups.size()))throw std::runtime_error("Invalid packed batch group");return groups[index];}
 void pack(int index,uint8_t* output,int64_t capacity){auto& g=group(index);int area=g.side*g.side;
  int64_t bytes=8*int64_t(area)*g.rows.size();
  if(!output || capacity<bytes)throw std::runtime_error("Packed plane buffer too small");
  std::memset(output,0,size_t(bytes));
  for(size_t i=0;i<g.rows.size();++i){auto* row=info.data()+12*int64_t(g.rows[i]);int side=int(row[0]);
   const auto* src=planes.data()+row[10];auto* dst=output+8*int64_t(area)*i;
   if(side==g.side)std::memcpy(dst,src,8*size_t(area));
   else for(int channel=0;channel<8;++channel)for(int y=0;y<side;++y)
    std::memcpy(dst+channel*area+y*g.side,src+channel*side*side+y*side,size_t(side));
  }
 }
 void decode(int index,int start,int count,const float* output,int64_t capacity){auto& g=group(index);int stride=g.side*g.side+2;
  if(start<0 || count<1 || start>int(g.rows.size())-count || !output || capacity<int64_t(count)*stride)
   throw std::runtime_error("Invalid packed prediction buffers");
  // Validate before writing, including grid cells outside the legal mask.
  for(int64_t i=0;i<int64_t(count)*stride;++i)if(!std::isfinite(output[i]))throw std::runtime_error("Nonfinite dense model predictions");
  for(int i=start;i<start+count;++i)if(decoded[g.rows[i]])throw std::runtime_error("Duplicate packed row completion");
  for(int i=0;i<count;++i){int id=g.rows[start+i];auto* row=info.data()+12*int64_t(id);int side=int(row[0]);
   const float* prediction=output+int64_t(i)*stride;double q=std::tanh(double(prediction[stride-1])/2);
   double far=double(prediction[stride-2])-(row[9]?std::log(double(row[9])):0);
   for(int64_t j=offsets[id];j<offsets[id+1];++j){int64_t cell=cells[j];
    logits[j]=cell<0?far:double(prediction[cell/side*g.side+cell%side]);values[j]=q;
   }
   decoded[id]=1;
  }
 }
 void outputs(void** out){if(!out || std::find(decoded.begin(),decoded.end(),0)!=decoded.end())throw std::runtime_error("Incomplete packed batch");
  out[0]=offsets.data();out[1]=actions.data();out[2]=logits.data();out[3]=values.data();
 }
};
}

extern "C" {
HX_API void* hxgp_new(void* const* trees,const int* requests,int count,int merge_cells){try{
 if(count<1 || !trees || !requests)throw std::runtime_error("Invalid packed batch inputs");
 return new packing::Batch(trees,requests,count,merge_cells);
}catch(const std::exception& e){gumbel::error=e.what();return nullptr;}}
HX_API void hxgp_free(void* p){delete static_cast<packing::Batch*>(p);}
HX_API int hxgp_groups(void* p){return int(static_cast<packing::Batch*>(p)->groups.size());}
HX_API int hxgp_group(void* p,int index,int64_t* out){try{
 if(!out)throw std::runtime_error("Missing packed group output");
 auto& g=static_cast<packing::Batch*>(p)->group(index);out[0]=g.side;out[1]=int64_t(g.rows.size());return 1;
}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxgp_pack(void* p,int index,uint8_t* out,int64_t capacity){try{static_cast<packing::Batch*>(p)->pack(index,out,capacity);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
HX_API int hxgp_decode(void* p,int index,int start,int count,const float* out,int64_t capacity){try{static_cast<packing::Batch*>(p)->decode(index,start,count,out,capacity);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
// Borrowed output buffers remain valid until hxgp_free. No tree is accessed by decoding.
HX_API int hxgp_outputs(void* p,void** out){try{static_cast<packing::Batch*>(p)->outputs(out);return 1;}catch(const std::exception& e){gumbel::error=e.what();return 0;}}
}
