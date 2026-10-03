#include "hexo.hpp"
#include <stdexcept>
#include <string>

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
