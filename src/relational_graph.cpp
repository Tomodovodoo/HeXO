// Exact graph construction for relational_encoder.py, independent of search.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif
using I = int64_t;
using Point = std::array<I,2>;
struct Hash { size_t operator()(const Point& p) const { return uint64_t(p[0])*0x9e3779b97f4a7c15ULL ^ (uint64_t(p[1])*0xbf58476d1ce4e5b9ULL); } };
using Map = std::unordered_map<Point,I,Hash>;
struct Graph { std::array<std::vector<I>,10> a; std::vector<float> f; };
thread_local std::string error;
static constexpr I axes[3][2]={{1,0},{0,1},{1,-1}};
static constexpr I neighbors[6][2]={{1,0},{0,1},{-1,1},{-1,0},{0,-1},{1,-1}};
static I distance(Point a,Point b) { I q=a[0]-b[0],r=a[1]-b[1]; return std::max({std::abs(q),std::abs(r),std::abs(q+r)}); }
static void edge(std::vector<I>& v,I a,I b,I t,I d,I s) { v.insert(v.end(),{a,b,t,d,s}); }
API const char* hgr_error() { return error.c_str(); }
API void hgr_free(Graph* g) { delete g; }
API const I* hgr_array(Graph* g,int id,I* length) { auto& v=g->a.at(id); *length=I(v.size()); return v.data(); }
API const float* hgr_features(Graph* g,I* length) { *length=I(g->f.size()); return g->f.data(); }
API Graph* hgr_build(const I* stones,I ns,const I* actions,I na,I ng,int player,const float* phase,I maxnodes,I maxedges) {
 Graph* g=nullptr;
 try {
  if(ns<0||na<0||ng<1) throw std::runtime_error("Invalid graph dimensions");
  if(maxnodes>=0&&ns+na+ng>maxnodes) throw std::runtime_error("Stone/legal nodes exceed node budget");
  std::vector<std::array<I,3>> sorted;
  for(I i=0;i<ns;i++) sorted.push_back({stones[3*i],stones[3*i+1],stones[3*i+2]});
  std::sort(sorted.begin(),sorted.end());
  Map occupied,ids; occupied.reserve(ns*2); ids.reserve((ns+na)*2);
  std::set<std::array<I,3>> keys;
  for(I i=0;i<ns;i++) { auto c=sorted[i]; Point p={c[0],c[1]}; occupied[p]=c[2];ids[p]=i;
   for(I ax=0;ax<3;ax++) for(I k=0;k<6;k++) keys.insert({p[0]-k*axes[ax][0],p[1]-k*axes[ax][1],ax}); }
  I nw=I(keys.size()),spatial=ns+nw+na,n=spatial+ng;
  if(maxnodes>=0&&n>maxnodes) throw std::runtime_error("Position exceeds node budget; increase budget, never crop");
  I nonlocal=ns*ns+ng*(2*spatial+ng);
  auto check=[&](I local){ if(maxedges>=0&&nonlocal+2*local>maxedges) throw std::runtime_error("Position exceeds relation budget; increase budget, never crop"); };
  check(0);
  g=new Graph;
  auto& a=g->a;
  if(na) a[0].assign(actions,actions+2*na);
  a[1].resize(n,3); a[2].resize(n,2);a[3].resize(n,0);g->f.resize(n*8,0);
  for(I i=0;i<n;i++) std::copy(phase,phase+4,g->f.data()+i*8);
  for(I i=0;i<ns;i++) { a[1][i]=0;a[2][i]=sorted[i][2]!=player;a[8].insert(a[8].end(),{sorted[i][0],sorted[i][1]}); }
  for(I i=ns;i<ns+nw;i++) a[1][i]=1;
  for(I i=0;i<na;i++) { a[1][ns+nw+i]=2; ids[{actions[2*i],actions[2*i+1]}]=ns+nw+i; }
  auto& local=a[4];
  I reserve=nw*12+ns*432+na*6+spatial;
  if(maxedges>=0) reserve=std::min(reserve,(maxedges-nonlocal)/2);
  local.reserve(reserve*5);
  I wi=0;
  for(auto key:keys) {
   std::array<Point,6> cells; I vals[6],code=0,reverse=0,mul=1,own=0,other=0;
   for(I k=0;k<6;k++) { Point p={key[0]+k*axes[key[2]][0],key[1]+k*axes[key[2]][1]};cells[k]=p;
    a[9].insert(a[9].end(),{p[0],p[1]});auto it=occupied.find(p);vals[k]=it==occupied.end()?0:(it->second==player?1:2);
    code+=vals[k]*mul;mul*=3;own+=vals[k]==1;other+=vals[k]==2; }
   mul=1;for(I k=5;k>=0;k--) { reverse+=vals[k]*mul;mul*=3; }
   I node=ns+wi++;a[3][node]=std::min(code,reverse);g->f[node*8+4]=float(double(own)/6);g->f[node*8+5]=float(double(other)/6);
   for(I k=0;k<6;k++) { auto it=ids.find(cells[k]);if(it==ids.end()) continue;
    I slot=code==reverse?std::min(k,5-k):(code<reverse?k:5-k);
    edge(local,it->second,node,0,0,slot);edge(local,node,it->second,1,0,slot); }
   check(I(local.size()/5));
  }
  for(I i=0;i<ns;i++) for(I q=-8;q<=8;q++) for(I r=std::max(I(-8),-q-8);r<=std::min(I(8),-q+8);r++) {
   if(q==0&&r==0) continue;
   auto it=ids.find({sorted[i][0]+q,sorted[i][1]+r});
   if(it==ids.end()||it->second<ns+nw) continue;
   I d=std::max({std::abs(q),std::abs(r),std::abs(q+r)});edge(local,i,it->second,2,d,6);edge(local,it->second,i,3,d,6);check(I(local.size()/5));
  }
  check(I(local.size()/5));
  for(I i=0;i<na;i++) for(auto delta:neighbors) {auto it=ids.find({actions[2*i]+delta[0],actions[2*i+1]+delta[1]});
   if(it!=ids.end()&&it->second>=ns+nw) { edge(local,it->second,ns+nw+i,4,1,6);check(I(local.size()/5)); } }
  for(I i=0;i<spatial;i++) edge(local,i,i,9,0,6);
  check(I(local.size()/5));
  a[5].reserve(ns*ns*5);for(I i=0;i<ns;i++) for(I j=0;j<ns;j++) edge(a[5],i,j,5,distance({sorted[i][0],sorted[i][1]},{sorted[j][0],sorted[j][1]}),6);
  a[6].reserve(ng*n*5);for(I j=spatial;j<n;j++) for(I i=0;i<n;i++) edge(a[6],i,j,i<spatial?6:8,0,6);
  a[7].reserve(ng*spatial*5);for(I i=0;i<spatial;i++) for(I j=spatial;j<n;j++) edge(a[7],j,i,7,0,6);
  return g;
 } catch(const std::exception& e) {error=e.what();delete g;return nullptr;}
}
