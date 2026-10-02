#include "hexo.hpp"
#include "nnue.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
struct Cell {
    int64_t q=0, r=0;
    bool operator==(const Cell&) const = default;
    bool operator<(const Cell& b) const { return q != b.q ? q < b.q : r < b.r; }
    Cell operator+(Cell b) const { return {q+b.q,r+b.r}; }
    Cell operator*(int n) const { return {q*n,r*n}; }
};
constexpr uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x = (x^(x>>27))*0x94d049bb133111ebULL;
    return x^(x>>31);
}
struct CellHash {
    size_t operator()(Cell c) const { return mix(uint64_t(c.q)) ^ mix(uint64_t(c.r)+0x123456789abcdefULL); }
};
constexpr Cell axes[]={{1,0},{0,1},{1,-1}};
struct Window {
    Cell start;
    int axis;
    bool operator==(const Window&) const = default;
};
struct WindowHash {
    size_t operator()(const Window& w) const {
        constexpr uint64_t directions[]={mix(51),mix(52),mix(53)};
        return CellHash{}(w.start)^directions[w.axis];
    }
};
using Counts = std::array<uint8_t,2>;
struct WindowData { Counts counts{}; uint16_t pattern=0; };
// Search repeatedly creates and removes the same six-cell windows. An open
// table avoids a heap allocation on every make/undo and keeps probes contiguous.
// Backshift deletion leaves no tombstones to accumulate during long searches.
struct WindowTable {
    struct Slot {
        Cell start{};
        uint64_t hash=0;
        WindowData data{};
        uint8_t axis=0;
        Window key() const { return {start,axis}; }
    };
    std::vector<Slot> slots=std::vector<Slot>(128);
    size_t count=0;
    static uint64_t hash(Window w) { auto h=WindowHash{}(w);return h?h:1; }
    const WindowData* find(Window w) const {
        auto h=hash(w);size_t i=h&(slots.size()-1);
        while(slots[i].hash) {
            if(slots[i].hash==h && slots[i].key()==w) return &slots[i].data;
            i=(i+1)&(slots.size()-1);
        }
        return nullptr;
    }
    WindowData& get(Window w) {
        if((count+1)*4>=slots.size()*3) {
            auto old=std::move(slots);slots=std::vector<Slot>(old.size()*2);count=0;
            for(const auto& s:old) if(s.hash) get(s.key())=s.data;
        }
        auto h=hash(w);size_t i=h&(slots.size()-1);
        while(slots[i].hash) {
            if(slots[i].hash==h && slots[i].key()==w) return slots[i].data;
            i=(i+1)&(slots.size()-1);
        }
        slots[i]={w.start,h,{},uint8_t(w.axis)};++count;return slots[i].data;
    }
    void erase(Window w) {
        auto h=hash(w);const size_t mask=slots.size()-1;size_t hole=h&mask;
        while(slots[hole].hash!=h || !(slots[hole].key()==w)) hole=(hole+1)&mask;
        for(size_t next=(hole+1)&mask;slots[next].hash;next=(next+1)&mask) {
            size_t home=slots[next].hash&mask;
            if(((next-home)&mask)>=((next-hole)&mask)) {
                slots[hole]=slots[next];hole=next;
            }
        }
        slots[hole]={};--count;
    }
};
constexpr int powers[]={1,3,9,27,81,243};
constexpr int weight[]={0,1,12,150,2400,24000,1000000};
int value(Counts c) { return c[1]==0 ? weight[c[0]] : c[0]==0 ? -weight[c[1]] : 0; }
struct Undo { Cell c; int player,remaining,winner; };
struct Center {
    std::array<int32_t,3> codes{};
    std::array<int32_t,32> sum{};
};
struct CenterTable {
    struct Slot { Cell key{}; uint64_t hash=0; Center data{}; };
    std::vector<Slot> slots;
    size_t count=0;
    static uint64_t hash(Cell c) {auto h=CellHash{}(c);return h?h:1;}
    const Center* find(Cell c) const {
        if(slots.empty()) return nullptr;
        auto h=hash(c);size_t i=h&(slots.size()-1);
        while(slots[i].hash) {
            if(slots[i].hash==h && slots[i].key==c) return &slots[i].data;
            i=(i+1)&(slots.size()-1);
        }
        return nullptr;
    }
    Center& get(Cell c,const nnue::Model& model) {
        if(slots.empty()) slots.resize(128);
        if((count+1)*4>=slots.size()*3) {
            auto old=std::move(slots);slots=std::vector<Slot>(old.size()*2);count=0;
            for(const auto& s:old) if(s.hash) get(s.key,model)=s.data;
        }
        auto h=hash(c);size_t i=h&(slots.size()-1);
        while(slots[i].hash) {
            if(slots[i].hash==h && slots[i].key==c) return slots[i].data;
            i=(i+1)&(slots.size()-1);
        }
        slots[i]={c,h,{}};++count;
        for(int j=0;j<32;++j) slots[i].data.sum[j]=3*model.row(0)[j];
        return slots[i].data;
    }
    void erase(Cell c) {
        auto h=hash(c);size_t mask=slots.size()-1,hole=h&mask;
        while(slots[hole].hash!=h || !(slots[hole].key==c)) hole=(hole+1)&mask;
        for(size_t next=(hole+1)&mask;slots[next].hash;next=(next+1)&mask) {
            size_t home=slots[next].hash&mask;
            if(((next-home)&mask)>=((next-hole)&mask)) {slots[hole]=slots[next];hole=next;}
        }
        slots[hole]={};--count;
    }
};
struct CandidateData {
    std::array<int,2> gain{};
    int nearby=0,lines=0;
    bool occupied=false;
};
struct CandidateCache {
    struct Slot {Cell cell{};CandidateData data{};};
    std::vector<Slot> slots;
    // Compact records avoid scanning empty buckets. The coordinate index is
    // only a fast lookup; distant cells use the same records through outside.
    std::vector<uint32_t> lookup;
    std::unordered_map<Cell,uint32_t,CellHash> outside;
    size_t count=0;
    std::array<int,2> empty_gain{};
    static int index(Cell c) {
        if(c.q < -64 || c.q>=64 || c.r < -64 || c.r>=64) return -1;
        return int((c.q+64)*128+c.r+64);
    }
    void reserve(size_t n) {slots.reserve(n);lookup.resize(128*128);}
    const CandidateData* find(Cell c) const {
        const int i=index(c);
        uint32_t id=0;
        if(i>=0) id=lookup[i];
        else {auto found=outside.find(c);if(found!=outside.end()) id=found->second;}
        return id?&slots[id-1].data:nullptr;
    }
    CandidateData& get(Cell c) {
        const int i=index(c);
        auto& id=i>=0?lookup[i]:outside[c];
        if(!id) {slots.push_back({c,CandidateData{empty_gain}});id=uint32_t(++count);}
        return slots[id-1].data;
    }
};
struct Board {
    std::unordered_map<Cell,int,CellHash> cells;
    WindowTable windows;
    std::array<std::unordered_set<Window,WindowHash>,2> threats;
    std::vector<Undo> history;
    int player=0,remaining=1,winner=-1;
    int64_t evaluation=0;
    int64_t learned_score=0;
    std::array<int32_t,729> features{},adjustment{};
    uint64_t stones_hash=0;
    nnue::Handle model;
    CenterTable centers;
    std::array<int64_t,64> pool{};
    std::array<int64_t,2> stone_counts{};
    CandidateCache* candidates=nullptr;
    static int count_gain(Counts n,int p) {
        int result=0;
        if(n[1-p]==0) result+=weight[n[p]+1]-weight[n[p]];
        if(n[p]==0) result+=(weight[n[1-p]+1]-weight[n[1-p]])*3/4;
        return result;
    }
    int window_gain(Counts n,int pattern,int k,int p) const {
        int delta=adjustment[pattern+(p+1)*powers[k]]-adjustment[pattern];
        return count_gain(n,p)+(p==0?delta:-delta);
    }
    static bool promising(Counts n) { return (n[0]>=2 && !n[1]) || (n[1]>=2 && !n[0]); }
    void candidate_nearby(Cell c,int delta) {
        if(!candidates) return;
        candidates->get(c).occupied=delta>0;
        for(int q=-2;q<=2;++q) for(int r=std::max(-2,-q-2);r<=std::min(2,-q+2);++r)
            candidates->get(c+Cell{q,r}).nearby+=delta;
    }
    void nn_update(Cell c,int p,int delta) {
        if(!model) return;
        auto change=[&](Cell target,int axis,int digit,bool all_axes) {
            auto& center=centers.get(target,*model);
            const auto before=center.sum;
            for(int d=0;d<3;++d) if(all_axes || d==axis) {
                int old=center.codes[d];
                center.codes[d]+=delta*(p+1)*nnue::powers[digit];
                const auto* a=model->row(old);const auto* z=model->row(center.codes[d]);
                for(int j=0;j<32;++j) center.sum[j]+=z[j]-a[j];
            }
            for(int j=0;j<32;++j) {
                int positive=std::max(0,center.sum[j])-std::max(0,before[j]);
                pool[j]+=positive;
                // ReLU(-x) = ReLU(x)-x, exactly in the integer accumulator.
                pool[j+32]+=positive-(center.sum[j]-before[j]);
            }
            if(!(center.codes[0]|center.codes[1]|center.codes[2])) centers.erase(target);
        };
        change(c,0,5,true);
        for(int d=0;d<3;++d) for(int k=-5;k<=5;++k) if(k) change(c+axes[d]*k,d,5-k,false);
    }
    void set_model(nnue::Handle next) {
        if(next && next==model) return;
        model=std::move(next);centers=CenterTable{};pool.fill(0);
        adjustment.fill(0);learned_score=0;
        if(model) for(const auto& [c,p]:cells) nn_update(c,p,1);
    }
    std::array<float,4> context() const {
        float n=float(history.size());
        return {remaining==1?1.0f:0.0f,remaining==2?1.0f:0.0f,
            std::log1p(n)/8.0f,float(stone_counts[player]-stone_counts[1-player])/std::max(1.0f,n)};
    }
    std::array<float,68> inputs() const {
        std::array<float,68> out{};
        float scale=1.0f/(256.0f*std::sqrt(float(std::max(size_t(1),centers.count))));
        for(int j=0;j<64;++j) {
            int source=player==1 && j%32<16 ? (j+32)%64:j;
            out[j]=float(pool[source])*scale;
        }
        auto phase=context();std::copy(phase.begin(),phase.end(),out.begin()+64);return out;
    }
    std::array<int32_t,3> codes(Cell c) const {
        if(model) {auto entry=centers.find(c);return entry?entry->codes:std::array<int32_t,3>{};}
        std::array<int32_t,3> out{};
        for(int d=0;d<3;++d) for(int k=-5;k<=5;++k) {
            int p=at(c+axes[d]*k);if(p>=0) out[d]+=(p+1)*nnue::powers[k+5];
        }
        return out;
    }
    std::array<float,4> pair(Cell c) const {
        if(remaining!=1 || history.empty() || history.back().player!=player) return {};
        auto first=history.back().c;int64_t q=c.q-first.q,r=c.r-first.r;
        auto distance=std::max({std::abs(q),std::abs(r),std::abs(q+r)});
        bool axis=q==0 || r==0 || q+r==0;
        return {1,float(axis),float(std::min(int64_t(8),distance))/8.0f,
            axis?float(std::max(int64_t(0),6-distance))/5.0f:0.0f};
    }
    std::array<float,16> rank_context() const {
        std::array<float,16> out{};if(!model) return out;
        auto x=inputs();
        out=model->policy_b;
        for(int j=0;j<64;++j) for(int h=0;h<16;++h) out[h]+=model->policy_w[j*16+h]*x[j];
        for(int j=0;j<4;++j) for(int h=0;h<16;++h) out[h]+=model->policy_w[(96+j)*16+h]*x[64+j];
        return out;
    }
    float rank(Cell c,const std::array<float,16>& shared) const {
        if(!model) return 0;
        auto code=codes(c);std::array<float,32> local{};
        for(int d=0;d<3;++d) {
            auto row=model->row(code[d]+(player+1)*nnue::powers[5]);
            for(int j=0;j<32;++j) local[j]+=float(row[j])/256.0f;
        }
        if(player==1) for(int j=0;j<16;++j) local[j]=-local[j];
        auto correlation=pair(c);float out=model->policy_bias;
        auto hidden=shared;
        for(int j=0;j<32;++j) for(int h=0;h<16;++h) hidden[h]+=model->policy_w[(64+j)*16+h]*local[j];
        for(int j=0;j<4;++j) for(int h=0;h<16;++h) hidden[h]+=model->policy_w[(100+j)*16+h]*correlation[j];
        for(int h=0;h<16;++h) out+=std::max(0.0f,hidden[h])*model->policy_out[h];
        return out;
    }
    int at(Cell c) const { auto it=cells.find(c); return it==cells.end() ? -1 : it->second; }
    bool legal(Cell c) const {
        if(winner>=0 || at(c)>=0) return false;
        // A representation limit, not a spatial crop. Leave ample arithmetic headroom.
        constexpr int64_t limit=1000000000000LL;
        if(c.q < -limit || c.q > limit || c.r < -limit || c.r > limit) return false;
        if(cells.empty()) return c==Cell{0,0};
        for(auto [p,_]:cells) {
            int64_t q=c.q-p.q,r=c.r-p.r;
            if(std::max({std::abs(q),std::abs(r),std::abs(q+r)})<=8) return true;
        }
        return false;
    }
    uint64_t hash() const { return stones_hash ^ mix(100+player*3+remaining) ^ mix(200+winner); }
    void update(Cell c,int p,int delta) {
        for(int d=0;d<3;++d) for(int k=0;k<6;++k) {
            Window w{c+axes[d]*(-k),d};
            auto& data=windows.get(w);
            Counts& n=data.counts;
            auto& pattern=data.pattern;
            const auto old_counts=n;const int old_pattern=pattern;
            if(pattern) --features[pattern];
            learned_score-=adjustment[pattern];
            evaluation-=value(n);
            for(int side=0;side<2;++side)
                if(n[side]>=4 && n[1-side]==0) threats[side].erase(w);
            n[p]+=delta;
            pattern+=delta*(p+1)*powers[k];
            if(candidates && n[0]+n[1]<6 && old_counts[0]+old_counts[1]<6) {
                const std::array<int,2> gain_delta={count_gain(n,0)-count_gain(old_counts,0),
                                                   count_gain(n,1)-count_gain(old_counts,1)};
                const int pattern_delta=adjustment[old_pattern]-adjustment[pattern];
                const int line_delta=int(promising(n))-int(promising(old_counts));
                for(int j=0;j<6;++j) if(j!=k && pattern/powers[j]%3==0) {
                    auto& item=candidates->get(w.start+axes[d]*j);
                    for(int side=0;side<2;++side) {
                        int change=adjustment[pattern+(side+1)*powers[j]]-
                                   adjustment[old_pattern+(side+1)*powers[j]]+pattern_delta;
                        item.gain[side]+=gain_delta[side]+(side==0?change:-change);
                    }
                    item.lines+=line_delta;
                }
            }
            if(pattern) ++features[pattern];
            learned_score+=adjustment[pattern];
            evaluation+=value(n);
            for(int side=0;side<2;++side)
                if(n[side]>=4 && n[1-side]==0) threats[side].insert(w);
            if(n[p]>=6) winner=p;
            if(n[0]+n[1]==0) windows.erase(w);
        }
    }
    void make(Cell c) {
        candidate_nearby(c,1);
        history.push_back({c,player,remaining,winner});
        cells.emplace(c,player);
        stones_hash^=mix(CellHash{}(c)^mix(player+991));
        update(c,player,1);
        nn_update(c,player,1);++stone_counts[player];
        if(--remaining==0) { player=1-player; remaining=2; }
    }
    void undo() {
        auto u=history.back(); history.pop_back();
        update(u.c,u.player,-1);
        nn_update(u.c,u.player,-1);--stone_counts[u.player];
        cells.erase(u.c);
        stones_hash^=mix(CellHash{}(u.c)^mix(u.player+991));
        player=u.player;remaining=u.remaining;winner=u.winner;
        candidate_nearby(u.c,-1);
    }
    int score(int p) const {
        int64_t residual=0;
        if(model) residual=int64_t(std::clamp(model->value(inputs())*6000.0f,-1000000.0f,1000000.0f))*(player==0?1:-1);
        auto v=std::clamp<int64_t>(evaluation+learned_score+residual,-500000,500000);
        return int(p==0?v:-v);
    }
    std::vector<Cell> legal_moves() const {
        if(winner>=0) return {};
        if(cells.empty()) return {{0,0}};
        constexpr int64_t limit=1000000000000LL;
        int64_t q0=limit,q1=-limit,r0=limit,r1=-limit;
        for(auto [c,_]:cells) {q0=std::min(q0,c.q);q1=std::max(q1,c.q);r0=std::min(r0,c.r);r1=std::max(r1,c.r);}
        const int64_t width=q1-q0+17,height=r1-r0+17;
        if(width<=int64_t(1)<<22 && height<=int64_t(1)<<22 && width*height<=int64_t(1)<<22) {
            // Dense (q, r)-major grid over the stones' box plus radius 8: row-major scanning is the sorted order.
            std::vector<uint8_t> grid(size_t(width*height));
            for(auto [c,_]:cells) for(int q=-8;q<=8;++q) {
                auto row=grid.data()+(c.q-q0+8+q)*height+(c.r-r0+8);
                for(int r=std::max(-8,-q-8);r<=std::min(8,-q+8);++r) row[r]=1;
            }
            for(auto [c,_]:cells) grid[size_t((c.q-q0+8)*height+c.r-r0+8)]=0;
            std::vector<Cell> result;
            for(int64_t i=0;i<width;++i) for(int64_t j=0;j<height;++j) if(grid[size_t(i*height+j)]) {
                Cell p{q0-8+i,r0-8+j};
                if(std::abs(p.q)<=limit && std::abs(p.r)<=limit) result.push_back(p);
            }
            return result;
        }
        std::unordered_set<Cell,CellHash> out;
        for(auto [c,_]:cells) for(int q=-8;q<=8;++q)
            for(int r=std::max(-8,-q-8);r<=std::min(8,-q+8);++r) {
                Cell p=c+Cell{q,r};
                if(at(p)<0 && std::abs(p.q)<=1000000000000LL && std::abs(p.r)<=1000000000000LL) out.insert(p);
            }
        std::vector<Cell> result(out.begin(),out.end());
        std::sort(result.begin(),result.end());
        return result;
    }
    std::vector<Cell> empty(Window w) const {
        std::vector<Cell> result;
        for(int i=0;i<6;++i) { auto c=w.start+axes[w.axis]*i; if(at(c)<0) result.push_back(c); }
        std::sort(result.begin(),result.end());
        return result;
    }
    std::vector<std::vector<Cell>> completions(int p,int stones=2) const {
        std::vector<std::vector<Cell>> out;
        for(auto w:threats[p]) {
            auto e=empty(w);
            if(!e.empty() && int(e.size())<=stones) out.push_back(std::move(e));
        }
        std::sort(out.begin(),out.end());
        out.erase(std::unique(out.begin(),out.end()),out.end());
        return out;
    }
    int gain(Cell c,int p) const {
        if(candidates) {
            auto data=candidates->find(c);
            return data?data->gain[p]:candidates->empty_gain[p];
        }
        int score=0;
        for(int d=0;d<3;++d) for(int k=0;k<6;++k) {
            auto data=windows.find({c+axes[d]*(-k),d});
            Counts n=data?data->counts:Counts{0,0};
            int pattern=data?data->pattern:0;
            score+=window_gain(n,pattern,k,p);
        }
        return score;
    }
    int placed_score(Cell c,int p) const {
        // Exact scalar evaluation after one empty cell is filled. Candidate
        // ranking need not update the board, feature counts or threat sets.
        int64_t score=evaluation+learned_score;
        for(int a=0;a<3;++a) for(int k=0;k<6;++k) {
            auto data=windows.find({c+axes[a]*(-k),a});
            Counts n=data?data->counts:Counts{};
            int pattern=data?data->pattern:0;
            score-=value(n)+adjustment[pattern];
            ++n[p];
            score+=value(n)+adjustment[pattern+(p+1)*powers[k]];
        }
        int result=int(std::clamp(score,int64_t(-500000),int64_t(500000)));
        return p==0?result:-result;
    }
};
struct CandidateGuard {
    Board& b;CandidateCache cache;
    explicit CandidateGuard(Board& board):b(board) {
        if(b.model) return;
        for(int p=0;p<2;++p) for(int k=0;k<6;++k) cache.empty_gain[p]+=3*b.window_gain({},0,k,p);
        cache.reserve(b.cells.size()*12+128);
        for(auto [c,p]:b.cells) {
            cache.get(c).occupied=true;
            for(int q=-2;q<=2;++q) for(int r=std::max(-2,-q-2);r<=std::min(2,-q+2);++r)
                ++cache.get(c+Cell{q,r}).nearby;
        }
        for(const auto& slot:b.windows.slots) if(slot.hash) {
            const auto& data=slot.data;
            for(int j=0;j<6;++j) if(data.pattern/powers[j]%3==0) {
                auto& item=cache.get(slot.start+axes[slot.axis]*j);
                for(int p=0;p<2;++p) item.gain[p]+=b.window_gain(data.counts,data.pattern,j,p)-b.window_gain({},0,j,p);
                item.lines+=int(Board::promising(data.counts));
            }
        }
        b.candidates=&cache;
    }
    ~CandidateGuard(){b.candidates=nullptr;}
};
struct CandidatePause {
    Board& b;CandidateCache* saved;
    // Leaf tactics and scalar probes do not rank moves. Their make/undo must
    // finish before the parent's candidate cache is reattached.
    explicit CandidatePause(Board& board,bool pause=true):b(board),saved(b.candidates) {if(pause) b.candidates=nullptr;}
    ~CandidatePause(){b.candidates=saved;}
};
struct Restore {
    Board& b; size_t size;
    explicit Restore(Board& b):b(b),size(b.history.size()){}
    ~Restore(){ while(b.history.size()>size) b.undo(); }
};
struct Turn {
    std::array<Cell,2> cells{};
    int count=0,score=0;
};
void apply(Board& b,const Turn& t) { for(int i=0;i<t.count && b.winner<0;++i) b.make(t.cells[i]); }
constexpr int mate=10000000;
struct Timeout {};
struct Entry { uint64_t key=0; Turn best; };
struct Search {
    Clock::time_point deadline;
    int width;
    uint64_t nodes=0;
    std::vector<Entry> tt, frozen_hints;
    bool inject_tt=false;
    Search(int ms,int width,bool inject=false):deadline(Clock::now()+std::chrono::milliseconds(ms)),width(width),tt(inject?1<<16:0),inject_tt(inject){}
    void check() const { if(Clock::now()>=deadline) throw Timeout{}; }
    Turn immediate(Board& b) {
        auto completions=b.completions(b.player,b.remaining);
        std::stable_sort(completions.begin(),completions.end(),[](const auto& a,const auto& z){return a.size()<z.size();});
        for(const auto& e:completions) {
            CandidatePause pause(b);Restore restore(b);
            Turn t;
            for(auto c:e) { if(!b.legal(c)) break; t.cells[t.count++]=c;b.make(c);if(b.winner>=0) return t; }
        }
        return {};
    }
    using RankedCells=std::vector<std::pair<int,Cell>>;
    static RankedCells candidate_scores(Board& b) {
        if(b.candidates) {
            RankedCells ranked;
            ranked.reserve(b.candidates->count);
            for(const auto& slot:b.candidates->slots) {
                const auto& c=slot.cell;const auto& item=slot.data;
                if(!item.occupied && (item.nearby || item.lines) && std::abs(c.q)<=1000000000000LL && std::abs(c.r)<=1000000000000LL)
                    ranked.emplace_back(item.gain[b.player],c);
            }
            return ranked;
        }
        std::unordered_set<Cell,CellHash> set;
        for(auto [c,_]:b.cells) {
            for(int q=-2;q<=2;++q) for(int r=std::max(-2,-q-2);r<=std::min(2,-q+2);++r)
                if(b.at(c+Cell{q,r})<0) set.insert(c+Cell{q,r});
        }
        // Include every empty cell of a promising line, even far from the last move.
        for(const auto& slot:b.windows.slots) if(slot.hash) {
            const auto& data=slot.data;
            if((data.counts[0]>=2 && !data.counts[1]) || (data.counts[1]>=2 && !data.counts[0]))
                for(auto c:b.empty(slot.key())) set.insert(c);
        }
        RankedCells ranked;
        // Every candidate is empty and within five cells of an occupied cell.
        // Only the coordinate representation limit needs checking here.
        ranked.reserve(set.size());
        if(b.model) {
            auto shared=b.rank_context();
            for(auto c:set) if(std::abs(c.q)<=1000000000000LL && std::abs(c.r)<=1000000000000LL)
                ranked.emplace_back(b.gain(c,b.player)+int(std::clamp(b.rank(c,shared)*6000.0f,-1000000.0f,1000000.0f)),c);
        } else {
            for(auto c:set) if(std::abs(c.q)<=1000000000000LL && std::abs(c.r)<=1000000000000LL)
                ranked.emplace_back(b.gain(c,b.player),c);
        }
        return ranked;
    }
    static std::vector<Cell> select_candidates(Board& b,RankedCells ranked,int limit) {
        const auto better=[](const auto& a,const auto& z){ return a.first!=z.first ? a.first>z.first : a.second<z.second; };
        if(int(ranked.size())>limit) {
            std::nth_element(ranked.begin(),ranked.begin()+limit,ranked.end(),better);
            ranked.resize(limit);
        }
        std::sort(ranked.begin(),ranked.end(),better);
        std::vector<Cell> out;
        for(int i=0;i<std::min(limit,int(ranked.size()));++i) out.push_back(ranked[i].second);
        // Tactical cells cannot be removed by ordinary move ordering.
        for(int p=0;p<2;++p) for(auto& e:b.completions(p)) for(auto c:e)
            if(std::find(out.begin(),out.end(),c)==out.end() && b.legal(c)) out.push_back(c);
        return out;
    }
    static std::vector<Cell> candidates(Board& b,int limit) {
        if(b.cells.empty()) return {{0,0}};
        return select_candidates(b,candidate_scores(b),limit);
    }
    static bool candidate_cell(const Board& b,Cell c) {
        for(int q=-2;q<=2;++q) for(int r=std::max(-2,-q-2);r<=std::min(2,-q+2);++r)
            if(b.at(c+Cell{q,r})>=0) return true;
        for(int a=0;a<3;++a) for(int k=0;k<6;++k) {
            auto data=b.windows.find({c+axes[a]*(-k),a});
            if(data && ((data->counts[0]>=2 && !data->counts[1]) || (data->counts[1]>=2 && !data->counts[0]))) return true;
        }
        return false;
    }
    static RankedCells following_scores(Board& b,const RankedCells& base,Cell first) {
        if(b.candidates) return candidate_scores(b);
        // Only the 18 windows through the first stone changed. Keep all other
        // scalar rankings and discover only the newly reachable candidates.
        RankedCells ranked;ranked.reserve(base.size()+48);
        for(const auto& [score,c]:base) if(!(c==first)) {
            auto q=c.q-first.q,r=c.r-first.r;
            bool changed=(q==0 || r==0 || q+r==0) && std::max(std::abs(q),std::abs(r))<=5;
            // A blocked line can stop admitting a distant empty endpoint.
            if(changed && std::max(std::abs(q),std::abs(r))>2 && !candidate_cell(b,c)) continue;
            ranked.emplace_back(changed?b.gain(c,b.player):score,c);
        }
        std::vector<Cell> added;
        for(int q=-2;q<=2;++q) for(int r=std::max(-2,-q-2);r<=std::min(2,-q+2);++r)
            added.push_back(first+Cell{q,r});
        for(int a=0;a<3;++a) for(int k=0;k<6;++k) {
            Window w{first+axes[a]*(-k),a};auto data=b.windows.find(w);
            if(data && ((data->counts[0]>=2 && !data->counts[1]) || (data->counts[1]>=2 && !data->counts[0])))
                for(auto c:b.empty(w)) added.push_back(c);
        }
        std::sort(added.begin(),added.end());added.erase(std::unique(added.begin(),added.end()),added.end());
        for(auto c:added) {
            if(b.at(c)>=0 || std::abs(c.q)>1000000000000LL || std::abs(c.r)>1000000000000LL) continue;
            auto found=std::lower_bound(base.begin(),base.end(),c,[](const auto& a,Cell z){return a.second<z;});
            if(found==base.end() || !(found->second==c)) ranked.emplace_back(b.gain(c,b.player),c);
        }
        return ranked;
    }
    void covers(const std::vector<std::vector<Cell>>& threats,std::vector<Cell>& selected,
                int remaining,std::vector<std::vector<Cell>>& out) {
        for(const auto& e:threats) {
            bool hit=false;
            for(auto c:e) if(std::find(selected.begin(),selected.end(),c)!=selected.end()) hit=true;
            if(hit) continue;
            if(!remaining) return;
            for(auto c:e) { selected.push_back(c);covers(threats,selected,remaining-1,out);selected.pop_back(); }
            return;
        }
        out.push_back(selected);
    }
    bool cover_exists(const std::vector<std::vector<Cell>>& threats,std::array<Cell,2>& selected,
                      int used,int remaining) const {
        for(const auto& completion:threats) {
            bool hit=false;
            for(auto cell:completion) for(int i=0;i<std::min(used,2);++i) if(selected[i]==cell) hit=true;
            if(hit) continue;
            if(remaining<=0 || used>=2) return false;
            for(auto cell:completion) {
                selected[used]=cell;
                if(cover_exists(threats,selected,used+1,remaining-1)) return true;
            }
            return false;
        }
        // A spare placement may be anywhere legal. Its strategic value is
        // unknown; existence of a cover is never treated as a forced win.
        return true;
    }
    bool unavoidable_loss(const Board& b) const {
        if(b.threats[1-b.player].empty()) return false;
        auto threats=b.completions(1-b.player);
        std::erase_if(threats,[&](const auto& completion) {
            return !std::all_of(completion.begin(),completion.end(),[&](Cell c){return b.legal(c);});
        });
        std::array<Cell,2> selected{};
        return !cover_exists(threats,selected,0,b.remaining);
    }
    std::vector<Turn> turns(Board& b,bool timed=true,Turn hint={}) {
        Turn win=immediate(b); if(win.count) {win.score=mate;return {win};}
        int side=b.player;
        auto constraints=b.completions(1-side);
        std::vector<Turn> result;
        std::unordered_set<uint64_t> seen;
        auto add=[&](Turn t,bool mandatory=false) {
            if(t.count<1 || t.count>2 || t.count>b.remaining) return;
            CandidatePause pause(b);Restore restore(b);
            for(int i=0;i<t.count;++i) {
                if(!b.legal(t.cells[i])) return;
                b.make(t.cells[i]);
                if(b.winner>=0) {t.count=i+1;break;}
            }
            if(b.winner<0 && b.player==side) return;
            if(mandatory && b.winner<0 && !b.completions(1-side).empty()) return;
            if(!seen.insert(b.hash()).second) return;
            t.score=b.winner==side?mate:b.score(side);
            // Every immediate opponent completion must be covered.
            if(b.winner<0 && !b.completions(1-side).empty()) t.score=-mate;
            result.push_back(t);
        };
        // Validate both placements in order before reserving a slot. A defensive
        // hint must cover every immediate threat; own wins were handled above.
        add(hint,!constraints.empty());
        const bool pinned=!result.empty();
        RankedCells base;
        if(!b.model && b.remaining==2) {
            base=candidate_scores(b);
            std::sort(base.begin(),base.end(),[](const auto& a,const auto& z){return a.second<z.second;});
        }
        auto follow=[&](Cell first,int limit) {
            Restore restore(b);b.make(first);
            auto seconds=b.model?candidates(b,limit):select_candidates(b,following_scores(b,base,first),limit);
            if(b.model) {
                b.undo();
                for(auto c:seconds) add({{first,c},2,0});
                return;
            }
            for(auto c:seconds) {
                // Immediate wins were returned above. These two placements
                // cannot make a new opponent threat; they only cover old ones.
                // Deduplicate using the same final-position hash as add().
                auto key=b.stones_hash^mix(CellHash{}(c)^mix(side+991))^mix(100+(1-side)*3+2)^mix(199);
                if(!seen.insert(key).second) continue;
                int score=b.placed_score(c,side);
                for(const auto& threat:constraints)
                    if(std::find(threat.begin(),threat.end(),first)==threat.end() &&
                       std::find(threat.begin(),threat.end(),c)==threat.end()) {score=-mate;break;}
                result.push_back({{first,c},2,score});
            }
        };
        if(!constraints.empty()) {
            std::vector<std::vector<Cell>> defenses;std::vector<Cell> selected;
            covers(constraints,selected,b.remaining,defenses);
            for(auto cover:defenses) {
                if(timed) check();
                if(int(cover.size())==b.remaining) {
                    Turn t; t.count=int(cover.size());std::copy(cover.begin(),cover.end(),t.cells.begin());add(t);
                    if(t.count==2) {std::swap(t.cells[0],t.cells[1]);add(t);}
                } else {
                    if(!b.legal(cover[0])) continue;
                    follow(cover[0],width);
                }
            }
            if(!result.empty()) {std::sort(result.begin()+int(pinned),result.end(),[](const Turn& a,const Turn& z){return a.score>z.score;});return result;}
        }
        auto firsts=base.empty()?candidates(b,width):select_candidates(b,base,width);
        for(auto a:firsts) {
            if(timed) check();
            if(b.remaining==1) {add({{a,{}},1,0});continue;}
            follow(a,std::max(6,width/2));
        }
        std::stable_sort(result.begin()+int(pinned),result.end(),[](const Turn& a,const Turn& z){return a.score>z.score;});
        if(int(result.size())>width*2) result.resize(width*2);
        return result;
    }
    int negamax(Board& b,int depth,int alpha,int beta) {
        ++nodes;check();
        if(b.winner>=0) return b.winner==b.player?mate:-mate;
        if(immediate(b).count) return mate;
        if(depth<=0) return unavoidable_loss(b)?-mate:b.score(b.player);
        Turn hint{};
        if(inject_tt) {
            uint64_t key=b.hash();const auto& entry=frozen_hints[key&(frozen_hints.size()-1)];
            if(entry.key==key) hint=entry.best;
        }
        auto moves=turns(b,true,hint);
        if(moves.empty()) return b.score(b.player);
        int best=-mate-1;Turn best_turn=moves.front();bool first=true;
        for(const auto& t:moves) {
            CandidatePause pause(b,depth==1);Restore restore(b);int side=b.player;apply(b,t);
            int score;
            if(b.winner==side) score=mate;
            else if(first) score=-negamax(b,depth-1,-beta,-alpha);
            else {
                score=-negamax(b,depth-1,-alpha-1,-alpha);
                if(score>alpha && score<beta) score=-negamax(b,depth-1,-beta,-alpha);
            }
            first=false;
            if(score>best) {best=score;best_turn=t;}
            alpha=std::max(alpha,score);if(alpha>=beta) break;
        }
        if(inject_tt) {uint64_t key=b.hash();tt[key&(tt.size()-1)]={key,best_turn};}
        return best;
    }
    std::vector<Turn> diversify(Board& b,std::vector<Turn> base,int seconds,int cap,bool timed=true) {
        // Existing tactical branches and all selected baseline turns survive.
        if(!seconds || int(base.size())>=cap || b.remaining!=2 || immediate(b).count ||
            !b.completions(1-b.player).empty()) return base;
        struct Alternative {int priority,first,second;Turn turn;};
        std::vector<Alternative> alternatives;
        auto firsts=candidates(b,width);
        for(int i=0;i<int(firsts.size());++i) {
            if(timed) check();
            Restore restore(b);b.make(firsts[i]);
            auto following=candidates(b,seconds);
            for(int j=0;j<int(following.size());++j)
                alternatives.push_back({(i+1)*(j+1),i,j,{{firsts[i],following[j]},2,0}});
        }
        // Rank-product admission gives low-ranked conditional replies to strong
        // first placements a chance without another global static-score cut.
        std::sort(alternatives.begin(),alternatives.end(),[](const auto& a,const auto& z) {
            if(a.priority!=z.priority) return a.priority<z.priority;
            if(a.first!=z.first) return a.first<z.first;
            return a.second<z.second;
        });
        std::unordered_set<uint64_t> seen;
        for(const auto& t:base) {Restore restore(b);apply(b,t);seen.insert(b.hash());}
        int side=b.player;
        for(auto& alternative:alternatives) {
            if(timed) check();
            if(int(base.size())>=cap) break;
            auto t=alternative.turn;Restore restore(b);
            bool valid=true;
            for(int i=0;i<t.count;++i) {
                if(!b.legal(t.cells[i])) {valid=false;break;}
                b.make(t.cells[i]);if(b.winner>=0) {t.count=i+1;break;}
            }
            if(!valid || (b.winner<0 && b.player==side) || !seen.insert(b.hash()).second) continue;
            t.score=b.winner==side?mate:b.score(side);
            if(b.winner<0 && !b.completions(1-side).empty()) t.score=-mate;
            base.push_back(t);
        }
        std::stable_sort(base.begin(),base.end(),[](const auto& a,const auto& z){return a.score>z.score;});
        return base;
    }
    HxResult run(Board& b,int max_depth,int root_seconds=0,int root_turns=0) {
        auto start=Clock::now();CandidateGuard candidate_cache(b);Restore restore(b);HxResult output{};
        if(b.winner>=0) return output;
        Turn chosen=immediate(b);
        if(chosen.count) {chosen.score=mate;output.depth=1;}
        else {
            // Always have a legal answer even when a tiny budget expires in generation.
            auto moves=candidates(b,1);
            if(moves.empty()) return output;
            chosen.cells[0]=moves.front();chosen.count=1;
            std::vector<std::vector<Cell>> defenses;std::vector<Cell> selected;
            auto constraints=b.completions(1-b.player);
            covers(constraints,selected,b.remaining,defenses);
            if(!constraints.empty() && !defenses.empty()) {
                auto cover=defenses.front();
                chosen.count=int(cover.size());
                std::copy(cover.begin(),cover.end(),chosen.cells.begin());
            }
            {Restore fallback(b);int side=b.player;b.make(chosen.cells[0]);
                if(b.player==side && b.winner<0) {
                    if(chosen.count<2) {auto second=candidates(b,1);chosen.cells[1]=second.front();chosen.count=2;}
                    b.make(chosen.cells[1]);
                }
                chosen.score=b.winner==side?mate:b.score(side);
                if(b.winner<0 && !b.completions(1-side).empty()) chosen.score=-mate;
            }
            try {
                auto roots=turns(b);
                if(!roots.empty()) chosen=roots.front();
                if(root_seconds) roots=diversify(b,std::move(roots),root_seconds,root_turns);
                for(int depth=1;depth<=max_depth;++depth) {
                    // Freeze admission for the whole iteration, including PVS
                    // re-searches. Scores from a different admitted tree cannot
                    // provide bounds; only previous-iteration moves are reused.
                    if(inject_tt) frozen_hints=tt;
                    int best=-mate-1;Turn iteration=chosen;
                    for(auto& t:roots) {
                        check();CandidatePause pause(b,depth==1);Restore branch(b);int side=b.player;apply(b,t);
                        int score=b.winner==side?mate:-negamax(b,depth-1,-mate-1,-best);
                        t.score=score;
                        if(score>best) {best=score;iteration=t;}
                        // No remaining root can beat mate. Finish this iteration
                        // before a later timeout can discard the winning move.
                        if(best>=mate) break;
                    }
                    chosen=iteration;chosen.score=best;output.depth=depth;
                    std::stable_sort(roots.begin(),roots.end(),[](auto a,auto z){return a.score>z.score;});
                    if(best>=mate || best<=-mate) break;
                }
            } catch(const Timeout&) {}
        }
        output.q1=chosen.cells[0].q;output.r1=chosen.cells[0].r;
        output.q2=chosen.cells[1].q;output.r2=chosen.cells[1].r;
        output.count=chosen.count;output.score=chosen.score;output.nodes=nodes;
        output.elapsed_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
        return output;
    }
};
}
extern "C" {
void* hx_new(){try{return new Board;}catch(...){return nullptr;}}
void hx_free(void* p){delete static_cast<Board*>(p);}
int hx_play(void* p,int64_t q,int64_t r){auto& b=*static_cast<Board*>(p);if(!b.legal({q,r}))return 0;b.make({q,r});return 1;}
int hx_undo(void* p){auto& b=*static_cast<Board*>(p);if(b.history.empty())return 0;b.undo();return 1;}
int hx_player(void* p){return static_cast<Board*>(p)->player;}
int hx_remaining(void* p){return static_cast<Board*>(p)->remaining;}
int hx_winner(void* p){return static_cast<Board*>(p)->winner;}
int hx_size(void* p){return int(static_cast<Board*>(p)->history.size());}
int hx_cell(void* p,int index,HxCell* out){auto& b=*static_cast<Board*>(p);if(index<0 || index>=int(b.history.size()))return 0;auto u=b.history[index];*out={u.c.q,u.c.r,u.player};return 1;}
int hx_legal(void* p,int64_t q,int64_t r){return static_cast<Board*>(p)->legal({q,r});}
int hx_moves(void* p,HxCell* out,int cap){auto cells=static_cast<Board*>(p)->legal_moves();for(int i=0;i<std::min(cap,int(cells.size()));++i)out[i]={cells[i].q,cells[i].r,-1};return int(cells.size());}
int hx_search(void* p,int ms,int depth,int width,HxResult* out){if(ms<1 || depth<1 || width<2 || width>128)return 0;try{auto start=Clock::now();Search s(ms,width);*out=s.run(*static_cast<Board*>(p),depth);out->elapsed_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();return 1;}catch(...){return 0;}}
int hx_search_root(void* p,int ms,int depth,int width,int seconds,int cap,HxResult* out) {
    if(!seconds && !cap) return hx_search(p,ms,depth,width,out);
    if(ms<1 || depth<1 || width<2 || width>128 || seconds<std::max(6,width/2) || seconds>128 || cap<2*width || cap>1024) return 0;
    try {
        auto start=Clock::now();Search search(ms,width);
        *out=search.run(*static_cast<Board*>(p),depth,seconds,cap);
        out->elapsed_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();return 1;
    } catch(...) {return 0;}
}
int hx_search_tt(void* p,int ms,int depth,int width,int seconds,int cap,HxResult* out) {
    if(ms<1 || depth<1 || width<2 || width>128 ||
        ((seconds || cap) && (seconds<std::max(6,width/2) || seconds>128 || cap<2*width || cap>1024))) return 0;
    try {
        auto start=Clock::now();Search search(ms,width,true);
        *out=search.run(*static_cast<Board*>(p),depth,seconds,cap);
        out->elapsed_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();return 1;
    } catch(...) {return 0;}
}
int hx_turns(void* p,int width,int seconds,int cap,HxTurn* out,int capacity) {
    if(width<2 || width>128 || ((seconds || cap) && (seconds<std::max(6,width/2) || seconds>128 || cap<2*width || cap>1024))) return -1;
    auto& b=*static_cast<Board*>(p);if(b.winner>=0) return 0;
    Search search(1,width,false);auto turns=search.turns(b,false);
    if(seconds) turns=search.diversify(b,std::move(turns),seconds,cap,false);
    if(out) for(int i=0;i<std::min(capacity,int(turns.size()));++i) {
        const auto& t=turns[i];out[i]={t.cells[0].q,t.cells[0].r,t.cells[1].q,t.cells[1].r,t.count,t.score};
    }
    return int(turns.size());
}
uint64_t hx_hash(void* p){return static_cast<Board*>(p)->hash();}
int hx_evaluate(void* p){auto& b=*static_cast<Board*>(p);return b.score(b.player);}
int hx_features(void* p,int32_t* out,int cap){auto& b=*static_cast<Board*>(p);for(int i=0;i<std::min(cap,729);++i)out[i]=b.features[i];return 729;}
int hx_load_table(void* p,const int32_t* weights,int count){
    if(count!=729 || weights[0]!=0) return 0;
    for(int i=0;i<729;++i) if(weights[i]<-10000 || weights[i]>10000) return 0;
    auto& b=*static_cast<Board*>(p);b.learned_score=0;
    if(b.model) b.set_model({});
    for(int i=0;i<729;++i){b.adjustment[i]=weights[i];b.learned_score+=int64_t(b.features[i])*weights[i];}
    return 1;
}
void* hx_model_load(const char* path) {
    try {auto model=nnue::load(path);nnue::error.clear();return new nnue::Handle(std::move(model));}
    catch(const std::exception& e) {nnue::error=e.what();return nullptr;}
}
void hx_model_free(void* p) {delete static_cast<nnue::Handle*>(p);}
const char* hx_model_error() {return nnue::error.c_str();}
int hx_set_model(void* p,void* model) {
    try {static_cast<Board*>(p)->set_model(model?*static_cast<nnue::Handle*>(model):nnue::Handle{});return 1;}
    catch(const std::exception& e) {nnue::error=e.what();return 0;}
}
int hx_nnue_centers(void* p,int64_t* coordinates,int32_t* codes,int capacity) {
    auto& b=*static_cast<Board*>(p);
    std::vector<std::pair<Cell,std::array<int32_t,3>>> result;
    if(b.model) {
        result.reserve(b.centers.count);
        for(const auto& s:b.centers.slots) if(s.hash) result.emplace_back(s.key,s.data.codes);
    } else {
        std::unordered_map<Cell,std::array<int32_t,3>,CellHash> out;
        for(const auto& [c,side]:b.cells) for(int d=0;d<3;++d) for(int k=-5;k<=5;++k)
            out[c+axes[d]*k][d]+=(side+1)*nnue::powers[5-k];
        result.assign(out.begin(),out.end());
    }
    std::sort(result.begin(),result.end(),[](const auto& a,const auto& z){return a.first<z.first;});
    for(int i=0;i<std::min(capacity,int(result.size()));++i) {
        if(coordinates) {coordinates[2*i]=result[i].first.q;coordinates[2*i+1]=result[i].first.r;}
        if(codes) std::copy(result[i].second.begin(),result[i].second.end(),codes+3*i);
    }
    return int(result.size());
}
int hx_nnue_context(void* p,float* out) {
    if(!out) return 0;
    auto x=static_cast<Board*>(p)->context();std::copy(x.begin(),x.end(),out);return 4;
}
int hx_nnue_inputs(void* p,float* out,int capacity) {
    auto x=static_cast<Board*>(p)->inputs();if(out) std::copy_n(x.begin(),std::max(0,std::min(capacity,68)),out);return 68;
}
int hx_nnue_policy_features(void* p,int64_t q,int64_t r,int32_t* out,float* pair) {
    auto& b=*static_cast<Board*>(p);Cell c{q,r};if(!b.legal(c) || !out || !pair) return 0;
    auto codes=b.codes(c);for(int d=0;d<3;++d) out[d]=codes[d]+(b.player+1)*nnue::powers[5];
    auto correlation=b.pair(c);std::copy(correlation.begin(),correlation.end(),pair);return 1;
}
int hx_nnue_policy_batch(void* p,const int64_t* coordinates,int count,int32_t* codes,float* pairs) {
    if(count<0 || (count && (!coordinates || !codes || !pairs))) return 0;
    for(int i=0;i<count;++i) {
        const size_t row=size_t(i);
        if(!hx_nnue_policy_features(p,coordinates[2*row],coordinates[2*row+1],codes+3*row,pairs+4*row)) return 0;
    }
    return 1;
}
float hx_nnue_rank(void* p,int64_t q,int64_t r) {
    auto& b=*static_cast<Board*>(p);if(!b.legal({q,r})) return std::numeric_limits<float>::quiet_NaN();
    return b.rank({q,r},b.rank_context());
}
int hx_candidates(void* p,int limit,HxCell* out,int capacity) {
    if(limit<1 || limit>128) return -1;
    auto& b=*static_cast<Board*>(p);if(b.winner>=0) return 0;
    auto cells=Search::candidates(b,limit);
    if(out) for(int i=0;i<std::min(capacity,int(cells.size()));++i) out[i]={cells[i].q,cells[i].r,-1};
    return int(cells.size());
}
int hx_tactical(void* p) {
    auto& b=*static_cast<Board*>(p);
    if(b.winner>=0) return 0;
    for(int side=0;side<2;++side) {
        for(const auto& completion:b.completions(side,side==b.player?b.remaining:2)) {
            // Each empty belongs to a six-cell window containing stones, so it
            // is already within the legal radius before either placement.
            if(std::all_of(completion.begin(),completion.end(),[&](Cell c){return b.legal(c);})) return 1;
        }
    }
    return 0;
}
}
