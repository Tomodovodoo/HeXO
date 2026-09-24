#include "hexo.hpp"
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
uint64_t mix(uint64_t x) {
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
    size_t operator()(const Window& w) const { return CellHash{}(w.start)^mix(w.axis+51); }
};
using Counts = std::array<int,2>;
struct WindowData { Counts counts{}; int pattern=0; };
constexpr int powers[]={1,3,9,27,81,243};
constexpr int weight[]={0,1,12,150,2400,24000,1000000};
int value(Counts c) { return c[1]==0 ? weight[c[0]] : c[0]==0 ? -weight[c[1]] : 0; }
struct Undo { Cell c; int player,remaining,winner; };
struct Board {
    std::unordered_map<Cell,int,CellHash> cells;
    std::unordered_map<Window,WindowData,WindowHash> windows;
    std::array<std::unordered_set<Window,WindowHash>,2> threats;
    std::vector<Undo> history;
    int player=0,remaining=1,winner=-1;
    int64_t evaluation=0;
    int64_t learned_score=0;
    std::array<int32_t,729> features{},adjustment{};
    uint64_t stones_hash=0;
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
            auto [it,_]=windows.try_emplace(w,WindowData{});
            Counts& n=it->second.counts;
            int& pattern=it->second.pattern;
            if(pattern) --features[pattern];
            learned_score-=adjustment[pattern];
            evaluation-=value(n);
            for(int side=0;side<2;++side)
                if(n[side]>=4 && n[1-side]==0) threats[side].erase(w);
            n[p]+=delta;
            pattern+=delta*(p+1)*powers[k];
            if(pattern) ++features[pattern];
            learned_score+=adjustment[pattern];
            evaluation+=value(n);
            for(int side=0;side<2;++side)
                if(n[side]>=4 && n[1-side]==0) threats[side].insert(w);
            if(n[p]>=6) winner=p;
            if(n[0]+n[1]==0) windows.erase(it);
        }
    }
    void make(Cell c) {
        history.push_back({c,player,remaining,winner});
        cells.emplace(c,player);
        stones_hash^=mix(CellHash{}(c)^mix(player+991));
        update(c,player,1);
        if(--remaining==0) { player=1-player; remaining=2; }
    }
    void undo() {
        auto u=history.back(); history.pop_back();
        update(u.c,u.player,-1);
        cells.erase(u.c);
        stones_hash^=mix(CellHash{}(u.c)^mix(u.player+991));
        player=u.player;remaining=u.remaining;winner=u.winner;
    }
    int score(int p) const {
        auto v=std::clamp<int64_t>(evaluation+learned_score,-500000,500000);
        return int(p==0?v:-v);
    }
    std::vector<Cell> legal_moves() const {
        if(winner>=0) return {};
        if(cells.empty()) return {{0,0}};
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
        int score=0;
        for(int d=0;d<3;++d) for(int k=0;k<6;++k) {
            auto it=windows.find({c+axes[d]*(-k),d});
            Counts n=it==windows.end()?Counts{0,0}:it->second.counts;
            int pattern=it==windows.end()?0:it->second.pattern;
            int delta=adjustment[pattern+(p+1)*powers[k]]-adjustment[pattern];
            score+=p==0?delta:-delta;
            if(n[1-p]==0) score+=weight[n[p]+1]-weight[n[p]];
            if(n[p]==0) score+=(weight[n[1-p]+1]-weight[n[1-p]])*3/4;
        }
        return score;
    }
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
struct Entry { uint64_t key=0; int depth=-1,score=0,flag=0; Turn best; };
struct Search {
    Clock::time_point deadline;
    int width;
    uint64_t nodes=0;
    std::vector<Entry> tt;
    Search(int ms,int width):deadline(Clock::now()+std::chrono::milliseconds(ms)),width(width),tt(1<<16){}
    void check() const { if(Clock::now()>=deadline) throw Timeout{}; }
    Turn immediate(Board& b) {
        auto completions=b.completions(b.player,b.remaining);
        std::stable_sort(completions.begin(),completions.end(),[](const auto& a,const auto& z){return a.size()<z.size();});
        for(const auto& e:completions) {
            Restore restore(b);
            Turn t;
            for(auto c:e) { if(!b.legal(c)) break; t.cells[t.count++]=c;b.make(c);if(b.winner>=0) return t; }
        }
        return {};
    }
    std::vector<Cell> candidates(Board& b,int limit) {
        if(b.cells.empty()) return {{0,0}};
        std::unordered_set<Cell,CellHash> set;
        for(auto [c,_]:b.cells) {
            for(int q=-2;q<=2;++q) for(int r=std::max(-2,-q-2);r<=std::min(2,-q+2);++r)
                if(b.at(c+Cell{q,r})<0) set.insert(c+Cell{q,r});
        }
        // Include every empty cell of a promising line, even far from the last move.
        for(auto [w,data]:b.windows) if((data.counts[0]>=2 && !data.counts[1]) || (data.counts[1]>=2 && !data.counts[0]))
            for(auto c:b.empty(w)) set.insert(c);
        std::vector<std::pair<int,Cell>> ranked;
        for(auto c:set) if(b.legal(c)) ranked.emplace_back(b.gain(c,b.player),c);
        std::sort(ranked.begin(),ranked.end(),[](auto a,auto z){ return a.first!=z.first ? a.first>z.first : a.second<z.second; });
        std::vector<Cell> out;
        for(int i=0;i<std::min(limit,int(ranked.size()));++i) out.push_back(ranked[i].second);
        // Tactical cells cannot be removed by ordinary move ordering.
        for(int p=0;p<2;++p) for(auto& e:b.completions(p)) for(auto c:e)
            if(std::find(out.begin(),out.end(),c)==out.end() && b.legal(c)) out.push_back(c);
        return out;
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
    std::vector<Turn> turns(Board& b,bool timed=true) {
        Turn win=immediate(b); if(win.count) {win.score=mate;return {win};}
        int side=b.player;
        auto constraints=b.completions(1-side);
        std::vector<Turn> result;
        std::unordered_set<uint64_t> seen;
        auto add=[&](Turn t) {
            Restore restore(b);
            for(int i=0;i<t.count;++i) {
                if(!b.legal(t.cells[i])) return;
                b.make(t.cells[i]);
                if(b.winner>=0) {t.count=i+1;break;}
            }
            if(b.winner<0 && b.player==side) return;
            if(!seen.insert(b.hash()).second) return;
            t.score=b.winner==side?mate:b.score(side);
            // Every immediate opponent completion must be covered.
            if(b.winner<0 && !b.completions(1-side).empty()) t.score=-mate;
            result.push_back(t);
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
                    Restore restore(b);
                    if(!b.legal(cover[0])) continue;
                    b.make(cover[0]);
                    auto seconds=candidates(b,width);
                    b.undo();
                    for(auto c:seconds) add({{cover[0],c},2,0});
                }
            }
            if(!result.empty()) {std::sort(result.begin(),result.end(),[](const Turn& a,const Turn& z){return a.score>z.score;});return result;}
        }
        auto firsts=candidates(b,width);
        for(auto a:firsts) {
            if(timed) check();
            if(b.remaining==1) {add({{a,{}},1,0});continue;}
            Restore restore(b);b.make(a);
            auto seconds=candidates(b,std::max(6,width/2));b.undo();
            for(auto c:seconds) add({{a,c},2,0});
        }
        std::stable_sort(result.begin(),result.end(),[](const Turn& a,const Turn& z){return a.score>z.score;});
        if(int(result.size())>width*2) result.resize(width*2);
        return result;
    }
    int negamax(Board& b,int depth,int alpha,int beta) {
        ++nodes;check();
        if(b.winner>=0) return b.winner==b.player?mate:-mate;
        if(immediate(b).count) return mate;
        if(depth<=0) return b.score(b.player);
        uint64_t key=b.hash();auto& entry=tt[key&(tt.size()-1)];
        if(entry.key==key && entry.depth>=depth) {
            if(entry.flag==0) return entry.score;
            if(entry.flag==1 && entry.score>=beta) return entry.score;
            if(entry.flag==2 && entry.score<=alpha) return entry.score;
        }
        const int original=alpha;
        auto moves=turns(b);
        if(moves.empty()) return b.score(b.player);
        int best=-mate-1;Turn best_turn=moves.front();bool first=true;
        for(const auto& t:moves) {
            Restore restore(b);int side=b.player;apply(b,t);
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
        entry={key,depth,best,best<=original?2:best>=beta?1:0,best_turn};
        return best;
    }
    HxResult run(Board& b,int max_depth) {
        auto start=Clock::now();Restore restore(b);HxResult output{};
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
                }
            }
            try {
                auto roots=turns(b);
                if(!roots.empty()) chosen=roots.front();
                for(int depth=1;depth<=max_depth;++depth) {
                    int best=-mate-1;Turn iteration=chosen;
                    for(auto& t:roots) {
                        check();Restore branch(b);int side=b.player;apply(b,t);
                        int score=b.winner==side?mate:-negamax(b,depth-1,-mate-1,-best);
                        t.score=score;
                        if(score>best) {best=score;iteration=t;}
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
uint64_t hx_hash(void* p){return static_cast<Board*>(p)->hash();}
int hx_evaluate(void* p){auto& b=*static_cast<Board*>(p);return b.score(b.player);}
int hx_features(void* p,int32_t* out,int cap){auto& b=*static_cast<Board*>(p);for(int i=0;i<std::min(cap,729);++i)out[i]=b.features[i];return 729;}
int hx_load_table(void* p,const int32_t* weights,int count){
    if(count!=729 || weights[0]!=0) return 0;
    for(int i=0;i<729;++i) if(weights[i]<-10000 || weights[i]>10000) return 0;
    auto& b=*static_cast<Board*>(p);b.learned_score=0;
    for(int i=0;i<729;++i){b.adjustment[i]=weights[i];b.learned_score+=int64_t(b.features[i])*weights[i];}
    return 1;
}
}
