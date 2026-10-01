// Run directly: g++ -std=c++20 -O2 tests/tt_injection.cpp -o build/tt_injection.exe
#include "../src/hexo.cpp"
#include <cassert>
#include <iostream>
#include <random>
int main() {
    Board b;b.make({0,0});Search search(10000,2,true);
    auto before=b.hash();auto features=b.features;
    Turn distant{{Cell{8,0},Cell{16,0}},2,0};
    auto plain=search.turns(b,false);
    auto turns=search.turns(b,false,distant);
    assert(turns.front().cells==distant.cells && turns.front().count==2);
    assert(turns.size()==plain.size());
    for(const auto& t:plain) assert(t.cells!=distant.cells);
    assert(b.hash()==before && b.features==features && b.history.size()==1);
    std::unordered_set<uint64_t> hashes;
    for(auto t:turns) {Restore restore(b);apply(b,t);assert(hashes.insert(b.hash()).second);}
    Turn reverse{{Cell{16,0},Cell{8,0}},2,0};
    auto rejected=search.turns(b,false,reverse);
    assert(rejected.front().cells==plain.front().cells);
    for(auto bad:std::vector<Turn>{{{Cell{8,0},Cell{8,0}},2,0}, {{Cell{8,0},Cell{}},1,0}, {{Cell{100,0},Cell{101,0}},2,0}}) {
        auto actual=search.turns(b,false,bad);
        assert(actual.size()==plain.size());
        for(size_t i=0;i<plain.size();++i) assert(actual[i].cells==plain[i].cells);
    }
    auto existing=plain.back();auto duplicate=search.turns(b,false,existing);
    assert(duplicate.front().cells==existing.cells);
    hashes.clear();for(auto t:duplicate) {Restore restore(b);apply(b,t);assert(hashes.insert(b.hash()).second);}
    // Four opposing stones create immediate completions. Distant hints cannot
    // displace mandatory defenses, even when both placements are legal.
    Board threat;
    for(Cell c:std::vector<Cell>{{0,0},{0,2},{1,2},{0,-2},{2,-2},{2,2},{3,2}}) threat.make(c);
    assert(!threat.completions(1-threat.player).empty());
    auto defended=search.turns(threat,false,distant);
    for(auto t:defended) {Restore restore(threat);int side=threat.player;apply(threat,t);assert(threat.completions(1-side).empty());}
    // Current-iteration hints cannot change the frozen admission set. Cached
    // ordering scores must never substitute for a searched value.
    search.frozen_hints=search.tt;distant.score=123456;
    auto key=b.hash();search.frozen_hints[key&(search.tt.size()-1)]={key,distant};
    search.tt[key&(search.tt.size()-1)]={key,reverse};
    int value=search.negamax(b,1,-mate-1,mate+1);assert(value!=123456);
    assert(b.hash()==before && b.features==features);
    b.make({1,0});auto partial=search.turns(b,false,distant);
    for(auto t:partial) assert(t.count==1);
    // A quiet second stone can sustain a forced attack for several turns.
    // The proof must remain valid when replayed and leave cached gains intact.
    std::vector<Cell> opening{{0,0},{0,8},{2,8},{1,0},{2,0},{4,8},{6,8}};
    Board open;
    for(auto c:opening) {assert(open.legal(c));open.make(c);}
    auto open_key=open.hash();auto open_features=open.features;
    Search forcing(5000,16);forcing.proof_deadline=forcing.deadline;
    CandidateGuard open_cache(open);
    auto gains=Search::candidate_scores(open);
    int root=forcing.prove(open,6);
    assert(root>=0 && forcing.replay(open,root));
    assert(open.hash()==open_key && open.features==open_features);
    assert(Search::candidate_scores(open)==gains);
    Board blocked;
    auto defended_opening=opening;defended_opening[5]=forcing.proof[root].attack.cells[0];
    for(auto c:defended_opening) {assert(blocked.legal(c));blocked.make(c);}
    auto blocked_key=blocked.hash();
    assert(!forcing.replay(blocked,root) && blocked.hash()==blocked_key);
    Board counter;
    for(Cell c:std::vector<Cell>{{0,0},{0,8},{1,8},{1,0},{2,0},{2,8},{3,8}}) {
        assert(counter.legal(c));counter.make(c);
    }
    auto counter_key=counter.hash();
    assert(!forcing.replay(counter,root) && counter.hash()==counter_key);
    Board spare;
    for(Cell c:std::vector<Cell>{{0,0},{-1,0},{0,5},{1,0},{2,0},{3,3},{-3,5},{3,0},{4,0}}) {
        assert(spare.legal(c));spare.make(c);
    }
    std::vector<Turn> free_replies;
    assert(!forcing.forced_replies(spare,0,free_replies));
    // Incremental second-stone ranking must equal a fresh full-board ranking,
    // including cells whose only promising line was just blocked.
    std::mt19937 rng(20261001);
    for(int trial=0;trial<12;++trial) {
        Board position;
        for(int step=0;step<25 && position.winner<0;++step) {
            auto legal=position.legal_moves();
            position.make(legal[rng()%legal.size()]);
            if(position.winner>=0 || position.remaining!=2) continue;
            if(trial&1) {
                for(int i=1;i<729;++i) position.adjustment[i]=(i*19)%101-50;
                position.learned_score=0;
                for(int i=1;i<729;++i) position.learned_score+=int64_t(position.features[i])*position.adjustment[i];
            }
            auto base=Search::candidate_scores(position);
            auto by_cell=[](const auto& a,const auto& z){return a.second<z.second;};
            std::sort(base.begin(),base.end(),by_cell);
            CandidateGuard cache(position);
            auto check_cache=[&]() {
                auto cached=Search::candidate_scores(position);
                position.candidates=nullptr;
                auto fresh=Search::candidate_scores(position);
                position.candidates=&cache.cache;
                std::sort(cached.begin(),cached.end(),by_cell);
                std::sort(fresh.begin(),fresh.end(),by_cell);
                assert(cached==fresh);
            };
            check_cache();
            for(auto first:Search::candidates(position,4)) {
                Restore restore(position);position.make(first);
                if(position.winner>=0) continue;
                check_cache();
                auto fresh=Search::candidate_scores(position);
                auto incremental=Search::following_scores(position,base,first);
                std::sort(fresh.begin(),fresh.end(),by_cell);
                std::sort(incremental.begin(),incremental.end(),by_cell);
                assert(fresh==incremental);
                for(auto second:Search::candidates(position,4)) {
                    auto scalar=position.placed_score(second,position.player);
                    int side=position.player;Restore undo(position);position.make(second);
                    assert(scalar==position.score(side));
                    if(position.winner<0) check_cache();
                }
            }
        }
    }
    std::cout<<"Native search checks passed\n";
}
