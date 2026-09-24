// Run directly: g++ -std=c++20 -O2 tests/tt_injection.cpp -o build/tt_injection.exe
#include "../src/hexo.cpp"
#include <cassert>
#include <iostream>
int main() {
    Board b;b.make({0,0});Search search(10000,2);
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
    // Frozen move source is distinct from mutable score entries. A bogus exact
    // score must never bypass search in injection mode.
    search.inject_tt=true;search.frozen_hints=search.tt;
    auto key=b.hash();search.frozen_hints[key&(search.tt.size()-1)]={key,8,123456,0,distant};
    search.tt[key&(search.tt.size()-1)]={key,8,123456,0,reverse};
    int value=search.negamax(b,1,-mate-1,mate+1);assert(value!=123456);
    assert(b.hash()==before && b.features==features);
    b.make({1,0});auto partial=search.turns(b,false,distant);
    for(auto t:partial) assert(t.count==1);
    std::cout<<"TT admission, legality, deduplication, mandatory cover, phase, frozen hints and score isolation passed\n";
}
