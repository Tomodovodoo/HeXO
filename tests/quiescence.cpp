// g++ -std=c++20 -O2 tests/quiescence.cpp -o build/quiescence.exe
#include "../src/hexo.cpp"
#include <cassert>
#include <iostream>
int main() {
    Board quiet;quiet.make({0,0});Search search(10000,16);
    auto key=quiet.hash();auto features=quiet.features;
    assert(search.quiescence(quiet,2,-mate-1,mate+1)==quiet.score(quiet.player));
    assert(search.nodes==1);
    Board threat;
    for(Cell c:std::vector<Cell>{{0,0},{0,2},{1,2},{0,-2},{2,-2},{2,2},{3,2}}) threat.make(c);
    assert(!threat.completions(1-threat.player).empty());
    auto before=threat.hash();auto original=threat.features;auto size=threat.history.size();
    Search leaf(10000,16);int static_value=leaf.quiescence(threat,0,-mate-1,mate+1);
    assert(leaf.nodes==1 && static_value==threat.score(threat.player));
    Search extended(10000,16);extended.quiescence(threat,2,-mate-1,mate+1);
    assert(extended.nodes>1);
    assert(threat.hash()==before && threat.features==original && threat.history.size()==size);
    threat.make({0,1});before=threat.hash();original=threat.features;size=threat.history.size();
    assert(threat.remaining==1);
    Search partial(10000,16);partial.quiescence(threat,2,-mate-1,mate+1);
    assert(threat.hash()==before && threat.features==original && threat.history.size()==size);
    Search expired(1,16);expired.deadline=Clock::now();bool timed_out=false;
    try {expired.quiescence(threat,2,-mate-1,mate+1);} catch(const Timeout&) {timed_out=true;}
    assert(timed_out && threat.hash()==before && threat.features==original);
    assert(quiet.hash()==key && quiet.features==features);
    std::cout<<"Quiet leaves, bounded defense extension, partial turn, restoration and deadline passed\n";
}
