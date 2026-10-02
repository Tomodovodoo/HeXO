// Run directly: g++ -std=c++20 -O2 tests/tt_injection.cpp -o build/tt_injection.exe
#include "../src/hexo.cpp"
#include <cassert>
#include <iostream>
#include <random>
#include <set>
int main() {
    // Filler generation must allow cancellation during dense scans and sparse
    // neighborhood generation/sorting, without changing the legal set or board.
    for(int span:{128,264}) {
        Board spread;
        for(int i=0;i<=span;++i) spread.make({8*i,0});
        for(int i=1;i<=span;++i) spread.make({8*span,8*i});
        const auto hash=spread.hash();const auto features=spread.features;
        const auto expected=spread.legal_moves();int checks=0;
        assert(spread.legal_moves([&]{++checks;})==expected && checks>1000);
        for(int stop:{1,checks/2,checks-1}) {
            int seen=0;bool cancelled=false;
            try {spread.legal_moves([&]{if(++seen==stop) throw Timeout{};});}
            catch(const Timeout&) {cancelled=true;}
            assert(cancelled && seen==stop && spread.hash()==hash && spread.features==features);
        }
    }
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
    // Lazy first-stone batches must preserve every distinct attack, while
    // spending the proof budget only once on each resulting position.
    std::unordered_set<uint64_t> independent,shared,returned;
    size_t independent_count=0,shared_count=0;
    for(auto first:Search::candidates(open,16)) {
        for(const auto& t:forcing.turns(open,false,{},true,&first)) {
            Restore restore(open);apply(open,t);independent.insert(open.hash());++independent_count;
        }
        for(const auto& t:forcing.turns(open,false,{},true,&first,&shared)) {
            Restore restore(open);apply(open,t);assert(returned.insert(open.hash()).second);++shared_count;
        }
    }
    assert(independent_count>shared_count && independent==returned && shared==returned);
    assert(open.hash()==open_key && Search::candidate_scores(open)==gains);
    // A saved Seal attack wins even though either block leaves a free stone.
    // Enumerate every legal filler after both independently known blocks, then
    // replay its recorded strategy without using the support shortcut.
    Board free_defense;
    for(Cell c:std::vector<Cell>{
        {0,0},{1,-2},{3,-3},{3,-1},{2,0},{4,-2},{4,-3},{1,0},{4,0},{3,0},
        {2,-3},{-1,0},{-1,-3},{-3,0},{-1,-2},{5,-3},{2,-2},{4,-5},
        {-3,-1},{-3,3},{1,-1},{-1,1},{-2,-1},{4,-1},{2,-1}
    }) {assert(free_defense.legal(c));free_defense.make(c);}
    const auto free_key=free_defense.hash();const auto free_features=free_defense.features;
    CandidateGuard free_cache(free_defense);const auto free_gains=Search::candidate_scores(free_defense);
    Search free_proof(10000,16);free_proof.proof_deadline=free_proof.deadline;
    Turn free_attack{{Cell{0,-1},Cell{6,-1}},2,0};
    int free_root=free_proof.free_attack(free_defense,free_attack,6);assert(free_root>=0);
    {
        Restore root_position(free_defense);apply(free_defense,free_attack);
        const auto threats=free_defense.completions(1);
        assert((threats==std::vector<std::vector<Cell>>{{{3,-4},{5,-6}}}));
        std::set<std::pair<Cell,Cell>> expected;
        for(Cell block:std::vector<Cell>{{3,-4},{5,-6}}) {
            Restore first(free_defense);free_defense.make(block);
            for(Cell filler:free_defense.legal_moves()) expected.insert(std::minmax(block,filler));
        }
        for(const auto& [reply,child]:free_proof.proof[free_root].replies) {
            assert(reply.count==2 && expected.erase(std::minmax(reply.cells[0],reply.cells[1]))==1);
            Restore response(free_defense);
            for(auto cell:reply.cells) {assert(free_defense.legal(cell));free_defense.make(cell);}
            assert(free_proof.replay(free_defense,child));
        }
        assert(expected.empty());
    }
    assert(free_defense.hash()==free_key && free_defense.features==free_features);
    assert(Search::candidate_scores(free_defense)==free_gains);
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
    // A fresh Board and Search on every request must still finish a proven
    // win, even when stateless reconstruction changes earlier placement order.
    WinningPlan plan;forcing.remember(open,root,plan);
    Board played;
    for(auto c:opening) played.make(c);
    int resumed_turns=0;
    for(int turn=0;turn<8 && played.winner<0;++turn) {
        Board position;
        auto order=played.history;
        if(turn%2==0) std::swap(order[3],order[4]);
        for(const auto& step:order) {assert(position.legal(step.c));position.make(step.c);}
        const auto key=position.hash();const auto features=position.features;
        Search continuation(5000,16);
        auto result=continuation.run(position,1,0,0,&plan);
        assert(result.score==mate && continuation.proof_nodes==0);
        if(!continuation.proof.empty()) ++resumed_turns;
        assert(position.hash()==key && position.features==features);
        Turn move{{Cell{result.q1,result.r1},Cell{result.q2,result.r2}},result.count,0};
        for(int i=0;i<move.count;++i) {assert(played.legal(move.cells[i]));played.make(move.cells[i]);}
        if(played.winner>=0) break;
        std::vector<Turn> replies;
        assert(continuation.forced_replies(played,plan.attacker,replies));
        // With no covering pair, any legal defense leaves an immediate win.
        auto defense=replies.empty()?continuation.turns(played,false).front():replies.back();
        if(defense.count==2) std::swap(defense.cells[0],defense.cells[1]);
        for(int i=0;i<defense.count;++i) {assert(played.legal(defense.cells[i]));played.make(defense.cells[i]);}
    }
    assert(played.winner==plan.attacker && resumed_turns>=2);
    Search unrelated(5000,16);unrelated.proof_deadline=unrelated.deadline;
    assert(unrelated.resume(counter,plan)<0 && counter.hash()==counter_key);
    Board spare;
    for(Cell c:std::vector<Cell>{{0,0},{-1,0},{0,5},{1,0},{2,0},{3,3},{-3,5},{3,0},{4,0}}) {
        assert(spare.legal(c));spare.make(c);
    }
    std::vector<Turn> free_replies;
    assert(!forcing.forced_replies(spare,0,free_replies));
    for(auto* source:{&b,&threat,&open,&blocked,&counter,&spare}) {
        auto& position=*source;
        const auto original_key=position.hash();const auto original_features=position.features;
        Search leaf(10000,16);
        if(position.winner<0 && !leaf.immediate(position).count) {
            const int side=position.player;int exact=-mate-1;
            auto moves=leaf.turns(position,false);
            for(const auto& move:moves) {
                Restore restore(position);apply(position,move);
                int value=position.winner==side?mate:
                    leaf.immediate(position).count?-mate:
                    leaf.unavoidable_loss(position)?mate:position.score(side);
                assert(!move.leaf_score_exact || value==move.score);
                exact=std::max(exact,value);
            }
            for(const auto& bounds:std::vector<std::pair<int,int>>{
                {-mate-1,mate+1},{exact-1,exact},{exact,exact+1},{-1,0},{0,1},{-2400,2400}}) {
                const auto [alpha,beta]=bounds;Search bounded(10000,16);
                int result=bounded.negamax(position,1,alpha,beta);
                assert(exact>=beta ? result>=beta && result<=exact :
                       exact<=alpha ? result<=alpha && result>=exact : result==exact);
                assert(position.hash()==original_key && position.features==original_features);
            }
        }
    }
    // Development losses: an attack hidden below defensive first-stone
    // rankings, and a winning continuation with no unblocked three-stone line.
    // Short continuations must also survive the proof work limit. All strategies
    // were checked separately by the raw-board Python verifier. These also
    // cover a blocking first stone, reuse across forced defenses, and ordering
    // the mandatory replies before spending the proof budget on them.
    for(const auto& history:std::vector<std::vector<Cell>>{
        {
            {0,0},{1,1},{-1,2},{-1,1},{3,-3},{1,-1},{1,0},{1,-3},{2,-3},{4,-3},
            {0,2},{-2,2},{0,-3},{1,2},{-1,-3},{1,3},{-3,3},{-4,4},{2,1},{-2,4},
            {-2,3}
        },
        {
            {0,0},{1,0},{0,2},{1,-3},{-1,-1},{-1,2},{0,1},{2,-1},{1,-1},{-2,2},
            {-3,2},{-4,2},{2,2},{-3,4},{-3,1},{-4,5},{-3,3},{-1,1},{-2,1},{-4,1},
            {2,1},{-2,-1},{-2,0},{-2,-3},{-2,3},{-4,3},{-1,0},{-6,5},{0,-1},{3,-1},
            {2,0},{0,-3},{-1,-3},{-3,-3},{3,-3},{-5,5},{-7,5},{-9,5},{-3,5},{-1,3},
            {0,-2},{0,-4},{0,3},{3,-5},{-3,-2},{2,-4},{0,4},{0,6},{4,-2},{1,-4},
            {3,-4},{-1,-4},{4,-4},{-1,4},{-2,4},{-4,4},{2,4},{3,0},{4,0},{3,-2},
            {6,0},{1,-2},{-5,0},{0,-5},{-1,-5},{1,-6},{-1,-2},{4,-5},{4,-3},{2,-5},
            {4,-6},{3,3},{4,2},{1,5},{7,-1},{6,-5},{-6,6},{-8,8},{7,-6},{-6,4},
            {-6,7},{-6,2},{-6,8},{-8,6},{-1,-6},{-9,7},{-1,-8},{2,6},{7,-5},{2,5},
            {8,-5}
        },
        {
            {0,0},{-2,-2},{-2,1},{-2,2},{-1,1},{2,-2},{-4,4},{-1,2},{-1,0},{-1,-2},
            {0,-2},{1,-2},{1,2},{-4,2},{-4,-2},{1,0},{-3,-2},{-4,0},{-4,1},{-4,-1},
            {-4,3},{-2,-1},{-3,0},{-5,2},{0,-3}
        },
        {
            {0,0},{-2,-2},{-2,1},{0,1},{0,2},{0,-1},{-2,-1},{-2,2},{0,5},{0,3},
            {-1,2},{-3,3},{-5,5},{-1,1},{-1,-1},{-1,0},{-4,4}
        },
        {
            {0,0},{2,-2},{-7,7},{0,-2},{0,-1},{0,-3},{-6,6},{0,3},{-5,5},{0,1},
            {1,-3},{2,-3},{-3,3},{-2,1},{-2,3},{4,-5},{3,-4},{1,-2},{-6,7},{7,-8},
            {1,-1},{-1,1},{5,-6},{2,-1},{-1,-1},{-2,-1},{3,-1},{-3,1},{-2,0},{-5,3},
            {-3,0},{1,-4},{2,-4},{0,-4},{-6,5},{-6,4},{4,-4},{5,-4},{-8,7},{4,-6},
            {-5,7},{-7,6},{-5,4},{-9,8},{-3,2}
        },
        {
            {0,0},{1,1},{-1,2},{-1,1},{3,-3},{1,-1},{1,0},{1,-3},{2,-3},{1,2},
            {4,-3},{1,3},{-1,-3},{-2,-3},{2,1},{0,3},{2,3},{0,2},{-1,3},{3,3},
            {-2,2}
        },
        {
            {0,0},{3,1},{-2,-6},{0,-1},{-1,1},{0,4},{-1,5},{1,-1},{-1,-1},{2,-2},
            {1,3}
        }
    }) {
        Board position;
        for(auto c:history) {assert(position.legal(c));position.make(c);}
        const auto key=position.hash();const auto features=position.features;
        Search continuation(5000,16);
        auto result=continuation.run(position,12);
        assert(result.score==mate && !continuation.proof.empty());
        continuation.proof_deadline=continuation.deadline;
        assert(continuation.replay(position,int(continuation.proof.size())-1));
        assert(position.hash()==key && position.features==features && position.history.size()==history.size());
    }
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
                for(const auto& slot:position.windows.slots) if(slot.hash) {
                    unsigned gaps=0;std::vector<Cell> empty;
                    for(int k=0;k<6;++k) {
                        Cell cell=slot.start+axes[slot.axis]*k;
                        if(position.at(cell)<0) {gaps|=1u<<k;empty.push_back(cell);}
                    }
                    assert(slot.data.empty==gaps && position.empty(slot.key())==empty);
                    assert(std::is_sorted(empty.begin(),empty.end()));
                }
                auto cached=Search::candidate_scores(position);
                position.candidates=nullptr;
                auto fresh=Search::candidate_scores(position);
                position.candidates=&cache.cache;
                std::sort(cached.begin(),cached.end(),by_cell);
                std::sort(fresh.begin(),fresh.end(),by_cell);
                assert(cached==fresh);
            };
            check_cache();
            if(step%5==0) {
                const auto original_key=position.hash();const auto original_features=position.features;
                Search leaf(10000,16);
                if(position.winner<0 && !leaf.immediate(position).count) {
                    const int side=position.player;int exact=-mate-1;
                    auto moves=leaf.turns(position,false);
                    for(const auto& move:moves) {
                        Restore restore(position);apply(position,move);
                        int value=position.winner==side?mate:
                            leaf.immediate(position).count?-mate:
                            leaf.unavoidable_loss(position)?mate:position.score(side);
                        assert(!move.leaf_score_exact || value==move.score);
                        exact=std::max(exact,value);
                    }
                    for(const auto& bounds:std::vector<std::pair<int,int>>{
                        {-mate-1,mate+1},{exact-1,exact},{exact,exact+1},{-1,0},{0,1},{-2400,2400}}) {
                        const auto [alpha,beta]=bounds;Search bounded(10000,16);
                        int result=bounded.negamax(position,1,alpha,beta);
                        assert(exact>=beta ? result>=beta && result<=exact :
                               exact<=alpha ? result<=alpha && result>=exact : result==exact);
                        assert(position.hash()==original_key && position.features==original_features);
                    }
                }
            }
            if(step==10 && trial<4) {
                auto key=position.hash();auto features=position.features;
                Search cached_search(10000,4);
                int cached_value=cached_search.negamax(position,2,-mate-1,mate+1);
                assert(position.hash()==key && position.features==features);
                check_cache();
                position.candidates=nullptr;
                Search fresh_search(10000,4);
                int fresh_value=fresh_search.negamax(position,2,-mate-1,mate+1);
                position.candidates=&cache.cache;
                assert(cached_value==fresh_value && cached_search.nodes==fresh_search.nodes);
                assert(position.hash()==key && position.features==features);
                check_cache();
            }
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
    for(int sign:{-1,1}) {
        Board boundary;
        for(int i=0;i<12;++i) boundary.make({int64_t(sign)*i*8,0});
        auto fresh=Search::candidate_scores(boundary);
        auto by_cell=[](const auto& a,const auto& z){return a.second<z.second;};
        std::sort(fresh.begin(),fresh.end(),by_cell);
        CandidateGuard cache(boundary);
        auto key=boundary.hash();auto features=boundary.features;
        Search search(10000,4);search.negamax(boundary,2,-mate-1,mate+1);
        auto cached=Search::candidate_scores(boundary);
        std::sort(cached.begin(),cached.end(),by_cell);
        assert(cached==fresh && boundary.hash()==key && boundary.features==features);
    }
    std::cout<<"Native search checks passed\n";
}
