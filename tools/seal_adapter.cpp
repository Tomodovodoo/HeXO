// Adapter only. Seal remains in its external source checkout.
#include "engine.h"
#include "pattern_data.h"
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
extern "C" EXPORT int seal_move(const int* cells,int n,int player,int remaining,int ms,int* out) {
    static opt::MinimaxBot bot;
    static bool initialized=false;
    if(!initialized) {bot.load_patterns(PATTERN_VALUES,PATTERN_COUNT,PATTERN_EVAL_LENGTH);initialized=true;}
    GameState game;
    for(int i=0;i<n;++i) {
        // Seal has a fixed 140x140 backing array. Avoid its unchecked edge accesses.
        if(std::abs(cells[3*i])>55 || std::abs(cells[3*i+1])>55) return -1;
        game.cells.push_back({cells[3*i],cells[3*i+1],int8_t(cells[3*i+2]+1)});
    }
    game.cur_player=int8_t(player+1);game.moves_left=int8_t(remaining);game.move_count=n;
    bot.time_limit=ms/1000.0;
    auto move=bot.get_move(game);
    out[0]=move.q1;out[1]=move.r1;out[2]=move.q2;out[3]=move.r2;
    return move.num_moves;
}
