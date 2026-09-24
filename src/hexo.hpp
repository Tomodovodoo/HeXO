#pragma once
#include <cstdint>

#ifdef _WIN32
#define HX_API __declspec(dllexport)
#else
#define HX_API __attribute__((visibility("default")))
#endif

extern "C" {
struct HxCell { int64_t q, r; int32_t player; };
struct HxResult {
    int64_t q1, r1, q2, r2;
    uint64_t nodes;
    double elapsed_ms;
    int32_t count, score, depth;
};
HX_API void* hx_new();
HX_API void hx_free(void*);
HX_API int hx_play(void*, int64_t q, int64_t r);
HX_API int hx_undo(void*);
HX_API int hx_player(void*);
HX_API int hx_remaining(void*);
HX_API int hx_winner(void*);
HX_API int hx_size(void*);
HX_API int hx_cell(void*, int index, HxCell*);
HX_API int hx_legal(void*, int64_t q, int64_t r);
// Returns required capacity. Writes at most capacity coordinates.
HX_API int hx_moves(void*, HxCell* output, int capacity);
HX_API int hx_search(void*, int milliseconds, int max_depth, int width, HxResult*);
HX_API uint64_t hx_hash(void*);
HX_API int hx_evaluate(void*);
}
