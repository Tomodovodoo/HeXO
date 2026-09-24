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
struct HxTurn { int64_t q1,r1,q2,r2; int32_t count,score; };
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
// Optional root-only admission; zeros retain the original candidate tree.
HX_API int hx_search_root(void*, int milliseconds, int max_depth, int width,
    int root_seconds, int root_turns, HxResult*);
// Experimental frozen-iteration TT move admission; no TT score cutoffs.
HX_API int hx_search_tt(void*, int milliseconds, int max_depth, int width,
    int root_seconds, int root_turns, HxResult*);
HX_API int hx_turns(void*, int width, int root_seconds, int root_turns, HxTurn*, int capacity);
HX_API uint64_t hx_hash(void*);
HX_API int hx_evaluate(void*);
HX_API int hx_features(void*, int32_t* output, int capacity);
HX_API int hx_load_table(void*, const int32_t* weights, int count);
HX_API void* hx_model_load(const char* utf8_path);
HX_API void hx_model_free(void* model);
HX_API const char* hx_model_error();
HX_API int hx_set_model(void* board, void* model);
// Sparse finite-support centers, sorted by coordinate; codes use absolute colors.
// Coordinates have 2*capacity entries; codes have 3*capacity entries.
HX_API int hx_nnue_centers(void*, int64_t* coordinates, int32_t* codes, int capacity);
HX_API int hx_nnue_context(void*, float* output);
HX_API int hx_nnue_inputs(void*, float* output, int capacity);
HX_API int hx_nnue_policy_features(void*, int64_t q, int64_t r, int32_t* codes, float* pair);
// Ordered n-by-2 coordinates to n-by-3 codes and n-by-4 pair features.
// Returns 0 on any illegal coordinate; outputs must be discarded on failure.
HX_API int hx_nnue_policy_batch(void*, const int64_t* coordinates, int count, int32_t* codes, float* pairs);
HX_API float hx_nnue_rank(void*, int64_t q, int64_t r);
HX_API int hx_candidates(void*, int limit, HxCell* output, int capacity);
// Quiet-turn exploration must defer to search when either side can finish now.
HX_API int hx_tactical(void*);
}
