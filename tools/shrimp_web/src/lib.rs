//! Shrimp's search (Cmiller132/hexo-bot at 6251fc6, MIT) as WebAssembly for the browser engine.
//!
//! The tree, the threat-space search, the featurizer and the evaluation cache are hexo-bot's own files, vendored
//! under `vendor/hexo-bot`; `search.rs` is the lockstep driver of hexo-bot's `ShrimpMctsSession.search` turned
//! inside out, so the page evaluates leaves with ONNX Runtime between steps instead of the search calling PyTorch.
//!
//! Calling sequence for one stone: `sh_begin`, then while `sh_step` returns a row count n > 0, read the n rows
//! (`sh_rows_*`), write the network's answers with `sh_fulfill` and call `sh_step` again; at 0 read `sh_result_*`.
//! Functions returning i32 return a negative number on error, with the message at `sh_error`.

#![allow(dead_code)]

#[path = "../vendor/hexo-bot/packages/shrimp/rust/src/cache.rs"]
mod cache;
#[path = "../vendor/hexo-bot/packages/shrimp/rust/src/constants.rs"]
mod constants;
#[path = "../vendor/hexo-bot/packages/shrimp/rust/src/features.rs"]
mod features;
mod search;
mod state;
#[path = "../vendor/hexo-bot/packages/shrimp/rust/src/support.rs"]
mod support;
#[path = "../vendor/hexo-bot/packages/shrimp/rust/src/threats_shared.rs"]
mod threats_shared;
#[path = "../vendor/hexo-bot/packages/shrimp/rust/src/tree.rs"]
mod tree;

use std::cell::RefCell;

use pyo3::prelude::*;
use search::Session;

thread_local! {
    static SESSIONS: RefCell<Vec<Option<Session>>> = const { RefCell::new(Vec::new()) };
    static ERROR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn fail(error: PyErr) -> i32 {
    ERROR.with(|e| *e.borrow_mut() = error.0.into_bytes());
    -1
}

fn with_session<T>(handle: i32, run: impl FnOnce(&mut Session) -> PyResult<T>) -> PyResult<T> {
    SESSIONS.with(|sessions| {
        let mut sessions = sessions.borrow_mut();
        let session = usize::try_from(handle)
            .ok()
            .and_then(|index| sessions.get_mut(index))
            .and_then(Option::as_mut)
            .ok_or_else(|| PyErr(format!("no shrimp session {handle}")))?;
        run(session)
    })
}

fn status(result: PyResult<i32>) -> i32 {
    result.unwrap_or_else(fail)
}

/// `len` bytes for the page to write into; release them with `sh_free`.
#[no_mangle]
pub extern "C" fn sh_alloc(len: usize) -> *mut u8 {
    let mut buffer = Vec::<u8>::with_capacity(len.max(1));
    let pointer = buffer.as_mut_ptr();
    std::mem::forget(buffer);
    pointer
}

/// # Safety
/// `pointer` and `len` must come from one `sh_alloc` call.
#[no_mangle]
pub unsafe extern "C" fn sh_free(pointer: *mut u8, len: usize) {
    drop(Vec::from_raw_parts(pointer, 0, len.max(1)));
}

/// The last error message (UTF-8, `sh_error_len` bytes).
#[no_mangle]
pub extern "C" fn sh_error() -> *const u8 {
    ERROR.with(|e| e.borrow().as_ptr())
}

#[no_mangle]
pub extern "C" fn sh_error_len() -> usize {
    ERROR.with(|e| e.borrow().len())
}

/// A new session: an empty tree store and an evaluation cache of `max_states`, as the driver's
/// `ShrimpMctsSession(max_states=...)`. `support_radius` is the featurizer radius the weights were trained with; the
/// first session fixes it for the module. `search_parity_mode` selects hexo-bot's parity divergences instead of its
/// production ones. Returns the session handle.
#[no_mangle]
pub extern "C" fn sh_session_new(max_states: u32, support_radius: i32, search_parity_mode: i32) -> i32 {
    status(Session::new(max_states as usize, support_radius, search_parity_mode != 0).map(|session| {
        SESSIONS.with(|sessions| {
            let mut sessions = sessions.borrow_mut();
            match sessions.iter().position(Option::is_none) {
                Some(index) => {
                    sessions[index] = Some(session);
                    index as i32
                }
                None => {
                    sessions.push(Some(session));
                    sessions.len() as i32 - 1
                }
            }
        })
    }))
}

#[no_mangle]
pub extern "C" fn sh_session_free(handle: i32) {
    SESSIONS.with(|sessions| {
        if let Some(slot) = usize::try_from(handle).ok().and_then(|i| sessions.borrow_mut().get_mut(i).map(Option::take)) {
            drop(slot);
        }
    });
}

/// Sets one search setting of the session by its keyword in hexo-bot's `ShrimpMctsSession.search` (`c_puct`,
/// `virtual_batch_size`, `fpu_reduction`, ...) or its `divergence_overrides` (`gumbel_m`, `lcb_z`, ...); booleans are
/// 0 or 1. `key` is UTF-8 of `key_len` bytes.
///
/// # Safety
/// `key` must point at `key_len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn sh_set(handle: i32, key: *const u8, key_len: usize, value: f64) -> i32 {
    let key = String::from_utf8_lossy(std::slice::from_raw_parts(key, key_len)).into_owned();
    status(with_session(handle, |session| session.set(&key, value).map(|_| 0)))
}

/// The position after `count` placements at `moves` (q, r pairs as i32): the player to move (0 or 1) plus 2 when
/// the game is over, plus 4 times the winner (0 or 1) when it was won.
///
/// # Safety
/// `moves` must point at `2 * count` readable i32.
#[no_mangle]
pub unsafe extern "C" fn sh_position(moves: *const i32, count: usize) -> i32 {
    let moves = if count == 0 { &[][..] } else { std::slice::from_raw_parts(moves, 2 * count) };
    status(search::position(moves))
}

/// Starts the search of one stone from the position of `count` placements at `moves` (q, r pairs as i32, the first
/// at the origin), with `visits` visits, the driver's `seed` and the session's `game_key`. A session reuses the tree
/// of its previous stone when that stone led here, as the driver's session does.
///
/// # Safety
/// `moves` must point at `2 * count` readable i32.
#[no_mangle]
pub unsafe extern "C" fn sh_begin(handle: i32, moves: *const i32, count: usize, visits: u32, seed: u64, game_key: u64) -> i32 {
    let moves = if count == 0 { &[][..] } else { std::slice::from_raw_parts(moves, 2 * count) };
    status(with_session(handle, |session| session.begin(moves, visits, seed, game_key).map(|_| 0)))
}

/// Runs the search until it needs network evaluations or is done: the number of rows to evaluate, or 0 when the
/// stone is chosen.
#[no_mangle]
pub extern "C" fn sh_step(handle: i32) -> i32 {
    status(with_session(handle, |session| session.step().map(|rows| rows as i32)))
}

/// Total support nodes over the rows of the current request.
#[no_mangle]
pub extern "C" fn sh_rows_nodes(handle: i32) -> i32 {
    status(with_session(handle, |session| Ok(session.request()?.nodes as i32)))
}

/// Row offsets into the nodes, rows + 1 i32.
#[no_mangle]
pub extern "C" fn sh_rows_offsets(handle: i32) -> *const i32 {
    with_session(handle, |session| Ok(session.request()?.offsets.as_ptr())).unwrap_or(std::ptr::null())
}

/// Legal cells per row (the first cells of each row), rows i32.
#[no_mangle]
pub extern "C" fn sh_rows_legal(handle: i32) -> *const i32 {
    with_session(handle, |session| Ok(session.request()?.legal.as_ptr())).unwrap_or(std::ptr::null())
}

/// Node features, nodes x 15 f32 (rounded through f16, as the driver sends them to its evaluator).
#[no_mangle]
pub extern "C" fn sh_rows_features(handle: i32) -> *const f32 {
    with_session(handle, |session| Ok(session.request()?.features.as_ptr())).unwrap_or(std::ptr::null())
}

/// Node coordinates, nodes x 2 i32 (q, r).
#[no_mangle]
pub extern "C" fn sh_rows_coords(handle: i32) -> *const i32 {
    with_session(handle, |session| Ok(session.request()?.coords.as_ptr())).unwrap_or(std::ptr::null())
}

/// Row-local neighbour indices, nodes x 6 i32, -1 where a neighbour is outside the support.
#[no_mangle]
pub extern "C" fn sh_rows_neighbours(handle: i32) -> *const i32 {
    with_session(handle, |session| Ok(session.request()?.neighbours.as_ptr())).unwrap_or(std::ptr::null())
}

/// Answers the current request: per row the decoded value in [-1, 1] and moves left in [0, 512], and per legal
/// cell (rows in order, each row's legal cells first) the raw policy logit.
///
/// # Safety
/// `values` and `moves_left` must point at one f32 per row and `logits` at one f32 per legal cell.
#[no_mangle]
pub unsafe extern "C" fn sh_fulfill(handle: i32, values: *const f32, moves_left: *const f32, logits: *const f32) -> i32 {
    status(with_session(handle, |session| {
        let request = session.request()?;
        let rows = request.legal.len();
        let cells = request.legal.iter().map(|&n| n as usize).sum();
        session
            .fulfill(
                std::slice::from_raw_parts(values, rows),
                std::slice::from_raw_parts(moves_left, rows),
                std::slice::from_raw_parts(logits, cells),
            )
            .map(|_| 0)
    }))
}

/// Visits completed by the current stone's search, for progress.
#[no_mangle]
pub extern "C" fn sh_completed(handle: i32) -> i32 {
    status(with_session(handle, |session| session.completed().map(|n| n as i32)))
}

/// After `sh_step` returned 0: the number of root moves in the result, the chosen stone first.
#[no_mangle]
pub extern "C" fn sh_result_count(handle: i32) -> i32 {
    status(with_session(handle, |session| Ok(session.result()?.moves.len() as i32 / 3)))
}

/// Root moves as (q, r, visits) i32 triples, the chosen stone first, then by visits.
#[no_mangle]
pub extern "C" fn sh_result_moves(handle: i32) -> *const i32 {
    with_session(handle, |session| Ok(session.result()?.moves.as_ptr())).unwrap_or(std::ptr::null())
}

/// Per root move of `sh_result_moves`: its Q for the side to move, then its share of the search's visits.
#[no_mangle]
pub extern "C" fn sh_result_stats(handle: i32) -> *const f32 {
    with_session(handle, |session| Ok(session.result()?.stats.as_ptr())).unwrap_or(std::ptr::null())
}

/// The root's value for the side to move, in [-1, 1].
#[no_mangle]
pub extern "C" fn sh_result_value(handle: i32) -> f32 {
    with_session(handle, |session| Ok(session.result()?.root_value)).unwrap_or(f32::NAN)
}

/// The visits the search added at the root (the driver's `visits`).
#[no_mangle]
pub extern "C" fn sh_result_visits(handle: i32) -> i32 {
    status(with_session(handle, |session| Ok(session.result()?.visits as i32)))
}
