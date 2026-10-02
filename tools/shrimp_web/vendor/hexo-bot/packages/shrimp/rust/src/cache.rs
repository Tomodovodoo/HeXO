//! Bounded evaluation cache and request dedup.
//!
//! FIFO insertion-order eviction bounded by a max-states limit. Evaluations are
//! Arc-shared so tree nodes reference the same prior vectors. Evaluations carry
//! an optional `moves_left` field.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use hexo_engine::{HexoState as RustHexoState, PackedCoord};
use hexo_utils::{hash_state, StateHash};

pub const EVAL_CACHE_MAX_STATES: usize = 262_144;

#[derive(Clone, Debug)]
pub struct RustEvaluation {
    /// Scalar value from the evaluated state's side-to-move perspective.
    pub value: f32,
    /// Legal move count; equals `priors.len()`.
    pub legal_action_count: usize,
    /// One prior per legal move, ordered descending by prior then ascending id.
    pub priors: Vec<(PackedCoord, f32)>,
    /// Moves-left estimate in decisions [0, 512], or None when the reply omitted it.
    pub moves_left: Option<f32>,
    /// Raw pre-softmax policy logits per legal move, aligned to the same legal
    /// action set as `priors` (not sorted or normalized). `None` when the reply
    /// omitted `priors_logits_bytes`. Used by the Gumbel target, root sampler,
    /// and non-root selection.
    pub logits: Option<Vec<(PackedCoord, f32)>>,
}

#[derive(Clone, Debug, Default)]
pub struct EvaluationStats {
    pub requested_states: usize,
    pub cache_hits: usize,
    pub duplicate_hits: usize,
    pub unique_states: usize,
    pub evaluator_chunks: usize,
    pub encoded_states: usize,
    pub encoded_nodes: usize,
    pub max_chunk_states: usize,
    pub input_bytes: usize,
    pub value_bytes: usize,
    pub prior_bytes: usize,
    pub cache_inserts: usize,
    pub cache_size_peak: usize,
    pub encoding_seconds: f64,
    pub evaluator_seconds: f64,
    pub parse_seconds: f64,
}

#[derive(Clone, Debug, Default)]
pub struct RustEvaluationCache {
    entries: HashMap<StateHash, Arc<RustEvaluation>>,
    insertion_order: VecDeque<StateHash>,
}

impl RustEvaluationCache {
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn get(&self, key: &StateHash) -> Option<Arc<RustEvaluation>> {
        self.entries.get(key).map(Arc::clone)
    }

    pub fn insert_bounded(
        &mut self,
        key: StateHash,
        evaluation: Arc<RustEvaluation>,
        max_states: usize,
    ) {
        if self.entries.contains_key(&key) {
            self.entries.insert(key, evaluation);
            return;
        }
        debug_assert!(max_states > 0);
        while self.entries.len() >= max_states {
            let Some(evicted) = self.insertion_order.pop_front() else {
                break;
            };
            if self.entries.remove(&evicted).is_some() {
                break;
            }
        }
        self.insertion_order.push_back(key);
        self.entries.insert(key, evaluation);
    }
}

pub type SharedEvaluationCache = Arc<Mutex<RustEvaluationCache>>;
pub type SharedEvaluationStats = Arc<Mutex<EvaluationStats>>;

pub fn new_shared_evaluation_cache() -> SharedEvaluationCache {
    Arc::new(Mutex::new(RustEvaluationCache::default()))
}

pub fn new_shared_evaluation_stats() -> SharedEvaluationStats {
    Arc::new(Mutex::new(EvaluationStats::default()))
}

#[inline]
pub fn lock_cache(cache: &SharedEvaluationCache) -> std::sync::MutexGuard<'_, RustEvaluationCache> {
    cache.lock().expect("evaluation cache mutex poisoned")
}

#[inline]
pub fn lock_stats(stats: &SharedEvaluationStats) -> std::sync::MutexGuard<'_, EvaluationStats> {
    stats.lock().expect("evaluation stats mutex poisoned")
}

pub fn state_hash(state: &RustHexoState) -> StateHash {
    hash_state(state)
}

pub struct RustEvaluationRequest<'a> {
    pub state: &'a RustHexoState,
    pub state_hash: StateHash,
}
