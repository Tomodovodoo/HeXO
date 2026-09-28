//! Fully-forcing (VCF) threat-space solver for HeXO + deep proof-search drivers.
//!
//! Pure combinatorial search with no neural-net, MCTS, or PyO3 dependencies.

pub mod forcing;
pub mod position;
pub mod prover;
pub mod vct_probe;

pub use forcing::{DefenseAnalysis, DefenseVerdict};
pub use prover::certificate::{
    ProofCertificate, ProofNode, ProofResponse, ProofSummary, verify as verify_proof_certificate,
};
pub use position::{
    DEFAULT_PN2_NODES, PositionSolveResult, SolverEngine, SolverPosition, is_game_valid_board,
    solve_defense_from_position, solve_defense_verdict_from_position,
    solve_from_position, solve_from_position_with_stats, solve_from_position_with_stats_configured,
    solve_wide_from_position, solve_wide_from_position_with_stats,
    solve_wide_from_position_with_stats_configured,
};
