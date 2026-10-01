# Timed matches and bot API

Design dated 2026-10-01, against main `3ab726d`. This document specifies the implementation. No timed matches, benchmarks, tests or runtime changes were made for this design. Existing compute jobs remain paused.

## Time controls and clock ownership

The match host owns both clocks and the game result. Bubble chooses how to spend its allowance. The same clock code serves automated matches and human play.

Support Absolute and Fischer controls. The command-line form `180` means 180 seconds per player; `180+2` adds two seconds after each completed turn. JSON uses explicit milliseconds:

```json
{"base_ms": 180000, "increment_ms": 2000}
```

A turn contains both placements, or one placement that immediately wins. The clock runs throughout the turn. There is one increment per completed turn, with no increment between stones. The automatic origin stone and an opening supplied before play consume no clock and earn no increments. Both players start with the specified base time after setup and model warmup.

Use a monotonic clock internally and retain nanosecond precision in the event record. UTC timestamps identify recordings; they never determine elapsed time. When the host starts a turn, it records the player's balance and the start timestamp. It charges elapsed time through receipt of the complete response, including engine queueing and transport. Receipt stops the clock only if the response is legal. Validation happens before committing any placement.

A response received at or after the player's deadline loses on time, even if it contains a winning move. Increment cannot rescue an expired clock. The host records timeout as a loss and rejects late replies. There is no hidden grace period. The engine leaves a return allowance inside its available time.

For a legal response, subtract elapsed time, add one increment and commit the complete turn. Start the opponent's clock when the host dispatches their turn request, or publishes their turn to a human client. Human clients can submit one placement at a time. After a non-winning first stone, the board contains a half-turn and the same clock continues without an increment. Engine replies can submit the complete turn atomically.

An optional per-turn cap can restrict the clock allowance further. A fixed move-time match uses that cap with no game clock. Unlimited analysis has no game clock and requires explicit cancellation. These modes are recorded separately.

## One shared timed engine

Use one turn implementation for [human play](../python/play.py), the [Six adapter](../python/six_engine.py), HTTTX and timed matches. Keep `DensePlayer` responsible for model selection and turn search. Add a small deadline and clock module, rather than a second search implementation.

The current `DensePlayer.turn(milliseconds=...)` ignores that argument. Its root solver can use a separate ten-second allowance, and each placement starts a fresh fixed-simulation search. `NeuralSearch.search(milliseconds=...)` already has cooperative deadline checks, but synchronous inference can finish after those checks. The Six server handles input synchronously, so its existing `stop` cannot interrupt a running turn. These are the first implementation changes.

Load each model, initialize its solver and warm inference before starting the game clock. Keep the evaluator, evaluation cache and search worker alive across turns. Reuse the tree through the first and second placements, and through confirmed subsequent moves. Model changes start a new session.

The API or engine controller remains responsive while its persistent worker searches. Use a persistent search process so blocking solver or inference work cannot block the controller's timer or transport. Give every request a generation identifier. The worker publishes its latest complete legal turn to the controller. At the internal return deadline the controller sends that turn without waiting for another inference batch. Cancel further work and ignore results from an abandoned generation. A worker still finishing a batch cannot start another search concurrently.

Create a complete legal fallback before neural or solver work. Initially this can be a deterministic legal turn. Replace it with stronger completed candidates as search progresses. A selected first stone is publishable only with a legal second stone, unless the first stone wins. Position preparation, fallback construction, proof verification and serialization all consume the same allowance.

Native calls and GPU kernels cannot guarantee completion at an arbitrary instant. The independent host deadline enforces the match result. An engine that overruns can lose on time; the controller and fallback prevent normal cooperative overruns from requiring a late reply. Do not advertise enforced time limits until this path works end to end.

## Spending the clock

Start with a small allocation rule. Let `T` be our remaining clock, `I` our increment and `H` the estimated remaining turns by this player. Use `H = 20` initially, configurable and recorded in the engine settings. This is an initial setting to measure, not a claim about typical game length on an unlimited board.

```text
usable = max(0, T - return_reserve)
normal = min(usable, (usable + (H - 1) * I) / H)
hard   = min(usable, 3 * normal, optional_turn_cap)
```

The increment influences allocation but never increases the current hard limit. An absent per-turn cap adds no limit. `hard` bounds delivery of the complete response; the controller's internal return deadline is earlier by its measured return reserve. A fixed move-time request instead uses its supplied limit as `hard` and spends the available search time inside it.

Measure complete inference-batch latency and response overhead during warmup and ordinary play. Maintain a rolling high percentile for each. Initially reserve the larger of 10 ms and twice the measured response overhead. Do not launch a batch whose measured latency will not fit before the relevant internal deadline. If available time is too small, send the prepared turn.

Give the first placement 60% of the normal search allowance and reserve 40% for the second. The first placement's hard deadline also preserves a second-placement allowance sufficient for one batch when that fits. The second placement receives the actual remaining allowance, including anything saved by the first. One absolute turn deadline bounds both searches.

Root solving, leaf solving and certificate verification share a solver allowance of at most 25% of the normal turn budget. Existing solver node caps remain additional limits. A proved result can end search early. A proved loss uses the existing longest-resistance choice. Solver work never resets the turn clock.

Stop at the normal allowance after a completed search round if the complete turn is stable. Allow search to continue toward `hard` when the leading turn changes or a new proof refutes it. Stability means the same recommended complete turn at the last two completed comparisons. These initial shares and multipliers must be measured in clocked play before calling them good defaults.

### Gumbel search must support interruption correctly

The native search preplans sequential halving from a simulation budget. Current `hxg_stats` favors eligible moves at the greatest visit count. Stopping halfway through a round can therefore compare only the moves that happened to receive their next visit first. This follows from source inspection; it has not been reproduced in a timed experiment here.

Add an explicit native recommendation for interrupted search. Keep a snapshot after every comparison round in which all surviving candidates have received comparable work. Publish the recommendation from that snapshot, applying any newer exact wins or losses before choosing. Before the first completed round, use the prepared legal turn. Preserve unresolved alternatives when the previously selected move becomes proven lost, as required by PR #216.

Estimate the initial simulation budget from the normal allowance and measured throughput. Finish the planned halving search when time permits. Extra time continues balanced work on the remaining finalists without redrawing root noise or discarding the tree. Deadline checks can stop physical work at any point; recommendations use completed comparisons. This avoids depending on an unfinished visit layer for the returned move.

Keep this change in the timed search path first. The existing fixed-simulation evaluator and actor targets need no behavior change for this feature.

## Existing engine practice

Stockfish uses normal and maximum time allocations, with adjustments for search stability. KataGo accounts for clock type and response lag. Six divides one move-time allowance between the two placements. These support the shared-turn deadline and bounded extension above. Their tuned constants do not establish the best constants for Bubble. [Stockfish time management](https://github.com/official-stockfish/Stockfish/blob/master/src/timeman.cpp), [KataGo time controls](https://github.com/lightvector/KataGo/blob/master/cpp/search/timecontrols.cpp), [Six turn search](https://github.com/CixMango/Six/blob/main/engine/src/mcts.cpp).

UCI supplies remaining time, increment, a fixed move-time limit and asynchronous stop. Its clock semantics are useful, but Bubble's coordinates and two-stone turns require its own move encoding. Preserve the existing Six command format, make `go movetime` effective, and make `stop` return the latest complete turn. Full clock information travels through the JSON API below. [UCI specification](https://backscattering.de/chess/uci/).

## HTTTX compatibility

Use the existing HTTTX routes. Its current definitions are pinned at `37d2385f1016abe8b25798238a7d0c4a17a25dda`, which was upstream HEAD when inspected. The existing adapter plays the native handwritten or NNUE engine; add dense Bubble through the shared turn implementation.

| Contract | Standard support | Bubble implementation |
| --- | --- | --- |
| Stateless turn | `POST /stateless/v1-alpha/turn`, `time_limit` in seconds | Enforce the whole request allowance; keep board reconstruction inside it |
| Stateful bot play | WebSocket `/bws/v1-alpha/game`, `move_time_limit` in seconds | Preferred for ordered moves and persistent trees |
| Position evaluation | WebSocket evaluation request and time limit | Analysis with no committed move |
| Request matching | Request identifiers | Echo identifiers and reject obsolete responses |
| Cancellation | WebSocket interrupt | Drop the request and roll back its tentative `previous` moves |
| Remaining clocks and increments | No standard fields | Match-host state and a negotiated Bubble configuration extension |

The schemas define per-request allowances; game clocks and control selection belong to the play layer. [Base capabilities](https://github.com/hex-tic-tac-toe/htttx-bot-api/blob/37d2385f1016abe8b25798238a7d0c4a17a25dda/definitions/bot-api-v1.yaml), [stateless schema](https://github.com/hex-tic-tac-toe/htttx-bot-api/blob/37d2385f1016abe8b25798238a7d0c4a17a25dda/definitions/stateless/stateless-v1-alpha.yaml), [WebSocket schema](https://github.com/hex-tic-tac-toe/htttx-bot-api/blob/37d2385f1016abe8b25798238a7d0c4a17a25dda/definitions/basic_websocket/bws-v1-alpha.yaml).

Expose supported Bubble extensions in `capabilities.json` under `meta.x-bubble`. Advertise the dynamic configuration feature only when implemented. Before a move request, a client that recognizes `clock_config: "v1"` can send:

```json
{
  "type": "config",
  "x-bubble-clock": {
    "version": 1,
    "cross_ms": 180000,
    "circle_ms": 179250,
    "increment_ms": 2000
  }
}
```

Then send a standard request:

```json
{
  "type": "move_request",
  "side": "o",
  "previous": [],
  "request_id": 1,
  "move_time_limit": 32.586
}
```

The example limit follows the initial rule for circle with `H = 20` and a 10 ms reserve. The engine spends toward its normal allowance and can extend inside this hard cap. The configuration is a clock snapshot for the next request, consumed once. Charge elapsed time since receipt of that configuration when deriving remaining time. The host's per-request cap remains authoritative. Clients that do not support the extension still send ordinary move-time limits allocated by the host.

Do not send Bubble configuration fields to other bots unless they advertise support. HTTTX defines the `x-` convention for custom configuration. Live spectator clocks are match-host events, not invented standard HTTTX packets.

Apply `previous` to a tentative session position. Commit that input position after answering; commit the proposed bot move when the client confirms it in a subsequent request. An interrupt restores the position from before the interrupted request, including undoing `previous`. Six `stop` returns a move; HTTTX interrupt discards it. They need different controller actions.

Stateless boards cannot recover the original chronological history. Bubble's dense inputs include the previous turn, so a reconstructed ordering can change the network input. Use ordered WebSocket sessions for comparable dense matches. Retain the stateless reconstruction limit and describe that input limitation instead of treating reconstructed history as equivalent.

### Turn histories

Reuse the single-stone winning turn and open final half-turn behavior already merged in [PR #228](https://github.com/Tomodovodoo/HeXO/pull/228) and [PR #229](https://github.com/Tomodovodoo/HeXO/pull/229). These positions remain valid histories and exports. An open half-turn does not transfer the clock to the opponent. A winning first stone completes the game immediately. Bring the older checks in the stateless adapter into line with these native turn rules when updating that adapter.

## Match API and live clocks

Add these routes to the existing local play service. They expose matches owned by this host, separately from the bot-facing HTTTX socket.

| Route | Purpose |
| --- | --- |
| `POST /matches` | Create a match, select players and control, prepare engines |
| `POST /matches/{id}/start` | Start after both engines are ready |
| `GET /matches/{id}` | Current board, result, clocks and search state |
| `POST /matches/{id}/turn` | Submit a complete human or externally driven turn |
| `POST /matches/{id}/place` | Submit one human placement, retaining the clock through a half-turn |
| `POST /matches/{id}/pause` | Debit elapsed time, cancel search, pause both clocks |
| `POST /matches/{id}/resume` | Resume the same uncompleted turn, with no increment |
| `POST /matches/{id}/resign` | End the match by resignation |
| `GET /matches/{id}/events` | Server-sent events for state, clock and optional analysis |

These are proposed routes, not endpoints already available. Use the existing model registry or checkpoint resolver; do not make a second list of models. A creation request can be:

```json
{
  "players": {
    "cross": {
      "kind": "bubble",
      "checkpoint": "main/132500",
      "search": {"enabled": true, "root_samples": 16},
      "solver": {"enabled": true, "nodes": 32768},
      "net_kernels": "fused"
    },
    "circle": {"kind": "human"}
  },
  "time_control": {"base_ms": 180000, "increment_ms": 2000}
}
```

Resolve mutable checkpoint names to model hashes before preparation. Options are fixed for a rated game. Human play and separate analysis sessions can choose other models, search budgets and solver budgets through the existing settings UI. Solver and search caps always remain inside the time allowance.

Return explicit states `preparing`, `ready`, `playing`, `paused` and `finished`. A snapshot includes `turn_id`, position revision, ordered history, side to move, placements remaining in this turn, remaining clock balances, increment, running side and result. A submission names `turn_id` and the expected position revision, so it cannot apply after another placement or invalidate a resumed request. Increment the position revision after each placement and the turn identifier after turn completion or request invalidation.

A clock event can be:

```json
{
  "type": "clock",
  "match_id": "m1",
  "sequence": 12,
  "turn_id": 7,
  "state": "playing",
  "cross_ms": 151820,
  "circle_ms": 169340,
  "increment_ms": 2000,
  "running": "x"
}
```

Balances are current at event creation. Emit on transitions and four times per second while a clock runs. The UI extrapolates between events using its local monotonic clock and corrects on the next snapshot. Sequence numbers prevent applying older snapshots. Reconnection fetches the current state; display interpolation never decides a timeout. Use browser `EventSource` for this one-way stream. The HTTTX bot connection uses WebSocket separately, with one supported transport dependency declared in the package metadata.

Optional thinking events report completed simulations, placements, solver work, the current complete-turn suggestion and its evaluation. Send proof lines only when verified and describe heuristic lines as suggestions. This stream runs inside the existing play search; analysis must not quietly start an additional search or spend another clock allowance.

Pause is an explicit administrative action recorded in the match. It invalidates the old request, retains the charged clock balance and cancels pending work. A restarted match service loads an unfinished game as paused. It does not invent elapsed time across downtime or resume GPU work automatically.

## Timed match runner

Add a thin `python/timed_match.py` command that calls the same host and engine sessions. Reuse legal game handling, opening selection, color-swapped pairing and result statistics. Keep clocked play outside `dense_eval.play`, whose large simulation batches have different scheduling semantics.

Illustrative command after implementation:

```text
python python/timed_match.py --run runs/dense-v1 --a main/132500 --b main/115000 --tc 180+2 --pairs 50 --concurrency 1 --net-kernels fused --out artifacts/timed-matches/example
```

Resolve these identifiers to `runs/dense-v1/checkpoints/main/132500/ema.pt` and `runs/dense-v1/checkpoints/main/115000/ema.pt` through the existing resolver. Support an external Six command or HTTTX endpoint as an opponent through the existing adapters. Engines with their own clock allocation receive full balances through supported interfaces; engines that accept only move time receive a host-allocated cap. Record who performed allocation so those settings can be reproduced.

Default to one active game. Load both models before clocks start. Batch leaves within the active search without waiting for unrelated games to fill a batch. Parallel games sharing a GPU pay their actual queueing time and define a different execution setting; they are not a shortcut to the same clocked result. Record device, concurrency and kernel settings with results. Fused kernels remain available; graph capture is warmed before clocks when supported.

Count timeouts, illegal turns and engine crashes explicitly. External engines that ignore their allowance still lose when the host deadline expires. A runner fault that prevents either engine receiving its turn is an interrupted game, not a fabricated engine loss. Administrative placement caps remain censored results, rather than timeout losses or real draws.

Keep timed results outside the existing fixed-simulation league. Each time control and execution setting has its own match records and rating input. Reuse the existing posterior calculation on those selected records; never feed them silently into `runs/dense-v1` evaluations. Different controls can produce different rankings.

Write the match specification, one append-only JSON event file per game and a final summary. Each completed turn records side, coordinates, charged time, clock before and after, search and solver work, stop reason and model identity. The final record includes result, finish reason, opening, source revision, native-library identity, engine options and hardware settings. This is enough to replay clock accounting and locate an overrun without a new telemetry service.

## Notation export

Use the existing [notation reader and writer](../python/notation.py). The supplied standard names Absolute and Fischer and defines an integer base with an optional integer increment. It also defines timeout as `endreason[time]`. Export metadata such as:

```text
version[1]name[Bubble timed match]platform[Bubble]utcdatetime[2026-10-01 12:00:00]playercross[Bubble 132500]playercircle[Bubble 115000]timecontrol[180+2]endreason[time]winner[circle];
```

Use `utcdatetime` from the metadata table and the plain numeric control grammar. The README example uses a different date key and prefixes its control with `Fischer`; do not copy those inconsistencies into exports. [Notation specification](https://github.com/hex-tic-tac-toe/hexagonal-tic-tac-toe-notation/blob/15bb7877ae020d661497e332adf0810d00d24e3e/README.md).

The specification does not explicitly state the control's units. Bubble documents seconds for the text form and milliseconds for JSON. Export integral-second controls through this grammar. JSON remains authoritative for finer precision, individual turn times, remaining clocks and finish reasons that the notation does not define. Never encode an illegal-move or crash result as a board win.

Keep the final half-turn behavior already implemented by PRs #228 and #229, as described in [notation-api.md](notation-api.md). Winning first stones and unfinished final turns export with their actual one coordinate. Preserve every placement. The JSON record adds the clock trace and whether an unfinished turn is still running, paused or ended on time.

## Implementation and verification

Implement the shared deadline and complete-turn controller first, including safe native recommendations on interruption. Then make `/bot` and Six `go movetime` use it and make Six `stop` responsive. Next add the match host and runner. HTTTX WebSocket sessions, the clock configuration extension, live match events and notation export can then use that same working host.

Verification uses the existing test modules for play, search, Six and notation, with a controlled clock and a deliberately slow evaluator. Check whole-turn charging and single increments, delivery or forfeiture at the host deadline, and interrupted search recommendations independent of which move received the first visit of a partial round. Exercise actual adapter messages to verify cancellation, confirmed history and request matching. Replay the recorded clock events and recover the reported result and final balances exactly.

No tests run during the current compute hold. When authorized, ordinary CPU checks use BelowNormal priority and `OMP_NUM_THREADS=2`; skip `test_engine`, `test_klent` and `test_gpu_nnue`. Use the authorized prebuilt native libraries where compatible. Any new native ABI must have a matching library before runtime verification; an old DLL cannot validate new interruption behavior.

After permission to use compute again, start with a short clocked match at one zero-increment control and one Fischer control. Measure late replies, unused final time and time spent in inference, solving and return work. Compare allocation settings through playing results only after clock accounting works. Do not claim a stronger engine from this design alone.
