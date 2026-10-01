# Timed matches and bot API

Bubble supports Absolute and Fischer clocks. One allowance covers solving, both stones and the response. A winning first stone ends the game. An unfinished half-turn retains the same clock and earns no increment yet.

## Start the API

Install the API extra alongside the learning dependencies. This feature requires the existing matching native libraries and adds no native ABI.

```sh
python -m pip install -e ".[learning,api]"
python python/timed_api.py --run runs/dense-v1 --checkpoint champion --device cuda --net-kernels fused --port 8790
```

`--model path/to/ema.pt` selects a file. `--device cpu` avoids GPU inference. Without a model or run, the handwritten native engine plays. Engines load and warm before game clocks start. The service binds to localhost. `GET /models` lists the existing play picker selections. Records go to `artifacts/timed-matches`, selectable with `--out`; run data is read only.

For shared prebuilt libraries, `HEXO_NATIVE_DIR` selects the C++ library directory and `HEXO_TACTICAL_PACKAGE` selects the complete tactical package. The solver still verifies its library and source hashes. An older search ABI cannot validate current search code.

## Pass the clock to Bubble

`POST /bot` accepts ordered history and clock balances, in milliseconds:

```json
{
  "history": [[0, 0]],
  "clock": {"cross_ms": 180000, "circle_ms": 179250, "increment_ms": 2000},
  "time_limit_ms": 30000
}
```

The optional per-turn cap restricts the remaining clock further. The response includes complete `moves`, model identity, work counters, proof status and the selected allowance. Omit `clock` for a fixed move-time request. `/analyze` runs the same search without committing a move to any match.

Chess UCI sends balances and increments with `go wtime ... btime ... winc ... binc ...`. The engine subtracts elapsed time locally while thinking. The host owns the actual clocks and timeout result. Bubble follows those semantics. [UCI specification](https://backscattering.de/chess/uci/).

The Six server honors `go movetime`, reads commands during search and returns a complete legal turn on `stop`. Full clocks use `xtime`, `otime`, `xinc`, `oinc`, or the UCI clock names with white mapped to cross. `isready` remains responsive. Coordinates retain the Six format.

```sh
python python/six_engine.py serve --run runs/dense-v1 --device cuda --net-kernels fused
```

```text
position radius 8 moves 0 0
go xtime 180000 otime 179250 xinc 2000 oinc 2000
```

## HTTTX

The service exposes `GET /capabilities.json`, `POST /stateless/v1-alpha/turn` and WebSocket `/bws/v1-alpha/game`. Standard HTTTX allowances are seconds, named `time_limit` for HTTP and `move_time_limit` for WebSocket. The standard has no remaining-clock or increment fields. [HTTTX definitions](https://github.com/hex-tic-tac-toe/htttx-bot-api/tree/37d2385f1016abe8b25798238a7d0c4a17a25dda/definitions).

Capabilities advertise `meta.x-bubble.clock_config: "v1"`. A client that recognizes it sends this configuration before a standard move request:

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

```json
{"type":"move_request","side":"o","previous":[],"request_id":1,"move_time_limit":30}
```

Bubble consumes the snapshot once, deducts time since receiving it and continues counting locally. Send a new snapshot before the next request. Send Bubble configuration only to bots that advertise it. These fields use the standard's `x-` convention for custom configuration.

WebSocket sessions retain ordered history. `previous` supplies confirmed moves, including the bot's earlier suggestion. The bot waits for that confirmation before committing its proposed turn. `interrupt` discards the request and rolls back its `previous` moves. Request IDs increase within a connection; abandoned replies cannot change the position. `eval_request` returns an evaluation without a committed turn.

To set an opening, configure `x-bubble-history` with ordered coordinates, then send `setup`. Capabilities advertise `history_config: "v1"`. Standard setup uses the origin board. Stateless reconstruction can change the previous-turn features used by the dense net, so use ordered sessions for comparable dense games.

First-stone wins and final half-turns retain the behavior merged in [#228](https://github.com/Tomodovodoo/HeXO/pull/228) and [#229](https://github.com/Tomodovodoo/HeXO/pull/229). The HTTP adapter also accepts the origin turn and the remainder of a half-turn, returning the actual legal placements.

## Clocked matches and live time

`POST /matches` accepts:

```json
{
  "players": {
    "cross": {
      "kind": "bubble",
      "checkpoint": "champion",
      "search": {"enabled": true, "root_samples": 16},
      "solver": {"enabled": true, "nodes": 32768, "leaf": false},
      "net_kernels": "fused"
    },
    "circle": {"kind": "human"}
  },
  "time_control": {"base_ms": 180000, "increment_ms": 2000}
}
```

Players can be `human`, `bubble`, `native`, `six` with `command`, or `htttx` with `url`. The HTTP opponent requires a stateless capability and receives a host-allocated limit. Bubble gets full balances. `champion` and `newest` resolve to a model hash before play. `search.max_simulations` optionally caps timed search; otherwise it follows measured throughput. Solver node caps remain inside the clock. `solver.leaf` shares the solver allowance with leaf proofs and certificate verification.

Creation returns HTTP 202, a `match_id` and `preparing` state. Poll until `ready`, then start. The initial history defaults to the origin. An optional `history` supplies an opening; `turn_cap_ms` adds a maximum time per turn.

| Route | Action |
| --- | --- |
| `POST /matches/{id}/start` | Start engines and clocks |
| `GET /matches/{id}` | Position, placements remaining, clocks, result and thinking |
| `POST /matches/{id}/place` | Human placement with `q`, `r`, `turn_id`, `revision` |
| `POST /matches/{id}/turn` | Human turn with `pieces`, `turn_id`, `revision` |
| `POST /matches/{id}/pause` | Charge elapsed time and cancel search |
| `POST /matches/{id}/resume` | Resume without increment |
| `POST /matches/{id}/resign` | Resign the supplied `side`, `x` or `o` |
| `GET /matches/{id}/events` | Server-sent events, four updates per second |
| `GET /matches/{id}/notation` | HTTTX export |

The host charges queueing and transport until receipt of a complete legal response. A response at or after its deadline loses on time, including a winning move. Apply increment once after turn completion. Opening placements receive no artificial increments. Pause retains the charged balance and any half-turn, and invalidates the old request.

Events include `cross_ms`, `circle_ms`, `increment_ms`, `running`, request identifiers, state and optional thinking. A UI can interpolate between events with its own monotonic clock. The host decides timeout. On service restart, saved unfinished matches load paused. An explicit resume prepares their engines again.

## Timed comparisons and records

```sh
python python/timed_match.py --run runs/dense-v1 --a main/132500 --b main/115000 --tc 180+2 --pairs 50 --device cuda --net-kernels fused --out artifacts/timed-matches/132500-v-115000
```

`180` is Absolute and `180+2` is Fischer, in seconds. `--opening file.htttx` supplies a position. Each pair swaps colors. `--b-command` selects a Six opponent; `--b-url` selects an HTTP HTTTX opponent. One game runs at a time so unrelated inference queues do not consume the competitors' clocks.

Each match saves its specification, append-only events, final JSON and notation. Identities include the model, source revision, settings and native-library hashes. Timeout, illegal reply and engine crash are explicit outcomes. Placement caps are censored. Results stay outside the fixed-simulation league.

Notation records `version`, `utcdatetime`, players, control, winner and `endreason[time]` for timeout. It preserves a one-stone final turn. Its control grammar does not explicitly name units; Bubble uses seconds in text and milliseconds in JSON. Detailed clocks and finish reasons absent from notation remain in JSON. [Notation specification](https://github.com/hex-tic-tac-toe/hexagonal-tic-tac-toe-notation/blob/15bb7877ae020d661497e332adf0810d00d24e3e/README.md).

## Allocation and interruption

The initial normal allowance divides remaining time and expected increments over 20 own turns. Its hard cap is at most three normal allowances and never spends future increment. Reserve 10 ms for returning the move. Fixed move-time requests use their supplied allowance inside that reserve. These are initial settings, not measured optimal values.

Give the first stone 60% of normal time and preserve time for the second. Root and optional leaf proofs share at most 25% of normal time. Keep a complete legal candidate before solving or inference. A persistent worker owns the model; the controller can return that candidate while a non-cancellable call finishes. Generation IDs discard late results.

Timed Gumbel search saves a completed comparison at the existing final halving boundary. Interruption returns that recommendation after filtering newly refuted moves. Before that boundary, retain the legal fallback. Fixed-simulation training is unchanged. If the last round changes the recommendation, search can compare finalists longer while preserving second-stone time. It reuses visits and caches; the existing `hxg_begin` redraws noise for that extension. No new ABI is needed.

Measure batch latency during warmup and search, and avoid starting a batch that the measured latency says will not fit. The controller and host enforce the response deadline. This is not a hard real-time operating-system guarantee.

CPU verification covers half-turn accounting, single increments, pause/resume, timeout precedence, restoration, WebSocket rollback and clocks, responsive Six commands, interrupted comparisons and a real dense worker using a tiny CPU checkpoint. Training and production evaluation need no restart. Start the new API or restart an existing Six/play adapter to load the new Python code.
