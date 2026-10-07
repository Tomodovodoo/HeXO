# HeXO search scheduler

Updated 2026-10-07. Merged source baseline: main `8c173186`, including PR433,
PR439, PR441, PR442 and PR444. PR438 integrates that baseline privately. Draft work
is identified below; its presence on this branch does not make it deployed.
This replaces the October 3 proposal.

The objective is useful search and faster model improvement per machine-hour.
Keep useful GPU and CPU work ready, retain evidence across turns, discover
alternatives within self-play, and let proofs stop unnecessary neural work.
GPU occupancy, simulation counts, proof counts, and Elo each answer only part
of that question.

## Implementation status

| Area | Current implementation | Delivery |
| --- | --- | --- |
| Shared game graph | Context nodes, rule-position outcomes, shared child evidence, edge-local visits, re-rooting | PR328 and subsequent graph changes are on main |
| Native feeding | Bulk selection, encoding, context deduplication, prediction cache, installation and backups | PR332, PR340 |
| Native owner | Independent same-game views, leased work, root completion records, bounded continuation discovery | PR354 |
| CPU proving | In-process native worker instances, immutable jobs, arbitrary retained-position proof installation, scoped facts, retries and cancellation | PR350, PR357, PR358, PR388 |
| Shared inference service | Native producers supply one model-keyed queue; Python launches and collects packed immutable batches | PR367, PR377 |
| Continuous games and actors | Slot release/refill, pause fences, native actor worker, bulk legal root records | PR368, PR370, PR371, PR373, PR380 |
| Lean graph work | Sparse graph update indices, complete legal lists with lazy mutable edge state, grouped untouched-action selection, stored-continuation index | PR379, PR382, PR384, PR397 |
| Graph retention | Bounded dormant archive and forward-game removal of colour-conflicting variations | PR389, PR395 |
| Inference capacity and packing | Larger small-canvas captures, measured crop planner, reduced tail padding | PR338, PR385, PR393 |
| Root batching | Optional round barriers retain sequential halving while allowing more work within a round | PR396 |
| Solver continuation | Bounded interrupted best-first proof frontiers retained between slices | PR398 |
| Browser | Native owner in WASM, bounded GPU captures/readback, external solver workers and endpoint handoff | PR400, PR401, PR402, PR403 |
| CPU-to-GPU paths | Actual quiet/unfinished forcing paths admitted as connecting prefixes and endpoint candidates | PR403 |
| Queue-pressure admission | Active-candidate lookup and existing-view-first gathering before allocating extra views | PR404 |
| Rectangular inference | Full-board rectangular encoding, packing, fused inference and browser paths with existing weights | PR413 |
| Retarget and actor records | Avoid redundant eviction scans; retain the raw root prediction; incremental prefix keys | PR425, PR428, PR429 |
| Proof supply | Wake one worker, close unsupported quiet sides, retry only after fresh work and count supply exclusions | PR415, PR427; PR440 sets opt-in proof defaults |
| Actor launcher | Separate inference pumping and shard writes; bounded event drain and no-progress backoff | PR433 merged, including the queued-shard naming fix |
| Actor graph allowance | Retain up to 1024 nodes per native actor game by default | PR444 merged; this changes the configured allowance, not proof of a loaded production binary |
| Clocked Play | Owned progress at completed comparisons, matching second-root replacement, final exact precedence and stop-reason delivery | This PR438; current IPC comparison returned two searched stones in all 12 HeXO turns; strict deadline checks and current Wasm verification remain open |
| Capture accounting | Attribute capture reservations to their model instead of a constructor-time device baseline | PR439 merged after actual two-model CUDA validation |
| Shared proof workers | Lend globally bounded workers across producer loops while graph owners install results | Draft PR443; owner benchmarks in progress and a concurrent cooling-retry admission finding remains open |

These are source deliveries. Native actor/continuous-proof modes remain opt-in.
They do not establish that production has loaded these binaries, that training
is using every mechanism, or that learning has improved. The fixed-work
evaluator remains a separate reproducible configuration.

## Evidence and measured limits

All hardware figures below are observations under their recorded conditions,
not forecasts. Local receipts are retained in
`artifacts/live/gpu-schedule-20261003` and
`artifacts/live/pipeline-profile-20261003`. These directories are ignored
runtime evidence, not files supplied by a public clone. Each receipt records
its source, binary, checkpoint, configuration and workload identities.

| Experiment | Observation | What it establishes |
| --- | --- | --- |
| Initial native feeder, October 3 | Median actual-GPU elapsed 1.4374 to 1.4132 seconds for 6,469 requests, about 1.7% | Moving orchestration alone did little for total search time. Diagnostic installation was 471-501 ms and selection 253-261 ms. |
| Batch-32 backend, shared load | Bubble fused CUDA graphs 8,897 rows/s at 24x24; Bubble TensorRT 8,297; Six TensorRT 6,838 at 25x25 | Bubble's measured forward backend was faster in this experiment. This is not full-search throughput or equal-work engine strength. |
| Shared native service, October 4 | At 128 and 384 concurrent book starts, proof-enabled rows/s about 9,970 and 11,893; roughly 1.72x and 1.90x the compared coordinator | Many independent games can supply large batches. Conditions and clocks differ from a single-game benchmark. |
| Endpoint export, October 6 | Midgame checked proofs 3/3 to 10/11 and completed-view depth 5/5 to 7/7, while rows/s fell slightly | Useful proof/depth work can improve while neural throughput declines. This small study establishes no general strength gain. |
| Round versus visit-layer barriers, endpoints enabled | Four trials per condition: quiet 3,142 to 4,493 rows/s; four midgames 4,008 to 4,944 | Higher unique NN admission rate under the same one-second clock, about 43% and 23%. Depth and credits were separately recorded. |
| Same-clock Six protocol play | Four sustained starts: HeXO proof-on about 4.0-4.7k versus Six 3.7-4.4k neural rows/s, two repeats | A local end-to-end comparison is competitive. Six's PUCT protocol player is not its Gumbel self-play pipeline; CPU use differed and no learning/strength conclusion follows. |
| Current IPC Play versus Six, October 7 | Three two-stone starts, two alternating repeats, HeXO solver-off/on: all 24 engine replies legal and complete. HeXO terminal neural rows per reply-plus-drain second are 3.52-4.50k solver-off and 3.05-3.87k solver-on; Six protocol neural rows per reply second are 3.23-4.84k. | Actual current caller and pinned binaries, not the older manual-pump harness. Terminal work can include work after reply. Only two repeats per condition; new canvas captures, external CPU activity and different search/proof algorithms limit the comparison. No general speedup or strength result. |
| PR404 balanced allocation study | 48 one-second trials across baseline, lookup control and pressure admission. Midgame owner-step wall per NN row falls about 16%; root comparison credits rise. Overall rows/s is approximately flat. | Admission improves a measured host/allocation cost, not a massive speedup. Neural-only continuation depths fell; proof-enabled depth/proof delivery was comparable. |
| Actor pipeline, October 6 | Four trials per arm, full-length frozen-main/200000 games, no proof workers: combined experimental changes raise placements/s from 180.8 to 204.2 at 128 slots and 160.4 to 187.0 at 64; unique NN rows/s 4,048 to 4,540 and 3,578 to 4,158 | Real data-generation gain of 13.0% and 16.5%. Includes pending PR433 and graph limit 1024; published rows were replayed through the learner. Not a proof-enabled or learning-strength comparison. |
| PR404 complete games | 8 neural-only and 8 concurrent-proof uncapped games, all terminal; 1,780 saved rows, 96 independently checked CPU certificates, no missing raw predictions | Legal play, proof/data/lifetime behavior and complete drainage. Not a strength or learner comparison. |
| Proof worker wakeups, October 6 | 12 proof workers, queue 48, whole machine, four A/B/B/A trials per arm: book-32 solver-on NN rows/s 7,025 to 10,283, proof queries 10.2k to 18.5k, checked certificates 0 to 8; saved-selfplay-8 3,909 to 4,205. Unchanged within noise at 4 workers | Waking every idle worker for each job made them queue on the proof mutex ahead of the graph owner. More workers now add coverage instead of starving the GPU feed. |
| Proof supply and worker split, October 6 | 12 proof workers, 4 s clocks, four A/B/B/A trials per arm (#427): worker idle on book-32 55% to 12%, fresh solver nodes +7 to +48% across cohorts, book-32 checked certificates 11.5 to 42, NN rows/s within 2% on the multi-game cohorts and -4 to -6% on quiet-handoff and saved-selfplay-8. Queue 8 per worker against 4: fresh nodes +9 to +24%, midgame certificates 100 to 147, NN rows/s -1 to -2%. 16 workers against 12 cost quiet-handoff 15% of NN rows/s | Most quiet dispatches were no-ops that never closed their side. Defaults: 12 proof workers per producer pool (one pool per model), queue 8 per worker |

An October 6 reverse-order queue study adds 96 one-second trials. With endpoints
and proofs enabled, producer outstanding depth 2 to 4 at chunk 64 raises median
NN rows/s from 4,156 to 4,440 on the quiet start, 4,313 to 4,782 across four
midgames, and 4,793 to 5,896 across 32 book starts. The 128-row chunk/four-deep
configuration reaches 6,890 on the book cohort but completes depth-one views,
versus depth-two at chunk 64. Throughput and allocation both matter.

A separate 96-trial CPU-capacity/endpoint study compares two proof workers with
queued-plus-running capacity 2, 8 and 32. In four proof-enabled midgame trials
per condition, capacity 8 to 32 raises median worker service-wall share from
69% to 91%, checked certificates 40 to 53, and NN rows/s 4,804 to 4,964. This
is wall service, not CPU occupancy. The quiet start returns no proofs and the
larger backlog spends more CPU effort while reducing NN rate/root credits.
A fixed largest backlog is therefore not the global allocation answer.

In 32 further five-second trials, endpoint export at the same queue settings
raises midgame checked certificates from 66 to 105 and completed-view depths
from 7-8 to 12-14. Quiet-position NN throughput rises from 3,725 to 4,643 rows/s,
while its certificates are 24 versus 22. Ten distinct nested history chains
join a CPU endpoint, a completed neural view, a deeper CPU query, and another
completed neural view. These are partial rule-position matches, not causal
dispatch timestamps or independent neural-context counts.

Relevant receipts include `native-feed-gpu-comparison.json`,
`production-native-inference-service-gpu-q64-l02.json`,
`solver-neural-frontier-summary-20261006.json`,
`endpoint-batch-supply-summary-20261006.json`,
`native-admission-isolation-summary-20261006.json`, and
`native-admission-fullgames-20261006.json`,
`endpoint-queue-*-20261006.json`, and
`endpoint-cpu-backlog-summary-20261006.json`.

The latest two screenshots show the earlier three-turn animation, captured on
October 5 at `cf3bf316`, not this source baseline. Its driver submitted and
collected one GPU batch at a time even though the broker allowed two. About
1.44 seconds of that capture's no-kernel spans overlap queued neural rows.
The spans establish an opportunity; they do not isolate its cause. A new trace
must exercise the actual two-flight launcher and distinguish queue/packing,
host preparation, installation, transfer and dependency waits. Open nodes that
are ready must not wait merely because view admission is busy elsewhere.

The best measured actor configuration above sampled about 95% device activity
at 128 slots, with machine CPU about 18.5% and the actor process averaging
1.62 logical cores. These are solver-off figures. Four-second solver-on cohort
trials separately sampled machine CPU about 70-77%, or 34% when seven of eight
roots were proven. Worker service-wall share is not CPU instruction occupancy;
`nvidia-smi dmon` activity is not achieved SM occupancy. The October 7 current
single-game comparison separately sampled HeXO device busy-time at 60-63%
solver-off and 53-56% solver-on, against Six at 57-72%. These ranges are means
of two trial means for each fixture. HeXO process CPU averaged 0.56-0.82 logical
cores without proofs and 1.18-2.54 with proofs; Six averaged 0.91-1.01. NVML's
averaging period can cross the reply boundary. External CPU services and 33
unowned monitor processes were present; their interference was not isolated.
These observations do not measure achieved SM occupancy or establish that all
remaining device time can be filled with useful search. Historical batch-32 backend
rates, single-game rates and sustained actor rates have different shapes and
supply conditions and cannot be substituted for each other.

A general 5x improvement has not been established. Neither an assumed
100 microseconds/row host cost nor four halving rounds supplies a speed forecast.
Dependency waves, collisions, shape fragmentation, graph updates and shared
GPU availability have to be measured.

## Ownership and task graph

There are three different graphs: positions and their continuations, CPU/GPU
task dependencies, and captured CUDA kernels. Position sharing does not remove
task dependencies or fuse CUDA launches.

```text
self-play search views
  -> native selection / context dedup / encoding
  -> model-and-shape ready queue
  -> immutable GPU batches
  -> native result installation / backup / candidate discovery
  -> self-play search views

native proof frontier -> CPU worker slices
  -> verified facts -> graph outcomes / propagation / later CPU premises
  -> checked unfinished paths -> neural continuation candidates
```

Each game graph has one mutation owner at a time. Independent games may use
parallel native workers; a producer owns its pools until service close.
Selection and installation phases finish before control/proof delivery mutates
the same graph. CPU solver workers own immutable jobs and private solver
instances. They return answers; they do not modify graph nodes.

Python configures, loads models, launches captured inference, collects fenced
outputs and writes actor data. Per-leaf selection, encoding, keys, deduplication,
proof dispatch and graph installation belong in native code. The current native
proof loop calls the Rust library in-process. The old proposal's JSON pipes to
Python solver subprocesses are not this desktop architecture.

Browser inference uses the same native owner compiled to WASM. External solver
workers receive immutable requests and return typed outcomes plus bounded paths.
They cannot dereference a native graph pointer. Scoped known facts cross as
explicit position/premise records. Browser cancellation and delivery must respect
the external worker's own slice/acknowledgement behavior.

## Identities and evidence

| Identity | Contents | What may be shared |
| --- | --- | --- |
| Rule position | Coloured stone sets, mover, remaining placements and rules | Verified outcomes, legal transitions, certified distances/premises |
| Neural context | Rule position plus encoder-required recent history, encoder/model identity | Prediction and encoded-input caches |
| Search view | Root, lease/generation, root sample/halving state, deadline and credits | Position evidence, while comparison accounting remains local |

Two placement orders can produce the same rule position while giving different
network inputs. Deduplicate proofs by rule identity and predictions by complete
neural context. Model changes cannot reuse old predictions as current weights.
A translation does not justify narrowing wide coordinates without a checked
encoding contract.

A shared child's improved Q can improve every parent reading it. This does not
create extra evaluations or independent root-comparison visits. Keep issued,
completed, cancelled and direct comparison credits distinct from shared evidence.
Existing edge-local weighting remains the source contract; this document does
not propose changing it to child-total visit weighting.

Requests retain the graph/path ownership needed until completion or fenced
abandonment. Do not implement an 8-bit arena generation that can wrap while
old references exist. The current retained objects avoid that proposed ABA
problem. Any future arena implementation needs an explicit lifetime contract.

## Repeatable CPU and GPU cooperation

The handoff is a cycle, not a one-time CPU prepass:

```text
CPU forcing search
  -> checked open position
  -> GPU evaluation and quiet continuation search
  -> newly expanded/changed positions offered to the CPU frontier
  -> another CPU slice
  -> more open positions or a verified proof
  -> GPU continuation or exact propagation
```

PR403 exports actual visited quiet/unfinished paths, up to eight per query and
64 new placements per path. The graph owner independently replays legality,
placement phase and nonterminal state, and checks containment of the current
coloured stones. Connecting prefixes remain available, so a returned line does
not replace opponent reply coverage. Explicit solver paths can extend up to 128
placements from the focus; ordinary automatic discovery has its separate depth
limit. Those are current bounded controls, not an inherent one-handoff rule.

`Loop::observe` in `src/gumbel_proof.cpp` is called after neural installation.
It offers the evaluated leaf using view relevance, path share, forcing material
and value movement. Active expanded view roots are also offered. Therefore
neural continuations under a CPU-exported path can become later CPU jobs.
Unfinished CPU results can export another set of endpoints. No fixed number of
CPU/GPU alternations is imposed, but every stage remains subject to admission,
retention, solver scope and the common clock.

A candidate is not guaranteed service. Admission chooses among competing roots,
continuations and proof tasks. Existing receipts can join endpoint histories to
neural-view records and later proof histories. Without dispatch/installation
causal timestamps, those joins are observations, not proof that one particular
GPU result caused a later certificate.

Forcing search is selective. A quiet endpoint is unresolved, and a forcing
negative is UNKNOWN. Neural estimates may rank quiet moves; neither an estimate
nor an unsupported proof-number bound establishes a game loss. Only checked
certificates or complete legal proof propagation may mark exact outcomes.

A proven win needs one winning continuation for the winner. A proven loss needs
all legal replies covered, with perspective changing when the mover changes,
not after each stone. Exact outcomes dominate late neural results. Release every
reservation exactly once even when its result becomes obsolete.

## CPU frontier and continuation

Current task priority uses root/view impact, value movement, measured recent
query cost and aging, with deliberate exploration. It excludes exact/dormant
nodes and nodes whose rule peers have neural work pending. It alternates mover
and defender work subject to scope status. A side closes for the same premises
when its search is disproved. Without stamps the defender side is closed from
the start when the solver cannot begin there (the mover completes now, or the
defender's attacker has no two-stone completion); premise changes reopen only
the mover. A task inside its retry delay ranks behind every other eligible task,
and only a worker idle at the refill continues it with the doubled slice, so
retries never queue ahead of fresh positions. A full frontier drops exact
entries and entries closed under the current facts before open ones. The
desktop pool uses native worker instances, continuation affinity and work
stealing rather than leaving a worker idle because its preferred continuation
is unavailable. `hxp_supply_stats` counts why each refill stopped and charges
idle worker time to that reason.

Queries use time slices under the root clock. Retries can increase the slice;
new relevant premises or material value change can reopen work. Fresh nodes,
historical/cache work, wall duration, queue waiting, continuation state and
termination reason are different fields. Cache hits must not be charged as
new solver expansions. A failed attempt is scheduling evidence, not a game
verdict.

Verified graph facts are snapshotted into each job's immutable premises. Scope
refresh selects facts compatible with the queried coloured stones; unsupported
or irrelevant facts cannot close it. Generalized stamps have their own validity
requirements. Proof dependencies must survive delivery and independent checking.

PR337 cancellation covers search/certificate work. Measurements recorded about
0.39 ms native, 0.89 ms isolated and 6.91 ms maximum in that study. This is not a
universal deadline bound. PR398 retains bounded interrupted best-first proof
frontiers and worker-resident tables. Recursive level-one frames and every kernel
memo are not fully resumable. "Resume at zero cost" is not a current guarantee.

Do not add synchronous per-leaf pool queries to fill this role. They formerly
added 0.8-5.8 seconds in sampled turns with little proof evidence. Keep existing
cheap immediate classification. Deeper checks earn CPU slices through utility
and cost; whether another inline tactical check pays is a measurement question.

Worker sharing is a separate unfinished step. Each current proof loop owns its
workers; when that loop's games settle, another producer cannot use the idle
capacity. A shared service needs fair admission and dispatch, client-specific
cancellation/drain, immutable job and generation identities, and stable callback,
library and worker-thread lifetimes. Only originating owners may install results.
Equal queue shares alone do not guarantee dispatch fairness.

Known-premise and stamp queries currently clear the worker's raw resident table
and frontier. This isolates proof scope but can destroy retained raw work. Sharing
a queue or remembering a preferred worker does not resolve that. Conditional
sessions require explicit isolation or a measured, disclosed reset policy;
continuation counters must distinguish preserved work from fresh rebuilding.

## Useful breadth, depth and batching

Keep the full legal action set. The implemented compact store separates legal
records from mutable edge state, allocating the latter when work/proof/child
state first requires it. Sparse selection groups untouched actions whose
completed Q is common, while treating visited, pending, transposed and proven
edges individually. Root sampling and exported targets still cover all legal
moves.

This removes repeated work without permanently capping candidates. Adaptive K
with a rest bucket remains an experiment, not the current store. Any future cap
must preserve omitted logits and legal reachability, allow widening/reconnection,
and never infer a loss from exhausting a truncated set. The old 450-byte/node
and two-million-nodes-in-one-GB claims omitted legal storage and are withdrawn.
Count legal records, mutable states, nodes, history, indices, caches and pinned
work when sizing memory.

Line views explore other roots into the same graph. Useful candidates arise
inside self-play from sampling, exploration, forcing analysis, prior/search
disagreement and discoveries below existing roots. External opponent games are
not a required training input. Low prior alone cannot permanently remove a
move. Six top-16 policy agreement does not establish complete coverage or explain
its wins.

The optional round barrier waits at halving decisions, not every visit layer.
Sequential halving still distinguishes issued work, completed work and comparison
credits. A candidate's later descent may need its first policy result; pending
collisions and cache/proof hits create additional waves. Four rounds are not four
launches, and even 8-12 launches is a hypothesis for a particular 128-simulation
configuration, not a universal bound.

Allocation should adapt between continued depth, root alternatives, refutations
and useful extra views. A higher rows/s configuration can support more useful
work under an unchanged clock. Rejecting it only because it explores less depth
at the same simulation count uses the wrong resource comparison. Conversely,
rows added solely to fill hardware do not earn policy-target credit.

### Clocked candidate delivery

A final completion can arrive after the caller's deadline even when a useful
candidate already exists. PR438 exposes one owned latest-progress frame per game
after completed primary-root comparisons. Observing it does not acknowledge the
command, advance its epoch or permit retargeting. Actor progress publication is
off by default.

Play validates producer, game, model, token, history and context. A new second-root
observation replaces its provisional row and cumulative counts. It adds no new
comparison credits. Final completion supersedes progress, and exact witnesses
remain authoritative. Reading deeper evidence about the same position does not
fabricate current-root sampling work. The first stone still needs final completion
before its second root can be requested. If no usable prediction arrives in time,
this mailbox cannot manufacture a searched turn.

Native ownership/backpressure, CPU Play, browser behavior and exact-head review
passed on implementation `6195d27`. Actual one-second GPU/controller delivery
failed in the eighteenth saved trial. The forcing-position second root had zero
completed comparisons, despite 80 issued requests; the fenced turn took 1.898
seconds against a one-second allowance. A new 32x40 capture and a competing CI
CUDA context were present, but neither is established as the cause. The caller
now keeps that attempt in the clock record with zero fresh root comparisons. A
matching final graph choice remains useful even without fresh comparison credits.
Each stone distinguishes current-root comparisons, accumulated position-edge
visits, an uncredited estimate and verified proof. Edge visits are neither unique
neural evaluations nor an age measurement. An uncredited estimate can already
include proof exclusions and the improved-policy transform, so it is not labelled
as a raw network prediction. A matching completed progress candidate remains
preferred to a later uncredited final estimate. Exact evidence remains usable
without neural comparison credits. The changed existing caller behavior passed
on private `0b9ce02` using immutable mocked frames, without model or CUDA work.
It checks inherited and uncredited choices, zero fresh credits and exact
precedence. It does not verify actual GPU timing. This reporting correction does not resolve
the missed work. Earlier one-searched-stone
timing failures are also retained. Measure caller return separately from
post-return neural/proof drainage and trace capture, installation and publication
boundaries before attributing the delay.

On private `b4d05171`, nine new direct causal traces and the actual IPC/Six
comparison were saved on October 7. The first unprewarmed direct turn spent
1,125 ms in the initial capture call, including its warmup, and returned no
searched stones. This does not establish a missing Play warmup: the actual
TimedEngine worker evaluates the center position before sending `ready`, outside
the move clock. All 12 actual IPC HeXO replies contained two aligned,
positive-credit position records. Eight reply times nevertheless exceeded
exactly 1,000 ms, with a maximum of 1,011.36 ms; Six also exceeded nominal
movetime in its protocol replies. Functional delivery is therefore verified on
these fixtures, while strict deadline success is not. The original failed
receipt remains unchanged; no cause for its competing-context delay is inferred
from the new cold-capture observation.

The published `evaluated` counter is launched neural rows and can include
in-flight work. Final service counters follow mandatory drainage. Neither
counter says how many predictions affected the move delivered by the controller.
The current comparison keeps reply snapshots, terminal totals and their wall
boundaries separate. Its receipts are `current-caller-six-turns-20261007.json`,
`current-caller-six-summary-20261007.json` and
`clocked-causal-delivery-runtime-20261007.json`.

### Queueing and transfers

Prefer a ready surplus of useful, independent work to starvation. Keep queued
work cheap, deduplicated and revocable; do not equate a larger ready backlog with
a larger immutable GPU backlog. Proofs, changed roots and deadlines may invalidate
work before launch. Once submitted, preserve its buffers until the GPU is fenced.

The current broker accepts producer chunk sizes 1-128 and outstanding chunks
1-4. Each pool's ready watermark is twice its chunk size. The Python launcher
keeps two GPU forwards in flight. These are separate limits. CPU proof capacity
includes queued and live jobs and is separate again. Raising one setting does not
automatically raise the others.

Group by model and compatible canvas after context deduplication. Launch on
sufficient real rows, request age, deadline or blocked useful dependencies. A
submitted capture cannot accept appended rows. Packed input/output snapshots and
buffer leases prevent overwriting an in-flight batch. Larger captures, tail
packing and crop choice need both forward-row/padding counts and elapsed time.

Dynamic high/low watermarks, queue priorities and CPU/GPU resource allocation
remain work. Compare ready depth, producer outstanding depth, shape grouping,
active game count and solver capacity under the same clock. A deeper queue earns
its memory and latency cost if it improves relevant root coverage, continuation
progress or proof delivery while preventing idle feeding. Withdraw newly exact
rows and deprioritize obsolete work before launch.

## Retention, pruning and reconnection

Store search evidence across turns without searching every disconnected branch.
The bounded dormant archive keeps useful disconnected nodes and predictions;
active stores can reconnect them when a legal continuation reaches the same
context. Shared rule outcomes remain distinct from neural contexts.

For current coloured sets X/O and a retained variation X'/O', a stone in
X intersect O' or O intersect X' makes that variation impossible in forward-only
play. PR395's opt-in forward archive mode removes conflicting dormant entries;
pending or pinned entries wait until their lifetime permits removal. Active
entries receive this check when archived. Analysis/undo retains its separate
lifetime semantics.

The missing-coloured-stone count gives a lower bound on placements needed to
reach a compatible variation. Use that bound with retained evidence, age, size
and estimated future usefulness to rank retention. It does not supply a legal
move ordering, mover/remaining phase or identical neural context by itself.
Do not keep disconnected graphs actively searching just to preserve them.

Index and update costs are part of the CPU schedule. Reverse parents, dirty
refreshes and compaction can delay the next batch. No blanket amortized-O(1)
claim follows from stopping a dirty walk at its first dirty ancestor. Avoid
counting eviction evidence both as a per-edge fallback and again in node-wide
carried totals.

## Training and machine-level allocation

A row records an estimate about its own position, raw prediction, root evidence,
comparison credits, model and search provenance. The learner constructs value
targets from outcome, bootstrapping and calibration under its existing settings.
Recorded root Q is not automatically the value target. Same-position inherited
evidence can improve an estimate; values carried from other positions are a
separate phenomenon. Target age has not been established as the training
regression's cause.

Refuted legal moves remain in the policy-loss normalization. Search eligibility
and training legality are separate masks; that behavior is already correct.
An off-game proof supplies only its own established labels and actual history,
not the outcome of the played game. Quiet preceding decisions need their own
search/evidence. Prefetch and unfinished exploration do not become full policy
targets merely because the GPU evaluated them.

New row kinds need explicit retention, masks, weights and learner-window
accounting. Native actor integration exists, but proof/depth rows and inherited
root estimates still need a controlled learning comparison. A solver auxiliary
head is an optional parallel experiment. It does not delay scheduling or create
proofs from predicted success.

Actors and the learner share the GPU. Optimize progress of both when training is
active. Keep learner updates productive, useful data sufficiently fresh, and
CPU capacity for graph feeding/data loading while proofs run. Maximum actor
inference alone is not maximum training improvement. Pause/fence mechanisms
retain queued search work; resource coordination must use them rather than
abandon CUDA readers or launch uncontrolled competing jobs.

## Validation and remaining work

Behavior tests establish legality, sound complete-coverage proofs, propagation
through shared parents, exact precedence, cancellation/reuse, balanced credits,
retargeting, bounded retention, prediction identity and native/browser contracts.
Encoder agreement with the encoding contract is useful. Reproducing old search
rows or random hashes is not the acceptance criterion for a changed algorithm.

Equal-clock comparisons pin weights, openings, resource ceilings and warmed
backends. Report actual CPU usage and shared clients. Fixed-work analysis remains
useful for controlled diagnostics, but fixed-simulation Elo is not a rejection
rule for a scheduler that changes available throughput and allocation.

Fixed counts alone do not guarantee deterministic asynchronous search. A
reproducible scheduler mode also needs defined initial table/cache state,
worker assignment, ordered cache-hit and GPU result visibility, and a specified
set of issued solver jobs to wait for or defer at each boundary. Sorting only
already-arrived answers by ID does not remove the completion-set race. The
interactive clocked path makes no bit-for-bit repeatability promise.

Measure unique submitted NN rows, forward rows, padding, graph installations,
cache/coalescing, queued withdrawals, issued/completed/cancelled work, root
credits/coverage, completed-view depth and refutation time separately. Solver
fresh nodes, independent certificates, waiting/service spans and continuation
reuse also stay separate. Worker wall share is not CPU occupancy. Gaps in our
CUDA stream are not device-wide GPU idle; occupancy claims need device/context
tracing. Profiled runs explain causes; unprofiled paired runs measure speed.

The next work is:

1. Trace ready-to-launch stalls with the actual two-flight launcher and classify
   unexpanded, dependency-blocked, reserved, ready, submitted and exact work.
   Do not infer readiness from the count of open graph nodes alone.
2. Measure repeated CPU/GPU cycles with causal handoff IDs/timestamps, relevant
   decision effects and avoided work. Existing history joins have limited scope.
3. Tune useful ready surplus and outstanding depth across one game and many games,
   then implement dynamic admission based on measured starvation/age/deadline
   costs. Preserve depth opportunities and legal reopening.
4. Reduce remaining native selection, expansion/installation and graph-update
   costs using the measured task graph. Do not assume Python removal solves them.
5. Improve solver continuation and premise reuse where measured fresh work or
   lost frontier state justifies it; refine task priorities alongside neural work.
6. Expose coherent time controls, cancellation, retargeting and pondering in Play
   and analysis. Source APIs exist; complete UI/adoption is unfinished.
7. Validate actor/replay/learner utility and learning at equal total resources,
   including weighting of new evidence. Controlled production adoption follows
   reviewed source and explicit resource-state coordination.

Six's native workers feeding one inference queue are a useful supply design.
Its self-play uses Gumbel top-m/sequential halving at the root and PUCT below;
its protocol play configuration is different. HeXO's backend is already fast
in measured cases. The opportunity is useful supply, lean graph handling and
repeatable proof/neural cooperation, then better learning from that evidence.

## Source map

| Contract | Source |
| --- | --- |
| Graph, complete legal records, sparse selection, root halving, exact propagation | `src/gumbel.cpp` |
| Views, candidate allocation, records, path admission, retarget/lifecycle | `src/gumbel_owner.cpp` |
| Bulk native feeding, deduplication, encoding, cache, packed results | `src/gumbel_feed.cpp`, `src/gumbel_batch.cpp` |
| Independent producers, broker queue, completion ownership, retirement | `src/gumbel_broker.hpp`, `src/gumbel_parallel.hpp` |
| CPU frontier, immutable jobs, scoped facts, native/external workers, endpoint replay | `src/gumbel_proof.cpp` |
| Search pools, proof and inference adapters | `python/native_scheduler.py`, `python/native_dense.py` |
| Native actor data/lifecycle | `python/native_selfplay.py`, `python/dense_selfplay.py` |
| CUDA capacities and tail packing | `python/hexnet_graphs.py` |
| Solver query/cancel/answer/frontier contracts | `tools/tactical/src/lib.rs`, `tools/tactical/src/native_answer.rs`, vendored prover sources |
| Browser owner/GPU/solver adapters | `web/engine/search.mjs`, `web/engine/tactical.mjs`, `web/engine/network.mjs` |
| Actor and learner target construction | `python/dense_selfplay.py`, `python/dense_data.py`, `python/dense_learn.py` |
| Six scheduling reference | `tools/six/src/mcts.cpp`, `tools/six/web_bot.cpp` |

Further design references: [KataGo graph search](https://github.com/lightvector/KataGo/blob/master/docs/GraphSearch.md),
[Gumbel planning](https://openreview.net/forum?id=bERaNdoegnO),
[Batch MCTS](https://arxiv.org/abs/2104.04278), and
[KataGo training methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md).
These are design references, not measurements of this implementation.
