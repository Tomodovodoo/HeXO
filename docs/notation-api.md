# Notation and bot API

`python/notation.py` reads and writes [HTTTX notation v1](https://github.com/hex-tic-tac-toe/hexagonal-tic-tac-toe-notation/tree/15bb7877ae020d661497e332adf0810d00d24e3e). Cross is native player 0; its origin stone is implicit in the text and present in imported histories. Import checks turn numbers, radius-8 legality and terminal states, and keeps metadata and `!` annotations as they are. From Python, use `loads(text)` and `dumps(record_or_history)`.

```sh
python python/notation.py import match.txt > match.json
python python/notation.py export match.json > match.txt
python python/bot_api.py --port 8790 --ms 100
```

`python/bot_api.py` serves the [HTTTX stateless bot API](https://github.com/hex-tic-tac-toe/htttx-bot-api/tree/37d2385f1016abe8b25798238a7d0c4a17a25dda/definitions) on localhost: `GET /capabilities.json` and `POST /stateless/v1-alpha/turn` with a body like `{"board":{"to_move":"o","cells":[{"q":0,"r":0,"p":"x"}]},"request_id":1}`. It answers with `move.pieces`. The board is unordered, so the adapter reconstructs a legal ordering of the given cells; an unreachable position is a 400, and a reconstruction over one second or 20,000 states is a 503. `--model` loads a native NNUE export; without it the handwritten evaluator plays. `time_limit` is advisory.

A final turn may hold one stone, whether it won or the turn is still open; import and export preserve it. The
origin-only board is `version[1];` with no turns. Only a history without the origin raises `NotationConflict`.
The HTTP adapter accepts the origin turn, a half-turn remainder and a first-stone win, returning their actual
placements. Finished boards return 409. `python -m unittest tests.test_notation_api -v` covers the protocol.
For dense models, enforced turn limits, WebSocket sessions and live clocks, use the
[timed API](time-controls.md).

## HTTTX v2 studies

The Play page's HTTTX export saves the whole study in [HTTTX v2](https://github.com/hex-tic-tac-toe/htttx-notation/tree/50fd5c7cde0475a5d8dcae5644075946a14b16f6), including side lines, evaluations, recorded clocks and imported cell highlights and labels. The other text formats still export the shown history. JSON replay files and browser backups retain the study alongside the engine records. V1 import and the existing Python history-to-v1 API remain available.

Play a different stone from an earlier position to create a variation. The move score shows the main line continuously and inserts each alternative at the turn it replaces. Alternatives can contain further alternatives at any supported depth. Select an individual stone to navigate; the board, graph and review follow that line. Each branch can be collapsed independently, and selecting a hidden descendant reveals its ancestors. Deep branches stop adding horizontal indentation so the score still fits a phone.

The selected branch has Promote, Make main and Return to main controls. Right-click an individual stone or use Shift+F10 to open actions for that move, including promotion of an unselected branch. The menu stays inside the viewport, supports arrow keys and Escape, and keeps the clicked position after promotion. The visible branch controls remain available on touch screens. Promote changes the preferred continuation within the nearest parent; Make main promotes the whole route to the main line. Neither discards alternatives. Exploration changes both seats to Human and pauses play, while retaining the analysis engine. Evaluations and clocks appear beside the stone they annotate, and the selected stone stays highlighted in the score. Cell highlights and labels apply only at their recorded position. V2 does not define prose comments; the viewer does not invent a comment syntax or represent imported closed evaluations as locally verified proofs.

Save engine line keeps the displayed principal variation or candidate turn as another study line. Position evaluations are associated with their placement prefixes, so switching lines also switches the graph, candidates and review to that line. Imported evaluations use X's perspective; native evaluations are converted on export. Closed evaluations remain annotations rather than locally verified solver proofs. Clocks are integer milliseconds.

`notation.loads` dispatches on the version tag. V2 returns `notation_v2.StudyRecord`, with metadata, optional setup and a recursive list of turns. Each turn contains two moves and its replacement variations. A move contains a coordinate or final `[/]`, optional info and visuals. `notation.dumps` accepts that record; the command-line JSON conversion preserves it. Unknown future visual blocks are retained and ignored by the renderer.

The notation parser and writer preserve arbitrary setup positions. The current play/search engine still requires a placement history, so the Play page refuses custom starting boards beyond the default origin or the empty board. It does not reconstruct or silently change them. Analysing those setups requires a separate engine position-loading change. Limits are 1 MB, 4096 explicit stones across all lines and 64 nested variation levels.

The interaction follows [Lichess's analysis context menu](https://github.com/lichess-org/lila/blob/master/ui/analyse/src/treeView/contextMenu.ts), including separate local promotion and main-line promotion, with controls visible for touch users.
