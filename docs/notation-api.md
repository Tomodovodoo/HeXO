# Notation and bot API

`python/notation.py` reads and writes [HTTTX notation v1](https://github.com/hex-tic-tac-toe/hexagonal-tic-tac-toe-notation/tree/15bb7877ae020d661497e332adf0810d00d24e3e). Cross is native player 0; its origin stone is implicit in the text and present in imported histories. Import checks turn numbers, radius-8 legality and terminal states, and keeps metadata and `!` annotations as they are. From Python, use `loads(text)` and `dumps(record_or_history)`.

```sh
python python/notation.py import match.txt > match.json
python python/notation.py export match.json > match.txt
python python/bot_api.py --port 8790 --ms 100
```

`python/bot_api.py` serves the [HTTTX stateless bot API](https://github.com/hex-tic-tac-toe/htttx-bot-api/tree/37d2385f1016abe8b25798238a7d0c4a17a25dda/definitions) on localhost: `GET /capabilities.json` and `POST /stateless/v1-alpha/turn` with a body like `{"board":{"to_move":"o","cells":[{"q":0,"r":0,"p":"x"}]},"request_id":1}`. It answers with `move.pieces`. The board is unordered, so the adapter reconstructs a legal ordering of the given cells; an unreachable position is a 400, and a reconstruction over one second or 20,000 states is a 503. `--model` loads a native NNUE export; without it the handwritten evaluator plays. `time_limit` is advisory.

A final turn may hold one stone, whether it won or the turn is still open, as the reference implementations write it; import accepts it. The origin-only board is `version[1];` with no turns. Only a history without the origin raises `NotationConflict`. The API's reply must carry two pieces and forbids placements after a win, so it returns 409 for a first-stone win, for partial turns and for finished boards. `python -m unittest tests.test_notation_api -v` covers the protocol.
