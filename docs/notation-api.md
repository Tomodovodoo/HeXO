## Official notation and local bot API

`notation.py` imports and exports [notation v1](https://github.com/hex-tic-tac-toe/hexagonal-tic-tac-toe-notation/tree/15bb7877ae020d661497e332adf0810d00d24e3e).
Cross is native player 0; its origin placement is implicit in the text and present
in imported histories. Turn numbers, sequential radius-8 legality, and terminal
states are checked. Metadata and `!` annotations are preserved without inferring
their meaning. The upstream example uses `datetime` rather than `utcdatetime`,
and a named time control that differs from its numeric grammar; these values are
kept intact. Python callers can use `loads(text)` and `dumps(record_or_history)`.

```sh
python python/notation.py import match.txt > match.json
python python/notation.py export match.json > match-roundtrip.txt
python python/bot_api.py --port 8790 --ms 100
```

The loopback HTTP adapter exposes `GET /capabilities.json` and
`POST /stateless/v1-alpha/turn` according to the
[published API definitions](https://github.com/hex-tic-tac-toe/htttx-bot-api/tree/37d2385f1016abe8b25798238a7d0c4a17a25dda/definitions).
For example, send `{"board":{"to_move":"o","cells":[{"q":0,"r":0,"p":"x"}]},"request_id":1}`
with content type `application/json`. Responses contain `move.pieces` objects
with `q` and `r`, and echo an optional request ID. Add `--model path/to/model.bin`
to use a native NNUE export; capabilities identify its SHA256. Otherwise the
adapter uses the handwritten evaluator. This command does not register a bot
with any external service.

Both published formats require exactly two placements per recorded turn or API
move, while the API also forbids placements after a win. A first-placement win
cannot satisfy both requirements. Notation export raises `NotationConflict` for
that case, partial turns, and empty boards. The API returns HTTP 409 for origin
turns, partial turns, already-terminal boards, or a chosen first-placement win;
it never pads a winning move. Full two-placement wins are supported.

The stateless board is unordered. The adapter reconstructs a legal ordering of
exactly the supplied cells and checks counts and `to_move`; it never assumes array
order is history. Unreachable positions return 400, and a reconstruction exceeding
one second or 20,000 visited states returns 503. Local limits are 1 MiB request
bodies and 1,025 board cells; notation accepts 4,097 placements and 1 MiB text.
`time_limit` is used as an advisory search cap, with zero returning 408. Native
search and reconstruction cannot guarantee a hard response deadline, so
`move_time_limit` is false. Websocket and matchmaking capabilities are not declared.
Run `python -m unittest tests.test_notation_api -v` for the protocol checks.

The local adapter bounds the request line and headers together to two seconds,
including clients that keep sending bytes. Request-body transfer has a separate
two-second deadline. These transport limits are separate from its advisory search
budget. A configured model that disappears or becomes unreadable returns JSON 503.
