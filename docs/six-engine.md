# Six engine protocol

Bubble can run as a Six engine with its promoted checkpoint:

```text
python python/six_engine.py serve --run runs/dense-v1 --device cpu --simulations 128 --solver-nodes 32768
```

`--model path/to/ema.pt` selects one export directly instead of `--run`. `--device auto` uses CUDA when available;
`cpu` keeps it off the GPU. The server accepts `position radius 8 moves ...`, `go`, `stop`, and the standard
handshake and game commands. It rejects `setup`, `tomove`, and other radii. `go movetime` is advisory: Bubble
uses its configured simulation and solver budgets.

In a Six checkout, add these four lines inside `parse_spec` in `arena/engines.py`. Set the two paths to this
checkout and the desired run. The Python executable should have Bubble's dependencies installed.

```python
    if kind == "bubble":
        sims = int(option or 128)
        command = [sys.executable, r"C:\HeXO\python\six_engine.py", "serve", "--run", r"C:\HeXO\runs\dense-v1", "--device", "cpu", "--simulations", str(sims)]
        return EngineSpec(f"Bubble {sims} sims", command, "go", 600)
```

Then run from Six's root:

```text
py -3.12 arena/match.py bubble:128 hexnet:1000:C:/Six/gen-0120.onnx --pairs 50 --radius 8 --concurrency 1
```

For evaluation inside Bubble, set the external anchor command and its league name:

```text
python python/dense_eval.py loop --run runs/dense-v1 --eval-external-engine "C:/Six/sixengine.exe --net C:/Six/gen-0120.onnx --cpu" --eval-external-name six
```

The command is split with `shlex`; quote paths containing spaces inside the command string. `seal_ms` is the
per-turn time for either anchor. Existing `seal` reports and league entries keep their IDs; Six results appear
under `six` in `league.json`, reports, and the dashboard.
