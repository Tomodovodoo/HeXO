"""Experimental full-legal KLENT: frozen collection, signed returns, one fresh fit pass.

Neural inference/fitting batches on CUDA; exact rules and replay run in C++ on CPU.
The native deployment export remains HXNNUE1. The Q head is training-only.
"""
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch
from torch import nn

from hexo import Game, ROOT, library
from nnue_model import NNUE, SCORE_SCALE
from train import digest, write_json

SCHEMA = "hexo-klent-scalar-v1"


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.nnue = NNUE()
        self.q = nn.Sequential(nn.Linear(104, 16), nn.ReLU(), nn.Linear(16, 1), nn.Tanh())
        nn.init.zeros_(self.q[2].weight)
        nn.init.zeros_(self.q[2].bias)

    def forward(self, batch):
        inputs, candidates = self.nnue.features(batch)
        return self.nnue.policy(candidates).squeeze(-1), self.q(candidates).squeeze(-1), inputs


def segmented_log_softmax(logits, owner, count):
    maxima = logits.new_full((count,), -torch.inf)
    maxima.scatter_reduce_(0, owner, logits, reduce="amax", include_self=True)
    shifted = logits-maxima[owner]
    mass = logits.new_zeros(count).scatter_add_(0, owner, shifted.exp())
    return shifted-mass.log()[owner]


def improved_policy(logits, q, owner, count, alpha, beta):
    if alpha < 0 or beta < 0 or alpha+beta <= 0:
        raise ValueError("Require alpha,beta >=0 with positive sum")
    dtype = torch.promote_types(torch.float32, torch.promote_types(logits.dtype, q.dtype))
    q = q.to(dtype)
    log_pi = segmented_log_softmax(logits.to(dtype), owner, count)
    log_mu = segmented_log_softmax((q+beta*log_pi)/(alpha+beta), owner, count)
    mu = log_mu.exp()
    vhat = q.new_zeros(count).scatter_add_(0, owner, mu*q)
    kl = q.new_zeros(count).scatter_add_(0, owner, mu*(log_mu-log_pi))
    entropy = q.new_zeros(count).scatter_add_(0, owner, -mu*log_mu)
    return mu, vhat, kl, entropy


def signed_returns(players, values, terminal, tail_player=None, tail_value=None, gamma=1., lam=math.exp(-1/16)):
    """Each row precedes one placement. A cap bootstraps its actual next state."""
    if not 0 <= lam <= 1 or not 0 < gamma <= 1 or not players or len(players) != len(values):
        raise ValueError("Invalid return parameters/trajectory")
    if not all(p in (0, 1) for p in players) or not np.isfinite(values).all() or np.max(np.abs(values)) > 1.00001:
        raise ValueError("Invalid acting players/values")
    result = np.empty(len(players), np.float32)
    if terminal:
        result[-1] = 1.
    else:
        if tail_player not in (0, 1) or tail_value is None or not math.isfinite(tail_value) or abs(tail_value) > 1.00001:
            raise ValueError("Capped trajectory requires its frozen-actor tail bootstrap")
        result[-1] = (1 if players[-1] == tail_player else -1)*gamma*tail_value
    for t in range(len(players)-2, -1, -1):
        sign = 1 if players[t] == players[t+1] else -1
        result[t] = sign*gamma*((1-lam)*values[t+1]+lam*result[t+1])
    return result


_digits = np.arange(729)[:, None]//(3**np.arange(6)) % 3
_black, _white = (_digits == 1).sum(1), (_digits == 2).sum(1)
_weights = np.asarray([0, 1, 12, 150, 2400, 24000, 1000000])
_baseline = np.where(_white == 0, _weights[_black], np.where(_black == 0, -_weights[_white], 0))


def observe(game, *, value_only=False):
    legal = game.legal_moves()
    if not legal:
        raise ValueError("Cannot act on terminal board")
    coords = np.asarray(legal, dtype="<i8")
    result = {"centers": game.nnue_centers(), "phase": game.nnue_context(), "player": game.player,
            "baseline": float(np.asarray(game.features(), np.int64)@_baseline)*(1 if game.player == 0 else -1),
            "legal": legal, "legal_sha256": hashlib.sha256(coords.tobytes()).hexdigest()}
    if not value_only:
        result["candidate_codes"], result["pairs"] = game.nnue_policy_batch(coords)
    return result


def pack(observations, device, *, value_only=False):
    nc = np.asarray([len(o["centers"]) for o in observations])
    na = np.asarray([len(o["legal"]) for o in observations])
    keys = ("centers",) if value_only else ("centers", "candidate_codes", "pairs")
    arrays = {key: np.concatenate([o[key] for o in observations]) for key in keys}
    arrays.update({key: np.asarray([o[key] for o in observations]) for key in ("phase", "player", "baseline")})
    arrays["baseline"] = arrays["baseline"].astype(np.float32)
    arrays["phase"] = arrays["phase"].astype(np.float32)
    arrays.update(center_owner=np.repeat(np.arange(len(nc)), nc), center_counts=nc.astype(np.float32))
    if not value_only:
        arrays.update(candidate_owner=np.repeat(np.arange(len(na)), na), offsets=np.cumsum(np.r_[0, na]))
    return {k: torch.as_tensor(v, device=device) for k, v in arrays.items()}


def chunks(observations, args):
    part, cells, centers = [], 0, 0
    for index, row in enumerate(observations):
        a, c = len(row["legal"]), len(row["centers"])
        if a > args.cells or c > args.centers:
            raise ValueError("A full position exceeds --cells/--centers; increase the budget (actions are never cropped)")
        if part and (len(part) >= args.batch or cells+a > args.cells or centers+c > args.centers):
            yield part
            part, cells, centers = [], 0, 0
        part.append(index)
        cells += a
        centers += c
    if part:
        yield part


@torch.no_grad()
def act(model, observations, args):
    results = [None]*len(observations)
    for ids in chunks(observations, args):
        batch = pack([observations[i] for i in ids], args.device)
        logits, q, _ = model(batch)
        mu, values, kl, entropy = improved_policy(logits, q, batch["candidate_owner"], len(ids), args.alpha, args.beta)
        flat = mu.cpu().numpy()
        offsets = batch["offsets"].cpu().numpy()
        values, kl, entropy = (x.cpu().numpy() for x in (values, kl, entropy))
        for j, index in enumerate(ids):
            p = flat[offsets[j]:offsets[j+1]].astype(np.float64)
            p /= p.sum()
            results[index] = (p.astype(np.float32), float(values[j]), float(kl[j]), float(entropy[j]))
    return results


def collect(model, args, iteration, progress=None):
    model.eval()
    rng = np.random.default_rng(args.seed+iteration*10000)
    episodes, rows, live = [], [], []
    started = 0
    try:
        while len(episodes) < args.games:
            while len(live) < args.envs and started < args.games:
                live.append({"game": Game(), "id": started, "moves": [], "rows": []})
                started += 1
            observations = [observe(slot["game"]) for slot in live]
            outputs = act(model, observations, args)
            finished, tails = [], []
            for slot, obs, (mu, value, kl, entropy) in zip(live, observations, outputs):
                game = slot["game"]
                probs = mu.astype(np.float64); probs /= probs.sum()
                chosen = int(rng.choice(len(mu), p=probs))
                move = obs["legal"][chosen]
                row = {"game": slot["id"], "ply": len(slot["moves"]), "player": game.player,
                       "remaining": game.remaining, "action": list(move), "chosen": chosen,
                       "legal_sha256": obs["legal_sha256"], "mu": mu, "vhat": value,
                       "kl": kl, "entropy": entropy}
                slot["rows"].append(row)
                slot["moves"].append(list(move))
                game.play(*move)
                if game.winner >= 0 or len(slot["moves"]) >= args.max_plies:
                    finished.append(slot)
                    if game.winner < 0:
                        tails.append(slot)
            tail_values = act(model, [observe(s["game"]) for s in tails], args) if tails else []
            for slot, output in zip(tails, tail_values):
                slot["tail_value"] = output[1]
            for slot in finished:
                game = slot["game"]
                terminal = game.winner >= 0
                returns = signed_returns([r["player"] for r in slot["rows"]], [r["vhat"] for r in slot["rows"]],
                                         terminal, game.player, slot.get("tail_value"), args.gamma, args.lambda_return)
                for row, target in zip(slot["rows"], returns):
                    row.update(target=float(target), bootstrapped=not terminal)
                rows.extend(slot["rows"])
                episodes.append({"id": slot["id"], "moves": slot["moves"], "winner": game.winner,
                                 "reason": "six-in-a-row" if terminal else "cap",
                                 "tail_value": slot.get("tail_value"), "tail_player": game.player})
                game.close()
                live.remove(slot)
            if progress:
                progress(len(episodes), sum(len(s["moves"]) for s in live)+len(rows))
        return episodes, rows
    finally:
        for slot in live:
            slot["game"].close()


def rebuild(row, episodes, *, value_only=False):
    with_game = Game(episodes[row["game"]]["moves"][:row["ply"]])
    try:
        obs = observe(with_game, value_only=value_only)
        if (obs["legal_sha256"] != row["legal_sha256"] or obs["player"] != row["player"]
                or with_game.remaining != row["remaining"] or obs["legal"][row["chosen"]] != tuple(row["action"])
                or len(obs["legal"]) != len(row["mu"])):
            raise ValueError("KLENT replay geometry/phase/action ordering changed")
        return obs
    finally:
        with_game.close()


def loss(model, batch, target_policy, taken, returns):
    logits, q, _ = model(batch)
    log_pi = segmented_log_softmax(logits, batch["candidate_owner"], len(returns))
    ce = -(target_policy*log_pi).sum()/len(returns)
    mse = (q[taken]-returns).square().mean()
    return ce+mse, ce.detach(), mse.detach()


def fit(model, optimizer, value_optimizer, episodes, rows, args, iteration, progress=None):
    """Exactly one actor pass, followed by one explicitly separate V-head pass."""
    model.train()
    episodes = {e["id"]: e for e in episodes}
    order = np.random.default_rng(args.seed+iteration).permutation(len(rows))
    totals = np.zeros(3)
    updates = value_updates = examples = value_examples = 0
    # Only one position-batch's observations are resident; full legal sets remain intact.
    for start in range(0, len(order), args.batch):
        selected = [rows[int(i)] for i in order[start:start+args.batch]]
        observations = [rebuild(row, episodes) for row in selected]
        optimizer.zero_grad(set_to_none=True)
        for ids in chunks(observations, args):
            batch = pack([observations[i] for i in ids], args.device)
            target = torch.as_tensor(np.concatenate([selected[i]["mu"] for i in ids]), device=args.device)
            returns = torch.tensor([selected[i]["target"] for i in ids], device=args.device)
            taken = batch["offsets"][:-1]+torch.tensor([selected[i]["chosen"] for i in ids], device=args.device)
            objective, ce, mse = loss(model, batch, target, taken, returns)
            (objective*(len(ids)/len(selected))).backward()
            totals[:2] += np.array([ce.item(), mse.item()])*len(ids)
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite KLENT parameters")
        updates += 1
        examples += len(selected)
        if progress:
            progress({"fit_phase": "actor and Q", "fit_completed": examples, "fit_total": len(rows),
                      "optimizer_steps": updates, "examples_processed": examples,
                      "value_optimizer_steps": 0, "value_examples_processed": 0,
                      "policy_ce": totals[0]/examples, "q_mse": totals[1]/examples,
                      "deployment_value_mse": None})
    # Freeze the shared representation during deployment value distillation.
    # This term is not part of the KLENT actor/critic objective.
    # A zero-Q, all-capped corpus supplies no reason to erase a pretrained V.
    value_order = order if any(abs(r["target"]) > 1e-8 for r in rows) else []
    for start in range(0, len(value_order), args.batch):
        selected = [rows[int(i)] for i in value_order[start:start+args.batch]]
        observations = [rebuild(row, episodes, value_only=True) for row in selected]
        value_optimizer.zero_grad(set_to_none=True)
        for ids in chunks(observations, args):
            batch = pack([observations[i] for i in ids], args.device, value_only=True)
            with torch.no_grad():
                inputs = model.nnue.position_features(batch)
            value = torch.tanh(batch["baseline"]/SCORE_SCALE+model.nnue.value(inputs).squeeze(-1))
            target = torch.tensor([selected[i]["target"] for i in ids], device=args.device)
            value_loss = (value-target).square().mean()
            (value_loss*(len(ids)/len(selected))).backward()
            totals[2] += value_loss.item()*len(ids)
        value_optimizer.step()
        value_updates += 1
        value_examples += len(selected)
        if progress:
            progress({"fit_phase": "deployment value", "fit_completed": value_examples, "fit_total": len(value_order),
                      "optimizer_steps": updates, "examples_processed": examples,
                      "value_optimizer_steps": value_updates, "value_examples_processed": value_examples,
                      "policy_ce": totals[0]/examples, "q_mse": totals[1]/examples,
                      "deployment_value_mse": totals[2]/value_examples})
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Nonfinite deployment value parameters")
    return dict(zip(("policy_ce", "q_mse", "deployment_value_mse"), (totals/len(rows)).tolist()),
                optimizer_steps=updates, examples_processed=examples, value_optimizer_steps=value_updates,
                value_examples_processed=value_examples, deployment_value_fitted=bool(len(value_order)))


def verify(directory, identity):
    manifest = json.loads((directory/"manifest.json").read_text())
    if manifest["schema"] != SCHEMA or manifest["identity"] != identity:
        raise ValueError("KLENT artifact identity changed")
    for name, sha in manifest["files"].items():
        if Path(name).name != name or digest(directory/name) != sha:
            raise ValueError("KLENT artifact hash changed")
    return manifest


def publish(directory, identity, writer, metrics=None):
    if directory.exists():
        raise ValueError("Refusing to overwrite a KLENT artifact")
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix="pending-") as temporary:
        stage = Path(temporary)/"artifact"
        stage.mkdir()
        writer(stage)
        manifest = {"schema": SCHEMA, "identity": identity, "metrics": metrics,
                    "files": {p.name: digest(p) for p in stage.iterdir()}}
        write_json(stage/"manifest.json", manifest)
        stage.rename(directory)
    return manifest


def save_corpus(directory, identity, episodes, rows):
    def writer(stage):
        write_json(stage/"episodes.json", episodes)
        write_json(stage/"rows.json", [{k: v for k, v in row.items() if k != "mu"} for row in rows])
        np.savez_compressed(stage/"policies.npz", offsets=np.cumsum([0]+[len(r["mu"]) for r in rows]),
                            probs=np.concatenate([r["mu"] for r in rows]))
    return publish(directory, identity, writer)


def load_corpus(directory, identity):
    manifest = verify(directory, identity)
    episodes = json.loads((directory/"episodes.json").read_text())
    rows = json.loads((directory/"rows.json").read_text())
    with np.load(directory/"policies.npz", allow_pickle=False) as arrays:
        offsets, probs = arrays["offsets"], arrays["probs"]
        if len(offsets) != len(rows)+1 or offsets[0] != 0 or offsets[-1] != len(probs) or np.any(np.diff(offsets) <= 0):
            raise ValueError("Malformed KLENT corpus offsets")
        for i, row in enumerate(rows):
            row["mu"] = probs[offsets[i]:offsets[i+1]].copy()
            if not np.isfinite(row["mu"]).all() or np.any(row["mu"] < 0) or not np.isclose(row["mu"].sum(), 1):
                raise ValueError("Malformed KLENT policy target")
    return episodes, rows, manifest


def main(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k not in ("run", "iterations")}
    initial = {}
    for key in ("initial_model", "initial_q"):
        path = getattr(args, key, None)
        if path:
            initial[key] = Path(path).read_bytes()
            config[key] = str(Path(path).resolve())
            config[key+"_sha256"] = hashlib.sha256(initial[key]).hexdigest()
    config["q_initialization"] = "explicit-state" if "initial_q" in initial else "zero-output"
    identity = {"run": str(run), "config": config, "engine_sha256": digest(library),
                "sources": {name: digest(ROOT/name) for name in ("klent.py", "nnue_model.py", "hexo.py", "train.py")}}
    lock = run/"training.lock"
    try:
        with lock.open("x") as handle:
            handle.write(str(os.getpid()))
    except FileExistsError:
        raise ValueError("Run has a training.lock; do not remove it while its process may be alive") from None
    try:
        for name in ("checkpoints", "corpus"):
            (run/name).mkdir(exist_ok=True)
        model = Model().to(args.device)
        optimizer = torch.optim.Adam([p for name, p in model.named_parameters() if not name.startswith("nnue.value.")],
                                     lr=args.lr, fused=args.device == "cuda")
        value_optimizer = torch.optim.Adam(model.nnue.value.parameters(), lr=args.lr, fused=args.device == "cuda")
        def checkpoint(stage):
            torch.save({"schema": SCHEMA, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "value_optimizer": value_optimizer.state_dict()}, stage/"klent.pt")
            torch.save(model.nnue.state_dict(), stage/"model.pt")
            torch.save({"schema": SCHEMA, "model_sha256": digest(stage/"model.pt"),
                        "state": model.q.state_dict()}, stage/"q.pt")
            model.nnue.export(stage/"model.nnue")
        checkpoints = sorted((run/"checkpoints").glob("[0-9][0-9][0-9][0-9]"))
        if checkpoints:
            for index, path in enumerate(checkpoints):
                if int(path.name) != index:
                    raise ValueError("KLENT checkpoint sequence has a gap")
                manifest = verify(path, identity)
                if index:
                    prior = run/"checkpoints"/f"{index-1:04d}"
                    corpus = run/"corpus"/f"{index:04d}"
                    corpus_identity = {**identity, "iteration": index, "actor_sha256": digest(prior/"klent.pt"),
                                       "policy": "softmax((Q+beta*logpi)/(alpha+beta))",
                                       "returns": "signed-lambda-v1-cap-bootstrap", "actions": "full-legal"}
                    verify(corpus, corpus_identity)
                    if digest(corpus/"manifest.json") != manifest["metrics"]["corpus_manifest_sha256"]:
                        raise ValueError("Consumed KLENT corpus manifest changed")
            latest = checkpoints[-1]
            saved = torch.load(latest/"klent.pt", map_location=args.device, weights_only=True)
            if saved["schema"] != SCHEMA:
                raise ValueError("Unknown KLENT checkpoint schema")
            model.load_state_dict(saved["model"], strict=True)
            optimizer.load_state_dict(saved["optimizer"])
            value_optimizer.load_state_dict(saved["value_optimizer"])
            iteration = int(latest.name)
        else:
            if "initial_model" in initial:
                model.nnue.load_state_dict(torch.load(io.BytesIO(initial["initial_model"]), map_location=args.device, weights_only=True), strict=True)
            if "initial_q" in initial:
                q_state = torch.load(io.BytesIO(initial["initial_q"]), map_location=args.device, weights_only=True)
                if ("initial_model" not in initial or q_state.get("schema") != SCHEMA
                        or q_state.get("model_sha256") != config["initial_model_sha256"]):
                    raise ValueError("Initial Q requires its exact matching model.pt representation")
                model.q.load_state_dict(q_state["state"], strict=True)
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Initial checkpoint has nonfinite parameters")
            iteration = 0
            latest = run/"checkpoints/0000"
            publish(latest, identity, checkpoint)
        for number in range(iteration+1, iteration+args.iterations+1):
            started = time.perf_counter()
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            corpus_identity = {**identity, "iteration": number, "actor_sha256": digest(latest/"klent.pt"),
                               "policy": "softmax((Q+beta*logpi)/(alpha+beta))",
                               "returns": "signed-lambda-v1-cap-bootstrap", "actions": "full-legal"}
            corpus = run/"corpus"/f"{number:04d}"
            def status(stage, **values):
                write_json(run/"status.json", {"iteration": number, "stage": stage,
                           "updated_at": time.time(), "schema": SCHEMA,
                           "actor_sha256": corpus_identity["actor_sha256"], "games_total": args.games, **values})
            if not corpus.exists():
                last = [0.]
                status("collection", games=0, positions=0)
                def progress(completed, positions):
                    if time.monotonic()-last[0] >= .5:
                        status("collection", games=completed, positions=positions)
                        last[0] = time.monotonic()
                episodes, rows = collect(model, args, number, progress)
                save_corpus(corpus, corpus_identity, episodes, rows)
            episodes, rows, corpus_manifest = load_corpus(corpus, corpus_identity)
            counts = {"games": len(episodes), "positions": len(rows),
                      "terminal_games": sum(e["winner"] >= 0 for e in episodes),
                      "bootstrapped_games": sum(e["winner"] < 0 for e in episodes)}
            status("fitting", **counts)
            last_fit = [0.]
            def fit_progress(values):
                if time.monotonic()-last_fit[0] >= .5 or values["fit_completed"] == values["fit_total"]:
                    status("fitting", **counts, **values)
                    last_fit[0] = time.monotonic()
            metrics = fit(model, optimizer, value_optimizer, episodes, rows, args, number, progress=fit_progress)
            metrics.update(iteration=number, games=len(episodes), terminal_games=sum(e["winner"] >= 0 for e in episodes),
                           bootstrapped_games=sum(e["winner"] < 0 for e in episodes), positions=len(rows),
                           acting_kl=float(np.mean([r["kl"] for r in rows])),
                           acting_entropy=float(np.mean([r["entropy"] for r in rows])),
                           acting_normalized_entropy=float(np.mean([r["entropy"]/math.log(len(r["mu"])) if len(r["mu"]) > 1 else 0 for r in rows])),
                           legal_action_rows=sum(len(r["mu"]) for r in rows),
                           acting_value_std=float(np.std([r["vhat"] for r in rows])),
                           target_std=float(np.std([r["target"] for r in rows])),
                           nonzero_return_fraction=float(np.mean([abs(r["target"]) > 1e-8 for r in rows])),
                           terminal_fraction=sum(e["winner"] >= 0 for e in episodes)/len(episodes),
                           critic_targets_informative=any(abs(r["target"]) > 1e-8 for r in rows),
                           seconds=time.perf_counter()-started,
                           gpu_memory_peak_mb=torch.cuda.max_memory_allocated()/2**20 if args.device == "cuda" else 0,
                           corpus_manifest_sha256=digest(corpus/"manifest.json"), actor_sha256=corpus_identity["actor_sha256"],
                           ratings="unrated; external paired evaluation required")
            latest = run/"checkpoints"/f"{number:04d}"
            publish(latest, identity, checkpoint, metrics)
            status("finished", **metrics)
            print(json.dumps(metrics), flush=True)
    finally:
        lock.unlink()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--initial-model", help="Optional existing NNUE model.pt; Q starts at zero")
    parser.add_argument("--initial-q", help="Optional q.pt bound to the exact --initial-model export; no implicit value-to-Q conversion")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=128)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--cells", type=int, default=65536)
    parser.add_argument("--centers", type=int, default=65536)
    parser.add_argument("--alpha", type=float, default=.03)
    parser.add_argument("--beta", type=float, default=.1)
    parser.add_argument("--lambda-return", type=float, default=math.exp(-1/16))
    parser.add_argument("--gamma", type=float, default=1.)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if any(getattr(args, key) < 1 for key in ("iterations", "games", "envs", "max_plies", "batch", "cells", "centers")):
        parser.error("Counts and memory budgets must be positive")
    if not all(math.isfinite(getattr(args, key)) for key in ("alpha", "beta", "lambda_return", "gamma", "lr")):
        parser.error("Hyperparameters must be finite")
    if args.alpha < 0 or args.beta < 0 or args.alpha+args.beta <= 0 or not 0 <= args.lambda_return <= 1 or not 0 < args.gamma <= 1 or args.lr <= 0:
        parser.error("Invalid learning hyperparameters")
    return args


if __name__ == "__main__":
    main(parse_args())
