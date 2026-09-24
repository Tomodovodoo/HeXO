"""Optional adapter for an external Orca source checkout and its published weights."""
import hashlib
from pathlib import Path
import subprocess
import sys

from hexo import Game


class Orca:
    def __init__(self, source, checkpoint=None, simulations=200, device="cpu", max_stones=800):
        source = Path(source).resolve()
        checkpoint = Path(checkpoint).resolve() if checkpoint else source / "orca/checkpoint.pt"
        if not (source / "hexbot.py").is_file() or not checkpoint.is_file():
            raise ValueError("Orca requires its source checkout and a checkpoint file")
        sys.path.insert(0, str(source))
        import torch
        from hexbot import Bot
        from main import HexGame
        from orca.network import HexNet
        torch.set_num_threads(2)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        weights = saved.get("model_state_dict", saved)
        filters, channels, _, _ = weights["conv_init.weight"].shape
        if channels != 7:
            raise ValueError("This adapter requires an unmodified seven-channel Orca checkpoint")
        blocks = 0
        while f"res_blocks.{blocks}.conv1.weight" in weights:
            blocks += 1
        net = HexNet(num_filters=filters, num_res_blocks=blocks)
        net.load_state_dict(weights, strict=True)
        net.to(device).eval()
        # Construct with the requested budget. Upstream Orca.load changes _sims
        # after creating its searcher, which does not change that searcher's budget.
        self.bot = Bot(net=net, sims=simulations, temperature=0)
        self.game_type = HexGame
        self.max_stones = max_stones
        self.metadata = {
            "name": "Orca", "source": str(source),
            "revision": subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()),
            "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "checkpoint_iteration": saved.get("iteration"), "simulations_per_placement": simulations,
            "filters": filters, "residual_blocks": blocks,
            "parameters": sum(p.numel() for p in net.parameters()),
            "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "device": device, "candidate_radius": 3, "max_stones": max_stones,
            "budget": "MCTS simulations per placement; not equal wall-clock time",
        }

    def __call__(self, game, ms):
        history = [(q, r) for q, r, _ in game.cells]
        opponent = self.game_type(candidate_radius=3, max_total_stones=self.max_stones)
        for q, r in history:
            opponent.place_stone(q, r)
        remaining = opponent.stones_per_turn - opponent.stones_this_turn
        winner = -1 if opponent.winner is None else opponent.winner
        if (opponent.current_player, remaining, winner) != (game.player, game.remaining, game.winner):
            raise ValueError("Orca replay disagrees with the native rules state")
        scratch = Game(history)
        moves = []
        try:
            for _ in range(game.remaining):
                move = self.bot.best_move(opponent)
                scratch.play(*move)
                moves.append(move)
                opponent.place_stone(*move)
                if scratch.winner >= 0:
                    break
            return moves
        finally:
            scratch.close()
