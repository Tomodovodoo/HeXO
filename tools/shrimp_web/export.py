"""Export Shrimp's network (Cmiller132/hexo-bot main_7 epoch 18, MIT) to ONNX for the browser engine.

The graph computes `ShrimpNet.forward_policy_value` with the moves-left head and the evaluator's decode
(`shrimp.inference.ShrimpEvaluator` on CPU) for B support rows padded to N nodes:

  feats   [B, N, 15] float32  the featurizer's node features, rounded through float16 as the driver sends them
  index   [B, N, 7]  int32    rows of the flattened [B*N + 1, C] activations each node's hex convolution gathers:
                              itself, then its six neighbours; B*N (a zero row) where a neighbour is missing
  mask    [B, N]     float32  1 at real nodes, 0 at padding
  pair    [B, S, S]  int32    with S = 8 + N: the bias-table row of every (query, key) of [8 tokens; cells], as
                              ShrimpNet._build_pair_u8 builds it (padded keys on the extra row)
  ->
  policy  [B, N]  raw logits (legal cells first in each row)
  value   [B]     the binned value's expectation, clamped to [-1, 1]
  moves_left [B]  decisions left, in [0, 209]

Each attention block gathers its bias from its table rounded to float16 (as the evaluator's no-grad path does), with
a row of -30000 for padded keys. `python tools/shrimp_web/export.py WEIGHTS OUT` writes OUT/shrimp-fp32.onnx and
OUT/manifest.json: the weights' digest, the search profile the driver runs (from the pinned shrimp_main_7.toml) and
the graph's parity with PyTorch.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tomllib
import types
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OPSET = 17
HERE = Path(__file__).resolve().parent
VENDOR = HERE/'vendor'/'hexo-bot'
PROFILE = VENDOR/'apps'/'showcase'/'profiles'/'shrimp_main_7.toml'
WEIGHTS = dict(url='https://media.githubusercontent.com/media/Cmiller132/hexo-bot/6251fc6d9072cb6b26ae6830e3c28d0c0666f502/'
                   'models/shrimp_main7_infer.pt',
               sha256='680eebd4381ad59511674b21d80ae8222cc1338ba4de3301eb2942ba6b96304f', size=32585249)
# The driver's architecture environment (Six's arena/drivers/shrimp_driver.py ARCH_ENV): read by hexo-bot at import.
ARCH = dict(SHRIMP_CHANNELS='192', SHRIMP_ATTENTION_HEADS='3', SHRIMP_TRUNK='CCACCACCACCACCA', SHRIMP_SUPPORT_RADIUS='4')
SUPPORT_RADIUS = 4
CACHE_STATES = 65536   # the driver's ShrimpMctsSession(max_states=65_536)
PAD_KEY = -3.0e4


def shrimp():
    """hexo-bot's `shrimp` Python package from the vendored files, without its __init__ (which needs the featurizer)."""
    if 'shrimp.model' not in sys.modules:
        for key, value in ARCH.items():
            if os.environ.setdefault(key, value) != value:
                raise RuntimeError(f'{key} is {os.environ[key]}, the weights need {value}')
        package = types.ModuleType('shrimp')
        package.__path__ = [str(VENDOR/'packages'/'shrimp'/'python'/'shrimp')]
        sys.modules['shrimp'] = package
    return types.SimpleNamespace(**{name: importlib.import_module(f'shrimp.{name}')
                                    for name in ('model', 'losses', 'config', 'constants', 'geometry')})


def load(path):
    """ShrimpNet with the weights at `path`, as the driver loads them."""
    s = shrimp()
    state = torch.load(path, map_location='cpu', weights_only=False)['model']
    model = s.model.ShrimpNet(**s.model.infer_net_kwargs_from_state_dict(state))
    model.load_state_dict(state, strict=True)
    return model.eval()


def search_profile():
    """The keyword arguments the driver passes to `ShrimpMctsSession.search`, from the pinned profile: its search
    settings, its divergence overrides and the evaluator's virtual batch."""
    s = shrimp()
    config = tomllib.loads(PROFILE.read_text(encoding='utf-8'))['model']['config']
    parsed = s.config.parse_shrimp_config({'device': 'cpu', 'selfplay': config['selfplay'],
                                           'multi_stage_eval': config['multi_stage_eval']})
    sp = parsed.selfplay
    settings = dict(c_puct=sp.c_puct, virtual_batch_size=int(parsed.multi_stage_eval.eval_virtual_batch_size or 32),
                    active_root_limit=sp.active_root_limit, widening_policy_mass=sp.widening_policy_mass,
                    widening_max_children=sp.widening_max_children, widening_min_children=sp.widening_min_children,
                    fpu_reduction=sp.fpu_reduction, tss_enabled=sp.tss_enabled)
    settings.update(s.config.build_divergence_overrides(sp))
    return dict(search_parity_mode=bool(sp.search_parity_mode), cache_states=CACHE_STATES,
                support_radius=SUPPORT_RADIUS, settings={k: float(v) for k, v in settings.items()})


class WebShrimp(nn.Module):
    """ShrimpNet's serve forward over the inputs in the module docstring, in operators ONNX Runtime Web runs."""

    def __init__(self, model):
        super().__init__()
        s = shrimp()
        c = s.constants
        self.net = model
        self.layout = model._trunk_layout
        self.tokens_count = c.NUM_TOKENS
        self.register_buffer('bins', s.losses.value_bins())
        for width in {conv.in_channels for conv in model.modules() if isinstance(conv, s.model.HexNodeConv)}:
            self.register_buffer(f'zero{width}', torch.zeros(1, width))
        self.moves_left_cap = c.MOVES_LEFT_CAP
        for i, table in enumerate(model.bias_tables):
            pad = torch.full((1, table.shape[1]), PAD_KEY)
            self.register_buffer(f'bias{i}', torch.cat([table.detach().half().float(), pad]))

    def conv(self, layer, x, index, mask):
        """HexNodeConv: the 7 taps of every node gathered from the flattened activations plus a zero row, one GEMM."""
        b, n = x.shape[0], x.shape[1]
        width = layer.in_channels
        flat = torch.cat([x.reshape(b*n, width), getattr(self, f'zero{width}')])
        gathered = flat[index].reshape(b, n, 7*width)
        return (gathered @ layer.weight.reshape(7*width, layer.out_channels) + layer.bias)*mask.unsqueeze(-1)

    def attention(self, block, seq, bias):
        a = block.attn
        b, s, ch = seq.shape[0], seq.shape[1], a.out_proj.in_features
        h, d = a.heads, a.head_dim
        q = a.q_proj(seq).reshape(b, s, h, d).transpose(1, 2)
        k = a.k_proj(seq).reshape(b, s, h, d).transpose(1, 2)
        v = a.v_proj(seq).reshape(b, s, h, d).transpose(1, 2)
        weights = torch.softmax(q @ k.transpose(-2, -1)*a.scale + bias, dim=-1)
        return a.out_proj((weights @ v).transpose(1, 2).reshape(b, s, ch))

    def forward(self, feats, index, mask, pair):
        net, m = self.net, mask.unsqueeze(-1)
        x = F.relu(net.stem_ln(self.conv(net.stem, feats, index, mask)))*m
        seq_mask = F.pad(mask, (self.tokens_count, 0), value=1.).unsqueeze(-1)
        tokens = net.tokens.unsqueeze(0).expand(feats.shape[0], -1, -1)
        ci = ai = 0
        seq = None
        for position, kind in enumerate(self.layout):
            if kind == 'C':
                block = net.conv_blocks[ci]
                y = F.relu(block.ln1(self.conv(block.conv1, x, index, mask)))*m
                y = block.ln2(self.conv(block.conv2, y, index, mask))*m
                x = F.relu(x + block.ls(y))
                ci += 1
                continue
            block = net.attn_blocks[ai]
            bias = getattr(self, f'bias{ai}')[pair].permute(0, 3, 1, 2)
            seq = torch.cat([tokens, x], 1)
            seq = seq + block.ls_attn(self.attention(block, block.ln1(seq), bias)*seq_mask)
            seq = seq + block.ls_mlp(block.fc2(F.gelu(block.fc1(block.ln2(seq))))*seq_mask)
            ai += 1
            if position != len(self.layout) - 1:
                tokens, x = seq[:, :self.tokens_count], seq[:, self.tokens_count:]
        seq = net.ln_final(seq)
        tokens, cells = seq[:, :self.tokens_count], seq[:, self.tokens_count:]*m
        pooled = (cells*m).sum(1)/mask.sum(1, keepdim=True).clamp(min=1)
        policy = net.policy_head(F.relu(self.conv(net.policy_conv, cells, index, mask))).squeeze(-1)*mask
        value = net.value_head(F.relu(net.value_reduction(torch.cat([tokens[:, 0], tokens[:, 1], pooled], 1))))
        left = net.moves_left_head(F.relu(net.ml_reduction(torch.cat([tokens[:, 4], tokens[:, 5], pooled], 1))))
        expect = lambda logits: (torch.softmax(logits, -1)*self.bins).sum(-1).clamp(-1, 1)
        return policy, expect(value), (expect(left) + 1)*.5*self.moves_left_cap


def bias_layout():
    """What the browser needs to build `pair`: the cell-offset table and its span, the token rows and the pad row."""
    s = shrimp()
    c, m = s.constants, s.constants.BIAS_RING_MAX + 1
    lut = [s.geometry.rel_bias_index(dq, dr) for dq in range(-m, m + 1) for dr in range(-m, m + 1)]
    return dict(span=m, lut=lut, tokens=c.NUM_TOKENS, token_token=c.BIAS_TOKEN_TOKEN_ROW,
                token_cell=c.BIAS_TOKEN_CELL_ROW, cell_token=c.BIAS_CELL_TOKEN_ROW, pad=c.BIAS_ROWS)


def pack(rows):
    """Graph inputs (numpy) for `rows`, each (feats [n, 15], coords [n, 2], nbr [n, 6] with -1 for a missing neighbour),
    padded to the longest row: what web/engine/shrimp/network.mjs builds. `pair` is each block's bias-table row for
    every (query, key) of [tokens; cells]: ShrimpNet._build_pair_u8 with padded keys on the pad row."""
    layout = bias_layout()
    lut, m, t = np.array(layout['lut'], np.int32), layout['span'], layout['tokens']
    b, n = len(rows), max(len(r[0]) for r in rows)
    feats, index = np.zeros((b, n, 15), np.float32), np.full((b, n, 7), b*n, np.int32)
    mask, pair = np.zeros((b, n), np.float32), np.full((b, t + n, t + n), layout['pad'], np.int32)
    for i, (f, c, nbr) in enumerate(rows):
        k = len(f)
        feats[i, :k], mask[i, :k] = f, 1
        index[i, :, 0] = i*n + np.arange(n)
        index[i, :k, 1:] = np.where(nbr < 0, b*n, i*n + nbr)
        coords = np.zeros((n, 2), np.int32)
        coords[:k] = c
        offset = np.clip(coords[None, :k] - coords[:, None], -m, m) + m
        pair[i, :t, :t] = layout['token_token']
        pair[i, :t, t:t + k] = layout['token_cell']
        pair[i, t:, :t] = layout['cell_token']
        pair[i, t:, t:t + k] = lut[offset[..., 0]*(2*m + 1) + offset[..., 1]]
    return dict(feats=feats, index=index, mask=mask, pair=pair)


def reference(model, rows):
    """The evaluator's CPU forward for `rows`: ShrimpNet.forward_policy_value on its own padded batch, decoded."""
    s = shrimp()
    b, n = len(rows), max(len(r[0]) for r in rows)
    feats, nbr = torch.zeros(b, n, 15), torch.full((b, n, 6), n, dtype=torch.long)
    mask, coords = torch.zeros(b, n, dtype=torch.bool), torch.zeros(b, n, 2, dtype=torch.long)
    for i, (f, c, row_nbr) in enumerate(rows):
        k = len(f)
        feats[i, :k], coords[i, :k], mask[i, :k] = torch.from_numpy(f), torch.from_numpy(c).long(), True
        nbr[i, :k] = torch.from_numpy(np.where(row_nbr < 0, n, row_nbr)).long()
    with torch.no_grad():
        out = model.forward_policy_value(feats, nbr, mask, coords, request_moves_left=True)
        return (out['policy'].numpy(), s.losses.decode_binned_value(out['value']).numpy(),
                s.losses.decode_moves_left(out['moves_left']).numpy())


DIRECTIONS = ((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1))


def synthetic_rows(seed=0, count=24):
    """Rows shaped like the featurizer's: hex supports of radius 4 around scattered stones plus their halo, with
    random 0/1 features and distances in eighths, in [legal | stones | halo] order."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(count):
        stones = {(0, 0)}
        while len(stones) < rng.integers(1, 24):
            q, r = list(stones)[rng.integers(len(stones))]
            dq, dr = DIRECTIONS[rng.integers(6)]
            stones.add((q + dq*int(rng.integers(1, 3)), r + dr*int(rng.integers(1, 3))))
        near = {(q + dq, r + dr) for q, r in stones for dq in range(-4, 5) for dr in range(-4, 5) if max(abs(dq), abs(dr), abs(dq + dr)) <= 4}
        legal = sorted(near - stones)
        halo = sorted({(q + dq, r + dr) for q, r in near for dq, dr in DIRECTIONS} - near)
        cells = legal + sorted(stones) + halo
        where = {cell: i for i, cell in enumerate(cells)}
        nbr = np.array([[where.get((q + dq, r + dr), -1) for dq, dr in DIRECTIONS] for q, r in cells], np.int32)
        feats = (rng.random((len(cells), 15)) < .3).astype(np.float32)
        feats[:, 11] = rng.integers(0, 10, len(cells))/8
        rows.append((feats.astype(np.float16).astype(np.float32), np.array(cells, np.int32), nbr))
    return rows


def export_onnx(model, path):
    web = WebShrimp(model).eval()
    with torch.no_grad():
        example = {k: torch.from_numpy(v) for k, v in pack(synthetic_rows(1, 2)).items()}
        torch.onnx.export(web, tuple(example.values()), str(path), input_names=list(example),
                          output_names=['policy', 'value', 'moves_left'],
                          dynamic_axes=dict(feats={0: 'batch', 1: 'nodes'}, index={0: 'batch', 1: 'nodes'},
                                            mask={0: 'batch', 1: 'nodes'}, pair={0: 'batch', 1: 'sequence', 2: 'sequence'},
                                            policy={0: 'batch', 1: 'nodes'}, value={0: 'batch'}, moves_left={0: 'batch'}),
                          opset_version=OPSET, dynamo=False, do_constant_folding=True)


def parity(model, path, rows, batch=8):
    """Max abs differences {policy (over real nodes), value, moves_left} of the graph at `path` under ONNX Runtime's
    CPU provider against the evaluator's PyTorch forward, over `rows` in batches of `batch`."""
    import onnxruntime
    session = onnxruntime.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    worst = dict(policy=0., value=0., moves_left=0.)
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        got = session.run(None, pack(chunk))
        want = reference(model, chunk)
        real = np.zeros(want[0].shape, bool)
        for i, row in enumerate(chunk):
            real[i, :len(row[0])] = True
        worst['policy'] = max(worst['policy'], float(np.abs(got[0] - want[0])[real].max()))
        worst['value'] = max(worst['value'], float(np.abs(got[1] - want[1]).max()))
        worst['moves_left'] = max(worst['moves_left'], float(np.abs(got[2] - want[2]).max()))
    return worst


def export(weights, out):
    """Write the graph and the manifest for the weights at `weights` into `out`; returns the manifest."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    source = hashlib.sha256(Path(weights).read_bytes()).hexdigest()
    if source != WEIGHTS['sha256']:
        raise ValueError(f'{weights} is not the pinned main_7 weights ({source})')
    model = load(weights)
    name = 'shrimp-fp32.onnx'
    export_onnx(model, out/name)
    rows = synthetic_rows()
    manifest = dict(schema='shrimp-web-v1', source_sha256=source, model_version=f'hexo-bot@6251fc6 main_7 ep18 {source[:16]}',
                    opset=OPSET, bias=bias_layout(), files={name: dict(sha256=hashlib.sha256((out/name).read_bytes()).hexdigest(),
                                                   bytes=(out/name).stat().st_size)},
                    search=search_profile(), parity=dict(rows=len(rows), fp32=parity(model, out/name, rows)))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('weights', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.weights, args.out), indent=1))


if __name__ == '__main__':
    main()
