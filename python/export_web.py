"""Export a HexNet checkpoint to ONNX for the browser engine (web/engine).

The graph input is `features` [B, 20, S, S] (S any hexcrop bucket): the 8 hexcrop planes, HexNet's 10 line
features and 2 zero channels, all multiplied by the crop mask (plane 3); web/engine/encode.mjs builds it in the
browser, `inputs` here. The zero channels keep the stem's input a multiple of four channels, since ONNX Runtime
Web's WebGPU convolution miscomputes others. WebNet then computes HexNet's reference inference path with
exportable operators: LineConv's three line convolutions become one depthwise 11x11 convolution whose kernel
holds the taps on the three axes, and act(y, ceiling) becomes relu(y)*mask. Outputs: policy [B, S*S], far [B]
and value [B] (the side to move's win logit), float32. The fp16 graph keeps pooling sums in float32.

`python python/export_web.py CHECKPOINT OUT` writes OUT/bubble-fp32.onnx, OUT/bubble-fp16.onnx and
OUT/manifest.json, which records the source digest and the parity of both graphs against PyTorch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import hexcrop
import hexnet

OPSET = 17
CHANNELS = 20
ROOT = Path(__file__).resolve().parents[1]


def inputs(model, planes):
    """The graph input for float planes [B, 8, S, S], from HexNet's own line features."""
    mask = planes[:, 3:4]
    features = torch.cat((planes, model.lines(planes[:, :1], planes[:, 1:2], mask)), 1)*mask
    return F.pad(features, (0, 0, 0, 0, 0, CHANNELS-features.shape[1]))


def _line_kernel(weight):
    """[C, 3, L] LineConv taps -> [C, 1, L, L] depthwise kernel with the same cross-correlation."""
    c, _, length = weight.shape
    kernel, centre = torch.zeros(c, 1, length, length), length//2
    for a, (dx, dy) in enumerate(hexnet.AXES):
        for i in range(length):
            kernel[:, 0, centre+(i-centre)*dy, centre+(i-centre)*dx] += weight[:, a, i]
    return kernel


class WebNet(nn.Module):
    """HexNet inference (aux heads off) over `inputs`, in exportable operators; see the module docstring."""

    def __init__(self, model):
        super().__init__()
        model = model.cpu().float().eval()
        tensors = dict(stem=F.pad(model.stem.weight*model.stem.hex, (0, 0, 0, 0, 0, CHANNELS-hexnet.FEATURES)))
        norms = [(f'b{i}_{n}', getattr(block, n)) for i, block in enumerate(model.blocks) for n in ('norm1', 'norm2')]
        for name, norm in norms+[('norm', model.norm)]:
            tensors.update({f'{name}_mean': norm.running_mean, f'{name}_var': norm.running_var,
                            f'{name}_weight': norm.weight, f'{name}_bias': norm.bias})
        self.eps = model.norm.eps
        self.blocks = []
        for i, block in enumerate(model.blocks):
            tensors.update({f'b{i}_conv1': block.conv1.weight*block.conv1.hex, f'b{i}_conv2': block.conv2.weight*block.conv2.hex})
            if block.line is not None:
                tensors[f'b{i}_line'] = _line_kernel(block.line.weight)
            if block.pool is not None:
                tensors.update({f'b{i}_pool_w': block.pool.weight, f'b{i}_pool_b': block.pool.bias})
            self.blocks.append((i, block.line is not None, block.pool is not None))
        for name in ('policy_hidden', 'policy', 'far', 'value_hidden', 'value'):
            layer = getattr(model, name)
            tensors.update({f'{name}_w': layer.weight, f'{name}_b': layer.bias})
        for name, tensor in tensors.items():
            self.register_buffer(name, tensor.detach().clone())

    def t(self, name):
        return getattr(self, name)

    def act(self, x, norm, mask):
        """relu(batch norm `norm`(x)) on the crop, 0 on padding."""
        t = self.t
        return F.relu(F.batch_norm(x, t(f'{norm}_mean'), t(f'{norm}_var'), t(f'{norm}_weight'), t(f'{norm}_bias'),
                                   False, 0., self.eps))*mask

    @staticmethod
    def pool(x, scale):
        """[B, 2C] masked mean (scale = 1/crop cells, float32 [B, 1, 1, 1]) and max of non-negative, masked x."""
        return torch.cat(((x.float()*scale).sum((2, 3)).to(x.dtype), x.amax((2, 3))), 1)

    def forward(self, features):
        t, mask = self.t, features[:, 3:4]
        scale = 1/mask.float().sum((2, 3), keepdim=True)
        x = F.conv2d(features, self.stem, padding=1)
        for i, line, pooled in self.blocks:
            y = F.conv2d(self.act(x, f'b{i}_norm1', mask), t(f'b{i}_conv1'), padding=1)
            if pooled:
                y = y+F.linear(self.pool(F.relu(y)*mask, scale), t(f'b{i}_pool_w'), t(f'b{i}_pool_b'))[:, :, None, None]
            if line:
                y = y*mask
                kernel = t(f'b{i}_line')
                y = y+F.conv2d(y, kernel, padding=kernel.shape[-1]//2, groups=kernel.shape[0])
            x = x+F.conv2d(self.act(y, f'b{i}_norm2', mask), t(f'b{i}_conv2'), padding=1)
        x = self.act(x, 'norm', mask)
        pooled = self.pool(x, scale)
        hidden = F.relu(F.conv2d(x, self.policy_hidden_w, self.policy_hidden_b))
        value = F.relu(F.linear(pooled, self.value_hidden_w, self.value_hidden_b))
        policy = F.conv2d(hidden, self.policy_w, self.policy_b).flatten(1)
        far = F.linear(pooled, self.far_w, self.far_b)[:, 0]
        value = F.linear(value, self.value_w, self.value_b)[:, 0]
        return policy.float(), far.float(), value.float()


def histories(path=ROOT/'tests'/'fixtures'/'actor_selfplay.npz', every=7):
    """Positions from the recorded games of `path` (every `every`th ply) plus a far-mode chain of stones."""
    data = np.load(path)
    found, start = [], 0
    for length in data['lengths']:
        game = data['moves'][start:start+length].tolist()
        start += length
        found += [game[:n] for n in range(1, len(game), every)]
    return found+[[[7*i, 0] for i in range(36)]]


def load(path):
    return hexnet.load_model(path, 'cpu', future_target='legacy').float().eval()


def export_onnx(model, path, half):
    net = WebNet(model)
    with torch.inference_mode():
        example = inputs(model, torch.from_numpy(hexcrop.encode([[0, 0], [1, 0], [0, 1]]).planes[None]).float())
    if half:
        net, example = net.half(), example.half()
    torch.onnx.export(net, (example,), str(path), input_names=['features'], output_names=['policy', 'far', 'value'],
                      dynamic_axes=dict(features={0: 'batch', 2: 'size', 3: 'size'}, policy={0: 'batch', 1: 'cells'},
                                        far={0: 'batch'}, value={0: 'batch'}),
                      opset_version=OPSET, dynamo=False, do_constant_folding=True)


def parity(model, path, positions, half=False):
    """Max abs differences {policy, far, value} of the ONNX graph at `path` under ONNX Runtime's CPU provider
    against PyTorch's reference path over `positions`; policy over in-crop legal cells."""
    import onnxruntime
    session = onnxruntime.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    samples = [hexcrop.encode(h) for h in positions]
    worst = dict(policy=0., far=0., value=0.)
    for size, indices in hexcrop.group_by_size(samples).items():
        with torch.inference_mode():
            planes = torch.from_numpy(np.stack([samples[i].planes for i in indices])).float()
            out = model(planes, planes[:, 3:4], aux=False)
            features = inputs(model, planes).numpy()
        policy, far, value = session.run(None, dict(features=features.astype(np.float16 if half else np.float32)))
        legal = np.stack([samples[i].planes[2].reshape(-1) > 0 for i in indices])
        worst['policy'] = max(worst['policy'], float(np.abs(policy-out['policy'].numpy())[legal].max()))
        worst['far'] = max(worst['far'], float(np.abs(far-out['far'].numpy()).max()))
        worst['value'] = max(worst['value'], float(np.abs(value-out['value_logit'].numpy()).max()))
    return worst


def export(checkpoint, out):
    """Write both graphs and the manifest for `checkpoint` into `out`; returns the manifest."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model = load(checkpoint)
    files = {}
    for name, half in (('bubble-fp32.onnx', False), ('bubble-fp16.onnx', True)):
        export_onnx(model, out/name, half)
        files[name] = dict(sha256=hashlib.sha256((out/name).read_bytes()).hexdigest(), bytes=(out/name).stat().st_size)
    positions = histories()
    manifest = dict(schema='bubble-web-v1', source_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                    model_version=hexnet.model_digest(model), opset=OPSET, channels=CHANNELS, files=files,
                    parity=dict(positions=len(positions), fp32=parity(model, out/'bubble-fp32.onnx', positions),
                                fp16=parity(model, out/'bubble-fp16.onnx', positions, half=True)))
    (out/'manifest.json').write_text(json.dumps(manifest, indent=1)+'\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.out), indent=1))


if __name__ == '__main__':
    main()
