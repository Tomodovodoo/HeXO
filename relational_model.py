"""Primary sparse relational policy/Q Transformer; no NNUE export or board crop."""
from dataclasses import dataclass, asdict
import math
import hashlib
import json
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from relational_encoder import encode, iter_batches, pack


@dataclass(frozen=True)
class ModelConfig:
    width: int = 256
    blocks: int = 8
    heads: int = 8
    ff: int = 1024
    global_tokens: int = 16
    edge_chunk: int = 8192
    checkpoint: bool = True

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in (self.width,self.blocks,self.heads,self.ff,self.global_tokens,self.edge_chunk)):
            raise ValueError('Model dimensions must be positive integers')
        if self.width % self.heads:
            raise ValueError('Width must be divisible by attention heads')


class FloatNorm(nn.LayerNorm):
    def forward(self, x):
        return F.layer_norm(x.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps)


class RelationAttention(nn.Module):
    """Exact sparse softmax over every incoming relation, with chunked messages."""
    def __init__(self, config):
        super().__init__()
        self.heads, self.width, self.chunk = config.heads, config.width, config.edge_chunk
        self.qkv = nn.Linear(config.width, 3*config.width)
        self.output = nn.Linear(config.width, config.width)
        self.relation = nn.Embedding(10,config.heads)
        self.slot = nn.Embedding(7,config.heads)
        self.distance = nn.Embedding(9,config.heads)
        self.log_distance = nn.Parameter(torch.zeros(config.heads))

    def forward(self, x, edges):
        n, h, d = len(x), self.heads, self.width//self.heads
        q, k, v = self.qkv(x).reshape(n,3,h,d).unbind(1)
        logits = []
        for start in range(0,len(edges),self.chunk):
            e = edges[start:start+self.chunk]
            src, dst, relation, distance, slot = e.unbind(1)
            # Projection may be mixed precision; products, softmax and sums are FP32.
            score = (q[dst].float()*k[src].float()).sum(-1)/math.sqrt(d)
            bias = self.relation(relation).float()+self.slot(slot).float()+self.distance(distance.clamp_max(8)).float()
            logits.append(score+bias+torch.log1p(distance.float())[:,None]*self.log_distance.float()[None,:])
        scores = torch.cat(logits)
        destinations = edges[:,1,None].expand(-1,h)
        maxima = scores.new_full((n,h),-torch.inf)
        maxima.scatter_reduce_(0,destinations,scores.detach(),reduce='amax',include_self=True)
        mass = (scores-maxima[edges[:,1]]).exp()
        denominator = scores.new_zeros((n,h)).scatter_add_(0,destinations,mass)
        weights = mass/denominator[edges[:,1]]
        aggregated = scores.new_zeros((n,h,d))
        for start in range(0,len(edges),self.chunk):
            e = edges[start:start+self.chunk]
            message = weights[start:start+len(e),:,None]*v[e[:,0]].float()
            aggregated.index_add_(0,e[:,1],message)
        return self.output(aggregated.reshape(n,self.width))


class Stage(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = FloatNorm(config.width)
        self.attention = RelationAttention(config)
        self.ff_norm = FloatNorm(config.width)
        self.ff = nn.Sequential(nn.Linear(config.width,config.ff),nn.GELU(),nn.Linear(config.ff,config.width))

    def forward(self, x, edges, targets):
        if targets.numel() == 0:
            return x
        messages = self.attention(self.norm(x),edges)
        updated = x[targets]+messages[targets].float()
        updated = updated+self.ff(self.ff_norm(updated)).float()
        return x.index_copy(0,targets,updated)


class RelationalBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.local_before = Stage(config)
        self.stones = Stage(config)
        self.global_read = Stage(config)
        self.global_write = Stage(config)
        self.local_after = Stage(config)

    def forward(self, x, batch):
        x = self.local_before(x,batch['local_edges'],batch['spatial_nodes'])
        x = self.stones(x,batch['stone_edges'],batch['stone_nodes'])
        x = self.global_read(x,batch['global_read_edges'],batch['global_nodes'])
        x = self.global_write(x,batch['global_write_edges'],batch['spatial_nodes'])
        return self.local_after(x,batch['local_edges'],batch['spatial_nodes'])


class RelationalNet(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.kind = nn.Embedding(4,c.width)
        self.owner = nn.Embedding(3,c.width)
        self.pattern = nn.Embedding(729,c.width)
        self.numeric = nn.Linear(8,c.width)
        self.globals = nn.Parameter(torch.randn(c.global_tokens,c.width)/math.sqrt(c.width))
        self.blocks = nn.ModuleList(RelationalBlock(c) for _ in range(c.blocks))
        def head():
            return nn.Sequential(FloatNorm(c.width),nn.Linear(c.width,c.width//2 or 1),nn.GELU(),nn.Linear(c.width//2 or 1,1))
        self.policy = head()
        self.critic = head()

    def forward(self, batch):
        if batch['global_tokens'] != self.config.global_tokens:
            raise ValueError('Encoder and model global-token counts differ')
        x = self.kind(batch['kinds'])+self.owner(batch['owners'])+self.pattern(batch['patterns'])+self.numeric(batch['features']).float()
        x = x.index_add(0,batch['global_nodes'],self.globals[batch['global_index']])
        for block in self.blocks:
            if self.config.checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(block,x,batch,use_reentrant=False)
            else:
                x = block(x,batch)
        legal = x[batch['action_nodes']]
        return {'logits': self.policy(legal).squeeze(-1).float(),
                'q': self.critic(legal).squeeze(-1).float().tanh(),
                'action_offsets': batch['action_offsets']}


class NeuralEvaluator:
    def __init__(self, model, device='cuda', *, max_nodes=12000, max_edges=600000, mixed_precision=True, backend='native', model_version=None):
        if backend == 'native':
            from relational_native import encode as encoder
        elif backend == 'reference':
            encoder = encode
        else:
            raise ValueError("Encoder backend must be 'native' or 'reference'")
        self.encoder, self.backend = encoder, backend
        if model_version is None:
            digest = hashlib.sha256(json.dumps(asdict(model.config),sort_keys=True).encode())
            for name,tensor in model.state_dict().items():
                digest.update(name.encode()+str(tensor.dtype).encode()+str(tuple(tensor.shape)).encode())
                digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            model_version = digest.hexdigest()
        if not isinstance(model_version,str) or not model_version:
            raise ValueError('Model version must be a nonempty immutable identity')
        self.model_version = model_version
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.max_nodes, self.max_edges = max_nodes, max_edges
        self.mixed_precision = mixed_precision

    @torch.inference_mode()
    def evaluate(self, histories):
        result = []
        graphs = (self.encoder(h,global_tokens=self.model.config.global_tokens,max_nodes=self.max_nodes,max_edges=self.max_edges) for h in histories)
        for group in iter_batches(graphs,max_nodes=self.max_nodes,max_edges=self.max_edges):
            batch = pack(group,self.device)
            context = (torch.autocast('cuda',dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
                       if self.device.type == 'cuda' and self.mixed_precision else nullcontext())
            with context:
                prediction = self.model(batch)
            logits, q = prediction['logits'].cpu().numpy(), prediction['q'].cpu().numpy()
            if not np.isfinite(logits).all() or not np.isfinite(q).all():
                raise FloatingPointError('Nonfinite relational model predictions')
            offset = 0
            for graph in group:
                end = offset+len(graph.actions)
                result.append({'actions':graph.actions.copy(),'logits':logits[offset:end].copy(),'q':q[offset:end].copy(),
                               'position_key':graph.position_key,'player':graph.player,'remaining':graph.remaining,
                               'model_version':self.model_version})
                offset = end
        return result
