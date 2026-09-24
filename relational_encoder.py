"""Exact sparse relational encoding. Coordinates never pass through a board crop.

Window occupancy and incidence positions are tied under reversal; relation
features use only hex distances and identity, so all twelve board symmetries
permute the graph without changing its features.
"""
from dataclasses import dataclass
import math
import hashlib
import numpy as np
from hexo import Game

AXES = ((1, 0), (0, 1), (1, -1))
NEIGHBORS = ((1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1))
BALL8 = tuple((q,r) for q in range(-8,9) for r in range(max(-8,-q-8), min(8,-q+8)+1) if q or r)


def distance(a, b):
    q, r = a[0]-b[0], a[1]-b[1]
    return max(abs(q), abs(r), abs(q+r))


def transform(point, symmetry):
    if not 0 <= symmetry < 12:
        raise ValueError('Symmetry must be in 0..11')
    q, r = point
    if symmetry >= 6:
        q, r = r, q
    for _ in range(symmetry % 6):
        q, r = -r, q+r
    return q, r


class WorkBudgetError(ValueError):
    pass


@dataclass
class Graph:
    actions: np.ndarray
    kinds: np.ndarray
    owners: np.ndarray
    patterns: np.ndarray
    features: np.ndarray
    local_edges: np.ndarray
    stone_edges: np.ndarray
    global_read_edges: np.ndarray
    global_write_edges: np.ndarray
    stone_coords: np.ndarray
    window_cells: np.ndarray
    player: int
    remaining: int
    global_tokens: int

    @property
    def position_key(self):
        # Absolute ownership and native phase identify the state, independently
        # of which chronological ordering reached it. Coordinates remain int64.
        n = len(self.stone_coords)
        occupied = np.empty((n,3),dtype='<i8')
        occupied[:,:2] = self.stone_coords
        occupied[:,2] = np.where(self.owners[:n] == 0,self.player,1-self.player)
        phase = np.asarray([self.player,self.remaining],dtype='<i8')
        return hashlib.sha256(b'hexo-six-radius8-v1\0'+phase.tobytes()+occupied.tobytes()).hexdigest()

    @property
    def node_count(self):
        return len(self.kinds)

    @property
    def edge_count(self):
        # Local relations are traversed before and after global exchange.
        return len(self.local_edges)+sum(len(getattr(self, name)) for name in EDGE_NAMES)


EDGE_NAMES = ('local_edges', 'stone_edges', 'global_read_edges', 'global_write_edges')


def _edges(rows):
    return np.asarray(rows, dtype=np.int64).reshape(-1, 5)


def encode(history, *, global_tokens=16, max_nodes=None, max_edges=None):
    if type(global_tokens) is not int or not 1 <= global_tokens <= 1_000_000:
        raise ValueError('Global token count must be in 1..1000000')
    game = Game(history)
    try:
        if game.winner >= 0:
            raise ValueError('Terminal positions are handled by exact search, not the network')
        actions = np.asarray(game.legal_moves(), dtype=np.int64).reshape(-1, 2)
        cells = {(q,r):p for q,r,p in game.cells}
        player, remaining = game.player, game.remaining
    finally:
        game.close()
    stones = sorted(cells)
    if max_nodes is not None and len(stones)+len(actions)+global_tokens > max_nodes:
        raise WorkBudgetError('Stone/legal nodes already exceed the node budget')
    window_keys = set()
    for q,r in stones:
        for axis,(dq,dr) in enumerate(AXES):
            for k in range(6):
                window_keys.add((q-k*dq, r-k*dr, axis))
    windows = []
    for q,r,axis in sorted(window_keys):
        dq,dr = AXES[axis]
        windows.append(tuple((q+k*dq, r+k*dr) for k in range(6)))
    ns, nw, na = len(stones), len(windows), len(actions)
    spatial = ns+nw+na
    if max_nodes is not None and spatial+global_tokens > max_nodes:
        raise WorkBudgetError(f'Position requires {spatial+global_tokens} nodes, budget {max_nodes}')
    nonlocal_edges = ns*ns+global_tokens*(2*spatial+global_tokens)
    local = []
    def check_edges():
        if max_edges is not None and nonlocal_edges+2*len(local) > max_edges:
            raise WorkBudgetError(f'Position exceeds {max_edges} relation traversals; increase budget, never crop')
    check_edges()
    kinds = np.concatenate((np.zeros(ns,np.int64), np.ones(nw,np.int64), np.full(na,2,np.int64), np.full(global_tokens,3,np.int64)))
    owners = np.full(spatial+global_tokens,2,np.int64)
    owners[:ns] = [int(cells[c] != player) for c in stones]
    patterns = np.zeros(spatial+global_tokens,np.int64)
    features = np.zeros((spatial+global_tokens,8),np.float32)
    features[:,0] = remaining == 1
    features[:,1] = remaining == 2
    features[:,2] = math.log1p(ns)/8
    features[:,3] = sum(p == player for p in cells.values())/max(1,ns)
    stone_ids = {c:i for i,c in enumerate(stones)}
    action_ids = {tuple(map(int,c)):ns+nw+i for i,c in enumerate(actions)}
    for wi, window in enumerate(windows):
        values = [0 if c not in cells else (1 if cells[c] == player else 2) for c in window]
        code = sum(v*3**i for i,v in enumerate(values))
        reverse = sum(v*3**(5-i) for i,v in enumerate(values))
        node = ns+wi
        patterns[node] = min(code,reverse)
        features[node,4] = values.count(1)/6
        features[node,5] = values.count(2)/6
        for i,c in enumerate(window):
            cell = stone_ids.get(c, action_ids.get(c))
            # At the engine's representational coordinate limit an empty segment
            # endpoint may be unplaceable; do not invent a legal-cell node there.
            if cell is None:
                continue
            slot = min(i,5-i) if code == reverse else (i if code < reverse else 5-i)
            local.extend(((cell,node,0,0,slot), (node,cell,1,0,slot)))
        check_edges()
    for c,stone in stone_ids.items():
        for dq,dr in BALL8:
            point = c[0]+dq,c[1]+dr
            action = action_ids.get(point)
            if action is not None:
                d = max(abs(dq),abs(dr),abs(dq+dr))
                local.extend(((stone,action,2,d,6),(action,stone,3,d,6)))
        check_edges()
    for c,node in action_ids.items():
        for dq,dr in NEIGHBORS:
            other = action_ids.get((c[0]+dq,c[1]+dr))
            if other is not None:
                local.append((other,node,4,1,6))
        check_edges()
    local.extend((i,i,9,0,6) for i in range(spatial))
    check_edges()
    stone_edges = [(a,b,5,distance(ca,cb),6) for a,ca in enumerate(stones) for b,cb in enumerate(stones)]
    globals_ = range(spatial,spatial+global_tokens)
    global_read = [(i,j,6 if i < spatial else 8,0,6) for j in globals_ for i in range(spatial+global_tokens)]
    global_write = [(j,i,7,0,6) for i in range(spatial) for j in globals_]
    return Graph(actions, kinds, owners, patterns, features, _edges(local), _edges(stone_edges),
                 _edges(global_read), _edges(global_write), np.asarray(stones,np.int64).reshape(-1,2),
                 np.asarray(windows,np.int64).reshape(-1,6,2), player, remaining, global_tokens)


def iter_batches(graphs, *, max_nodes=12000, max_edges=600000):
    """Whole-position batches in input order. Oversized positions raise, never crop."""
    if max_nodes < 1 or max_edges < 1:
        raise ValueError('Work budgets must be positive')
    batch, nodes, edges = [], 0, 0
    for graph in graphs:
        if graph.node_count > max_nodes or graph.edge_count > max_edges:
            raise WorkBudgetError(f'Position requires {graph.node_count} nodes/{graph.edge_count} edges; '
                                  f'budgets are {max_nodes}/{max_edges}; increase budgets, never crop actions')
        if batch and (nodes+graph.node_count > max_nodes or edges+graph.edge_count > max_edges):
            yield batch
            batch, nodes, edges = [], 0, 0
        batch.append(graph)
        nodes += graph.node_count
        edges += graph.edge_count
    if batch:
        yield batch


def pack(graphs, device='cpu'):
    import torch
    graphs = list(graphs)
    if not graphs:
        raise ValueError('Cannot pack an empty graph batch')
    if len({g.global_tokens for g in graphs}) != 1:
        raise ValueError('All graphs must use the same global token count')
    offsets = np.cumsum([0]+[g.node_count for g in graphs])
    action_offsets = np.cumsum([0]+[len(g.actions) for g in graphs])
    data = {name:np.concatenate([getattr(g,name) for g in graphs]) for name in ('actions','kinds','owners','patterns','features')}
    for name in EDGE_NAMES:
        shifted = []
        for offset,g in zip(offsets,graphs):
            edge = getattr(g,name).copy()
            edge[:,:2] += offset
            shifted.append(edge)
        data[name] = np.concatenate(shifted)
    data.update(action_offsets=action_offsets, action_owner=np.repeat(np.arange(len(graphs)),np.diff(action_offsets)),
                node_owner=np.repeat(np.arange(len(graphs)),np.diff(offsets)),
                action_nodes=np.flatnonzero(data['kinds'] == 2), stone_nodes=np.flatnonzero(data['kinds'] == 0),
                spatial_nodes=np.flatnonzero(data['kinds'] != 3), global_nodes=np.flatnonzero(data['kinds'] == 3),
                global_index=np.tile(np.arange(graphs[0].global_tokens),len(graphs)))
    result = {k:torch.as_tensor(v,device=device) for k,v in data.items()}
    result['global_tokens'] = graphs[0].global_tokens
    result['position_keys'] = tuple(g.position_key for g in graphs)
    result['players'] = torch.tensor([g.player for g in graphs],dtype=torch.int64,device=device)
    result['remaining'] = torch.tensor([g.remaining for g in graphs],dtype=torch.int64,device=device)
    return result
