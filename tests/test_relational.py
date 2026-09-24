import copy
import unittest
import numpy as np
import torch
from hexo import Game
from relational_encoder import encode, pack, iter_batches, transform, distance, WorkBudgetError, EDGE_NAMES
from relational_model import RelationalNet, ModelConfig, NeuralEvaluator


def identities(graph, symmetry=0):
    result = [('stone',transform(tuple(c),symmetry)) for c in graph.stone_coords]
    result += [('window',tuple(sorted(transform(tuple(c),symmetry) for c in w))) for w in graph.window_cells]
    result += [('action',transform(tuple(c),symmetry)) for c in graph.actions]
    result += [('global',i) for i in range(graph.global_tokens)]
    return result


class RelationalTests(unittest.TestCase):
    def setUp(self):
        self.threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads,self.threads)
        torch.manual_seed(214)
        self.config = ModelConfig(width=32,blocks=1,heads=4,ff=64,global_tokens=2,edge_chunk=1024)
        self.history = [(0,0),(8,0),(16,0),(7,1),(-1,-1)]

    def test_native_encoder_exact_parity_and_budgets(self):
        from relational_native import encode as native_encode
        for length in (0,1,2,5):
            for symmetry in range(12):
                history = [transform(c,symmetry) for c in self.history[:length]]
                reference = encode(history,global_tokens=2)
                native = native_encode(history,global_tokens=2)
                for name in reference.__dataclass_fields__:
                    left, right = getattr(reference,name), getattr(native,name)
                    if isinstance(left,np.ndarray):
                        self.assertEqual(left.dtype,right.dtype)
                        self.assertEqual(left.shape,right.shape)
                        self.assertEqual(left.tobytes(),right.tobytes(),name)
                    else:
                        self.assertEqual(left,right)
                self.assertEqual(reference.position_key,native.position_key)
                for kwargs in ({'max_nodes':native.node_count-1},{'max_edges':native.edge_count-1}):
                    with self.assertRaises(WorkBudgetError):
                        native_encode(history,global_tokens=2,**kwargs)
                accepted = native_encode(history,global_tokens=2,max_nodes=native.node_count,max_edges=native.edge_count)
                self.assertEqual(len(accepted.actions),len(reference.actions))

    def test_exact_actions_phase_incidence_and_uncovered_geometry(self):
        for length in range(len(self.history)+1):
            history = self.history[:length]
            g = encode(history,global_tokens=2)
            native = Game(history)
            try:
                np.testing.assert_array_equal(g.actions,np.asarray(native.legal_moves()).reshape(-1,2))
                self.assertEqual((g.player,g.remaining),(native.player,native.remaining))
            finally:
                native.close()
            self.assertLessEqual(len(g.window_cells),18*length)
            ids = identities(g)
            self.assertEqual(len(set(ids)),g.node_count)
            for src,dst,relation,d,slot in g.local_edges:
                if relation == 0:
                    self.assertIn(ids[src][1],ids[dst][1])
                if relation == 2:
                    self.assertEqual(distance(ids[src][1],ids[dst][1]),d)
                    self.assertLessEqual(d,8)
            if length:
                reached = set(g.local_edges[g.local_edges[:,2] == 2,1])
                self.assertEqual(reached,set(np.flatnonzero(g.kinds == 2)))
        g = encode([(0,0)],global_tokens=2)
        far = identities(g).index(('action',(7,1)))
        corner = identities(g).index(('action',(8,0)))
        for node in (far,corner):
            self.assertFalse(any(dst == node and rel == 1 for _,dst,rel,_,_ in g.local_edges))
            self.assertTrue(any(dst == node and rel == 2 and d == 8 for _,dst,rel,d,_ in g.local_edges))
        degrees = [sum(dst == node and rel == 4 for _,dst,rel,_,_ in g.local_edges) for node in (far,corner)]
        self.assertNotEqual(*degrees)

    def test_all_twelve_symmetries_preserve_graph_and_predictions(self):
        base = encode(self.history,global_tokens=2)
        model = RelationalNet(self.config).eval()
        with torch.no_grad():
            expected = model(pack([base]))
        for symmetry in range(12):
            graph = encode([transform(c,symmetry) for c in self.history],global_tokens=2)
            transformed = identities(base,symmetry)
            actual = {identity:i for i,identity in enumerate(identities(graph))}
            permutation = np.asarray([actual[key] for key in transformed])
            for name in ('kinds','owners','patterns','features'):
                np.testing.assert_array_equal(getattr(base,name),getattr(graph,name)[permutation])
            for name in EDGE_NAMES:
                edges = getattr(base,name).copy()
                edges[:,:2] = permutation[edges[:,:2]]
                self.assertEqual(sorted(map(tuple,edges)),sorted(map(tuple,getattr(graph,name))))
            with torch.no_grad():
                output = model(pack([graph]))
            action_index = {tuple(c):i for i,c in enumerate(graph.actions)}
            order = [action_index[transform(tuple(c),symmetry)] for c in base.actions]
            for key in ('logits','q'):
                torch.testing.assert_close(expected[key],output[key][order],rtol=2e-5,atol=2e-6)

    def test_batch_boundaries_budgets_and_no_action_loss(self):
        histories = [[],self.history[:2],self.history]
        graphs = [encode(h,global_tokens=2) for h in histories]
        model = RelationalNet(self.config).eval()
        with torch.no_grad():
            combined = model(pack(graphs))
            at = 0
            for graph in graphs:
                one = model(pack([graph]))
                for key in ('logits','q'):
                    torch.testing.assert_close(combined[key][at:at+len(graph.actions)],one[key],rtol=2e-5,atol=2e-6)
                at += len(graph.actions)
        self.assertEqual(combined['action_offsets'].tolist(),np.cumsum([0]+[len(g.actions) for g in graphs]).tolist())
        evaluator = NeuralEvaluator(model,'cpu',max_nodes=max(g.node_count for g in graphs),max_edges=max(g.edge_count for g in graphs))
        results = evaluator.evaluate(histories)
        for result,graph in zip(results,graphs):
            np.testing.assert_array_equal(result['actions'],graph.actions)
            self.assertEqual(result['logits'].dtype,np.float32)
            self.assertEqual(result['q'].dtype,np.float32)
            self.assertEqual(result['position_key'],graph.position_key)
            self.assertEqual((result['player'],result['remaining']),(graph.player,graph.remaining))
            self.assertEqual(result['model_version'],evaluator.model_version)
            self.assertTrue((np.abs(result['q']) <= 1).all())
        for kwargs in ({'max_nodes':1},{'max_edges':1}):
            with self.assertRaises(WorkBudgetError):
                list(iter_batches(graphs,**kwargs))
            with self.assertRaises(WorkBudgetError):
                encode(self.history,global_tokens=2,**kwargs)

    def test_checkpoint_gradients_and_production_dimensions(self):
        torch.set_num_threads(1)
        from dataclasses import replace
        graph = encode(self.history[:2],global_tokens=2)
        model = RelationalNet(self.config)
        baseline = RelationalNet(replace(self.config,checkpoint=False))
        baseline.load_state_dict(model.state_dict())
        for network in (model,baseline):
            result = network(pack([graph]))
            (result['logits'].square().mean()+result['q'].square().mean()).backward()
        for left,right in zip(model.parameters(),baseline.parameters()):
            self.assertIsNotNone(left.grad)
            self.assertTrue(torch.isfinite(left.grad).all())
            torch.testing.assert_close(left.grad,right.grad,rtol=0,atol=0)
        config = ModelConfig()
        self.assertEqual((config.width,config.blocks,config.heads,config.ff,config.global_tokens),(256,8,8,1024,16))
        with self.assertRaises(ValueError):
            ModelConfig(width=31,heads=8)

    def test_terminal_and_invalid_history_rejected(self):
        from tests.reference import interleave
        history = interleave([[(q,0) for q in range(6)],[(2*q,6) for q in range(6)]])
        with self.assertRaisesRegex(ValueError,'Terminal'):
            encode(history)
        with self.assertRaises(ValueError):
            encode([(0,0),(9,0)])

    def test_position_and_model_identity(self):
        history = [(0,0),(1,0),(2,0),(0,1),(0,2)]
        reordered = [history[i] for i in (0,2,1,4,3)]
        graph = encode(history,global_tokens=2)
        self.assertEqual(graph.position_key,encode(reordered,global_tokens=2).position_key)
        changed_owner = [history[i] for i in (0,3,2,1,4)]
        self.assertNotEqual(graph.position_key,encode(changed_owner,global_tokens=2).position_key)
        model = RelationalNet(self.config)
        first = NeuralEvaluator(model,'cpu').model_version
        self.assertEqual(first,NeuralEvaluator(copy.deepcopy(model),'cpu').model_version)
        with torch.no_grad():
            next(model.parameters()).add_(1)
        self.assertNotEqual(first,NeuralEvaluator(model,'cpu').model_version)
        self.assertEqual(NeuralEvaluator(model,'cpu',model_version='checkpoint-sha').model_version,'checkpoint-sha')
        evaluator = NeuralEvaluator(model,'cpu')
        before = evaluator.evaluate([history])[0]
        with torch.no_grad():
            model.policy[-1].bias.add_(3)
        after = evaluator.evaluate([history])[0]
        np.testing.assert_array_equal(before['logits'],after['logits'])
        self.assertEqual(before['model_version'],after['model_version'])
        self.assertTrue(all(not p.requires_grad for p in evaluator.model.parameters()))


if __name__ == '__main__':
    unittest.main()
