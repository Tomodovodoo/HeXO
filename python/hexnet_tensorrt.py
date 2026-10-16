"""A TensorRT fp16 engine behind the ActorGraph interface.

TensorRTGraph keeps ActorGraph's capacities, segments, memory budget and replay path; each (canvas, capacity)
capture records, in one CUDA graph, HexNet's input line features (export_web.inputs), one execution of a TensorRT
engine built from export_web.WebNet, and the packed [policy, far, value] copy. The engine has one dynamic-shape
profile over every canvas ActorGraph captures; its plan is cached under PLANS by the model digest, the GPU and the
TensorRT version, so a checkpoint builds once per GPU. Post-activation networks (HexNetConfig.block 'post') are where it
pays: their norms fold into the convolutions.
"""
from pathlib import Path
import io
import os

import torch

import export_web
import hexnet
from hexnet_graphs import ActorGraph

PLANS = Path.home()/'.cache'/'hexo-tensorrt'
MIN_SIDE, OPT_SIDE, MAX_SIDE, MAX_ROWS = 24, 32, 64, 128


def build_plan(model, path, logger, device):
    """Build and cache the plan on `device` (the current CUDA device). Concurrent builders each write their own
    temporary file and atomically replace `path` with identical plans."""
    import tensorrt as trt
    net = export_web.WebNet(model).half().to(device).eval()
    example = torch.zeros(1, export_web.CHANNELS, OPT_SIDE, OPT_SIDE, device=device, dtype=torch.float16)
    buffer = io.BytesIO()
    torch.onnx.export(net, (example,), buffer, input_names=['features'], output_names=['policy', 'far', 'value'],
                      dynamic_axes=dict(features={0: 'batch', 2: 'height', 3: 'width'},
                                        policy={0: 'batch', 1: 'cells'}, far={0: 'batch'}, value={0: 'batch'}),
                      opset_version=export_web.OPSET, dynamo=False, do_constant_folding=True)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(buffer.getvalue()):
        raise RuntimeError('; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    profile = builder.create_optimization_profile()
    channels = export_web.CHANNELS
    profile.set_shape('features', (1, channels, MIN_SIDE, MIN_SIDE), (64, channels, OPT_SIDE, OPT_SIDE),
                      (MAX_ROWS, channels, MAX_SIDE, MAX_SIDE))
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError('TensorRT could not build the engine')
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    pending.write_bytes(bytes(plan))
    pending.replace(path)


class TensorRTGraph(ActorGraph):
    CANVASES = tuple(c for c in ActorGraph.CANVASES if c <= MAX_SIDE)

    def __init__(self, model, max_incremental_bytes=384*1024*1024, max_batch=32):
        import tensorrt as trt
        super().__init__(model, max_incremental_bytes, max_batch)
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.ERROR)
        # WebNet reads the reference float weights; export from a CPU float copy.
        reference = hexnet.HexNet(model.config, model.future_target)
        reference.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
        reference = reference.float().eval()
        # TensorRT objects belong to the current CUDA device: build, load and capture on the model's.
        with torch.cuda.device(self.device):
            gpu = '{}-sm{}{}'.format(torch.cuda.get_device_name(self.device).replace(' ', '_'),
                                     *torch.cuda.get_device_capability(self.device))
            path = PLANS/f'{hexnet.model_digest(reference)}-{gpu}-trt{trt.__version__}.plan'
            if not path.exists():
                build_plan(reference, path, self.logger, self.device)
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(path.read_bytes())
        self.lines = model.lines
        self.contexts = []

    @classmethod
    def supports(cls, canvas):
        height, width = cls.shape(canvas)
        return MIN_SIDE <= min(height, width) and max(height, width) <= MAX_SIDE and height % 8 == width % 8 == 0

    def _features(self, planes):
        mask = planes[:, 3:4]
        features = torch.cat((planes, self.lines(planes[:, :1], planes[:, 1:2], mask)), 1)*mask
        return torch.nn.functional.pad(features, (0, 0, 0, 0, 0, export_web.CHANNELS-features.shape[1]))

    def _capture(self, side, capacity):
        with torch.cuda.device(self.device):
            return self._capture_here(side, capacity)

    def _capture_here(self, side, capacity):
        if self.incremental_reserved_bytes >= self.max_incremental_bytes:
            raise MemoryError('actor graph memory budget exhausted')
        height, width = self.shape(side)
        template = torch.empty((capacity, 8, height, width), device=self.device,
                               dtype=torch.bfloat16, memory_format=torch.channels_last).zero_()
        template[:, 3, 0, 0] = 1
        static_input = template.clone(memory_format=torch.channels_last)
        static_packed = torch.empty((capacity, height*width+2), device=self.device, dtype=torch.float32)
        context = self.engine.create_execution_context()
        context.set_input_shape('features', (capacity, export_web.CHANNELS, height, width))
        features = torch.empty((capacity, export_web.CHANNELS, height, width), device=self.device, dtype=torch.float16)
        outputs = {name: torch.empty(tuple(context.get_tensor_shape(name)), device=self.device, dtype=torch.float32)
                   for name in ('policy', 'far', 'value')}
        context.set_tensor_address('features', features.data_ptr())
        for name, tensor in outputs.items():
            context.set_tensor_address(name, tensor.data_ptr())

        def run():
            features.copy_(self._features(static_input))
            if not context.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream):
                raise RuntimeError('TensorRT execution failed')
            static_packed.copy_(torch.cat((outputs['policy'], outputs['far'].view(-1, 1),
                                           outputs['value'].view(-1, 1)), dim=1))

        with torch.cuda.stream(self.stream), torch.inference_mode():
            run()   # TensorRT allocates on its first execution; graphs capture a warmed context
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, pool=self.pool, stream=self.stream,
                                                      capture_error_mode='thread_local'):
            run()
        try:
            self._measure_memory((template, static_input, static_packed, features, *outputs.values()))
        except MemoryError:
            del graph
            raise
        if self.incremental_reserved_bytes > self.max_incremental_bytes:
            raise MemoryError(f'actor graph pool and staging need {self.incremental_reserved_bytes/2**20:.1f} MiB; '
                              f'budget is {self.max_incremental_bytes/2**20:.1f} MiB')
        self.contexts.append((context, features, outputs))
        self.graphs[side, capacity] = graph, template, static_input, static_packed
        return self.graphs[side, capacity]

    def close(self):
        """Drop the execution contexts and their tensors, after the stream drains, before ActorGraph empties the
        CUDA cache, so their memory is returned too."""
        with self.lock:
            self.stream.synchronize()
            self.contexts.clear()
        super().close()
