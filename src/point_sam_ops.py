"""Inference-only Torch equivalents for optional torkit3d operations on Windows.

The upstream model and trained weights are used unchanged. These operations
replace unavailable compiled sampling/indexing kernels, not the learned model.
"""
import importlib
import sys
import types


def install_torch_ops():
    try:
        importlib.import_module('torkit3d.ops.sample_farthest_points')
        importlib.import_module('torkit3d.nn.functional')
        return 'torkit3d'
    except ImportError:
        pass
    import torch

    def batch_index_select(values, index, dim=1):
        if dim != 1:
            raise ValueError('Point-SAM inference adapter supports dim=1 only')
        batch = torch.arange(values.shape[0], device=values.device).reshape(
            (values.shape[0],) + (1,) * (index.ndim - 1))
        return values[batch, index.long()]

    @torch.no_grad()
    def sample_farthest_points(points, num_samples):
        batch, count, _ = points.shape
        if not 0 < num_samples <= count:
            raise ValueError('FPS sample count exceeds cloud size')
        indices = torch.empty((batch, num_samples), dtype=torch.long, device=points.device)
        distances = torch.full((batch, count), float('inf'), device=points.device)
        farthest = torch.zeros(batch, dtype=torch.long, device=points.device)
        rows = torch.arange(batch, device=points.device)
        for step in range(num_samples):
            indices[:, step] = farthest
            d = ((points - points[rows, farthest, None]) ** 2).sum(-1)
            distances = torch.minimum(distances, d)
            farthest = distances.argmax(-1)
        return indices

    def chamfer_distance(first, second):
        distances = torch.cdist(first, second).square()
        return distances.min(2).values, distances.min(1).values

    modules = ['torkit3d', 'torkit3d.nn', 'torkit3d.nn.functional', 'torkit3d.ops',
               'torkit3d.ops.sample_farthest_points', 'torkit3d.ops.chamfer_distance']
    for name in modules:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    sys.modules['torkit3d.nn.functional'].batch_index_select = batch_index_select
    sys.modules['torkit3d.ops.sample_farthest_points'].sample_farthest_points = sample_farthest_points
    sys.modules['torkit3d.ops.chamfer_distance'].chamfer_distance = chamfer_distance
    return 'PyTorch（Windows 推理适配）'
