"""Persistent JSON-lines Point-SAM worker, running in the configured Python env."""
from __future__ import annotations
import contextlib
import importlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np


def sample_indices(count, limit, prompts, seed=42):
    required = np.unique(np.asarray(prompts, np.int64))
    if count < 3 or (len(required) and (required.min() < 0 or required.max() >= count)):
        raise ValueError('点云或提示点索引无效。')
    if len(required) > limit:
        raise ValueError('提示点数量超过采样点数，请减少提示点。')
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    pool = np.setdiff1d(np.arange(count), required, assume_unique=True)
    picked = np.random.default_rng(seed).choice(pool, limit - len(required), replace=False)
    return np.sort(np.r_[required, picked])


class Backend:
    def __init__(self):
        self.model = None
        self.key = None

    def probe(self, request):
        missing = []
        for name in ('torch', 'torchvision', 'hydra', 'omegaconf', 'timm', 'safetensors', 'scipy'):
            try:
                importlib.import_module(name)
            except Exception as exc:
                missing.append(f'{name}: {exc}')
        if missing:
            raise RuntimeError('Point-SAM 环境依赖不可用：\n' + '\n'.join(missing))
        repo, checkpoint = Path(request['repo']), Path(request['checkpoint'])
        if not (repo / 'pc_sam/model/pc_sam.py').is_file():
            raise ValueError('Point-SAM 源码路径无效：' + str(repo))
        if not checkpoint.is_file():
            raise ValueError('找不到预训练权重：' + str(checkpoint))
        import torch
        from point_sam_ops import install_torch_ops
        ops = install_torch_ops()
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from pc_sam.model.pc_sam import PointCloudSAM
        from safetensors import safe_open
        with safe_open(str(checkpoint), framework='pt', device='cpu') as weights:
            tensors = len(weights.keys())
        return {'message': f'环境可用 · {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}'
                           f' · {tensors} 个权重张量 · {ops}', 'cuda': torch.cuda.is_available()}

    def load(self, request):
        import torch
        repo, checkpoint = Path(request['repo']).resolve(), Path(request['checkpoint']).resolve()
        device = request.get('device', 'auto')
        device = ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('所选环境没有可用 CUDA，请选择自动或 CPU。')
        key = (str(repo), str(checkpoint), checkpoint.stat().st_mtime_ns, device)
        if self.model is None or key != self.key:
            self.probe(request)
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
            from safetensors.torch import load_model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            config = OmegaConf.load(repo / 'configs/model/default.yaml')
            # Native LayerNorm has the same learned weight/bias as Apex's fused variant.
            self.model = instantiate(config)
            load_model(self.model, str(checkpoint), strict=True)
            self.model.eval().to(device)
            self.key = key
        return self.model, device

    def predict(self, request):
        import torch
        from scipy.spatial import cKDTree
        started = time.perf_counter()
        model, device = self.load(request)
        with np.load(request['input'], allow_pickle=False) as data:
            xyz = data['xyz'].astype(np.float64)
            rgb = data['rgb'].astype(np.float32) / 255.0
            prompts = data['prompts'].astype(np.int64)
            labels = data['labels'].astype(np.int64)
            prior = data['prior'].astype(bool) if 'prior' in data else None
        if xyz.ndim != 2 or xyz.shape[1] != 3 or rgb.shape != xyz.shape or not np.isfinite(xyz).all():
            raise ValueError('推理点云形状或坐标无效。')
        if len(xyz) < 64:
            raise ValueError('当前区域少于 64 个点，请扩大区域。')
        if labels.shape != prompts.shape or not len(prompts) or not np.isin(labels, (0, 1)).all() or not labels.any():
            raise ValueError('请至少添加一个正点，提示标签必须为正点或负点。')
        indices = sample_indices(len(xyz), max(256, int(request.get('max_points', 10000))), prompts)
        center = xyz.mean(axis=0)
        scale = max(float(np.linalg.norm(xyz - center, axis=1).max()), 1e-9)
        coords = torch.as_tensor(((xyz[indices] - center) / scale).astype(np.float32), device=device)[None]
        features = torch.as_tensor(rgb[indices], device=device)[None]
        points = torch.as_tensor(((xyz[prompts] - center) / scale).astype(np.float32), device=device)[None]
        point_labels = torch.as_tensor(labels, device=device)[None]
        seed = None
        if prior is not None:
            if prior.shape != (len(xyz),):
                raise ValueError('已有标注掩膜与推理区域不一致。')
            seed = torch.as_tensor(np.where(prior[indices], 2., -2.).astype(np.float32), device=device)[None]
        grouper = model.pc_encoder.patch_embed.grouper
        grouper.num_groups = min(512, len(indices))
        grouper.group_size = min(64, len(indices))
        torch.manual_seed(42)
        amp = torch.autocast('cuda', dtype=torch.float16) if device == 'cuda' else contextlib.nullcontext()
        with torch.inference_mode(), amp:
            masks, scores = model.predict_masks(coords, features, points, point_labels,
                                                 prompt_masks=seed, multimask_output=seed is None)
        sampled_logits = masks[0].float().cpu().numpy()
        scores = scores[0].float().cpu().numpy()
        if sampled_logits.ndim != 2 or sampled_logits.shape[1] != len(indices) or not np.isfinite(sampled_logits).all():
            raise RuntimeError('Point-SAM 返回了无效的分割结果。')
        if len(indices) == len(xyz):
            logits = sampled_logits
        else:
            tree = cKDTree(xyz[indices])
            logits = np.empty((len(scores), len(xyz)), np.float32)
            distance_limit = float(request.get('max_distance', 0))
            for start in range(0, len(xyz), 100000):
                stop = min(start + 100000, len(xyz))
                distances, nearest = tree.query(xyz[start:stop], workers=1)
                logits[:, start:stop] = sampled_logits[:, nearest]
                if distance_limit > 0:
                    block = logits[:, start:stop]
                    block[:, distances > distance_limit] = -100
            logits[:, indices] = sampled_logits  # preserve duplicate-coordinate point identity
        # User prompts constrain their own points even when the model disagrees.
        logits[:, prompts] = np.where(labels[None] > 0, 100., -100.)
        np.savez_compressed(request['output'], logits=logits, scores=scores)
        return {'output': request['output'], 'sampled': len(indices), 'total': len(xyz),
                'seconds': round(time.perf_counter() - started, 3), 'device': device}


def main():
    backend = Backend()
    protocol = sys.stdout
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            # Upstream prints must never be mixed with the JSON response stream.
            with contextlib.redirect_stdout(sys.stderr):
                result = backend.probe(request) if request.get('op') == 'probe' else backend.predict(request)
            response = {'id': request['id'], 'ok': True, 'result': result}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {'id': request.get('id'), 'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
        print(json.dumps(response, ensure_ascii=False), file=protocol, flush=True)


if __name__ == '__main__':
    main()
