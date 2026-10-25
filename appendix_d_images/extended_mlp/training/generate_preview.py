"""Paired seeds 0..3 from the original paper noise; saved EMA only."""
from pathlib import Path
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

root = Path('/root/shared-nvme/celeba80_mlp_10h_20261018_lowmem')
original = Path('/root/figure07_cfm_20260921')
sys.path.insert(0, str(original / 'code'))
sys.path.insert(0, str(root / 'original_code'))
from models import make_model
from sampling import integrate, schedule


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


torch.set_num_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
manifest = json.loads((root / 'preview_checkpoint_manifest.json').read_text())
pause = json.loads((root / 'preview_pause_01.json').read_text())
assert pause['state'] == 'training_resumed'
for key, pid in pause['resumed_worker_pids'].items():
    cmd = Path('/proc') / str(pid) / 'cmdline'
    assert cmd.exists() and str(root).encode() in cmd.read_bytes()
    assert json.loads((root / key / 'status.json').read_text())['state'] == 'training'
output = root / 'preview_01'
output.mkdir(exist_ok=False)
cfg = json.loads((original / 'protocol/experiment.json').read_text())
with np.load(original / 'noise.npz') as f:
    noise = f['noise'][:4].copy()
assert noise.shape == (4, 6400)
for seed in range(4):
    check = torch.randn(6400, generator=torch.Generator().manual_seed(seed)).numpy()
    assert np.array_equal(noise[seed], check)
grid = schedule(cfg['sampler_points'], cfg['tau_end'], cfg['capture_tau'])
arrays = dict(noise=noise, grid_tau=grid, seeds=np.arange(4))
report = dict(created_epoch=time.time(), weighting='EMA', groups=['s1', 's2'],
              n_values=[10000, 98023], source_seeds=[0, 1, 2, 3],
              original_noise_sha256=sha(original / 'noise.npz'),
              sampler='FP32 Heun, original 160-interval grid plus original inserted captures',
              tau_end=cfg['tau_end'], grid_intervals=len(grid)-1,
              display='clip clean endpoint readout to [0,1], no contrast enhancement', tasks={})
for n in [10000, 98023]:
    for group in ['s1', 's2']:
        key = f'N{n}_{group}'
        cp = manifest['tasks'][key]
        path = Path(cp['path'])
        assert sha(path) == cp['sha256']
        saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
        assert saved['step'] == cp['step']
        model = make_model('mlp').cuda().eval()
        model.load_state_dict(saved['ema_model'])
        del saved
        with torch.inference_mode():
            state, clean, _ = integrate(model, torch.from_numpy(noise).cuda(), grid)
        raw = clean.cpu().numpy()
        assert raw.shape == (4, 6400) and np.isfinite(raw).all()
        arrays[key + '_raw'] = raw
        arrays[key + '_output'] = np.clip(raw, 0, 1)
        arrays[key + '_state'] = state.cpu().numpy()
        arrays[key + '_step'] = np.asarray(cp['step'])
        report['tasks'][key] = dict(step=cp['step'], checkpoint=str(path), sha256=cp['sha256'],
            clip_fraction=float(((raw < 0) | (raw > 1)).mean()),
            output_min=float(raw.min()), output_max=float(raw.max()))
        print('SAMPLED', key, cp['step'], flush=True)
        del model, state, clean
        gc.collect()
        torch.cuda.empty_cache()
with (output / 'samples.npz').open('xb') as f:
    np.savez_compressed(f, **arrays)
report['samples_sha256'] = sha(output / 'samples.npz')
report['finished_epoch'] = time.time()
(output / 'provenance.json').write_text(json.dumps(report, indent=2))
print('PREVIEW_COMPLETE', json.dumps(report), flush=True)
