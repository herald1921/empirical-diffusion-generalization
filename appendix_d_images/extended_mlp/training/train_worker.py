"""Preserved original CFM recipe, with full recovery and a shared wall deadline.

New files only. Four workers share the existing GPU; no convergence early stop.
The original final checkpoints have no optimizer: that boundary is explicit.
"""
from pathlib import Path
import argparse
import copy
import csv
import fcntl
import hashlib
import json
import os
import random
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    q = path.with_suffix(path.suffix + '.tmp')
    with q.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(q, path)


def cpu_tree(x):
    if torch.is_tensor(x):
        return x.detach().clone() if x.device.type == 'cpu' else x.detach().cpu()
    if isinstance(x, dict):
        return {k: cpu_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [cpu_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple(cpu_tree(v) for v in x)
    return x


def append_csv(path, row):
    existed = path.exists()
    with path.open('a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not existed:
            w.writeheader()
        w.writerow(row)
        f.flush()


def snapshot(model, ema, opt, stream, step, cfg, task, timing, loss):
    # Finish GPU->CPU copies before training can mutate any tensors.
    return dict(
        model=cpu_tree(model.state_dict()),
        ema_model=cpu_tree(ema.state_dict()),
        optimizer=cpu_tree(opt.state_dict()),
        stream_state=stream.get_state().cpu().clone(),
        torch_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),
        numpy_rng=np.random.get_state(), python_rng=random.getstate(),
        step=step, optimizer_updates_since_reset=step-task['start_step'],
        optimizer_reset_at=task['start_step'],
        scheduler_state=None, scaler_state=None,
        scheduler_description='constant LR; no scheduler state exists',
        scaler_description='BF16 uses no gradient scaler',
        protocol=cfg, task=task, timing=timing, last_training_loss=loss,
    )


def verify_payload(payload, saved):
    assert saved['step'] == payload['step']
    assert saved['task'] == payload['task']
    for name in ['model', 'ema_model']:
        assert saved[name].keys() == payload[name].keys()
        assert all(torch.equal(v, saved[name][k]) for k, v in payload[name].items())
    expected = payload['optimizer']['state']
    actual = saved['optimizer']['state']
    assert actual.keys() == expected.keys()
    for k, state in expected.items():
        for name, value in state.items():
            got = actual[k][name]
            assert torch.equal(value, got) if torch.is_tensor(value) else value == got
        assert int(actual[k]['step'].item()) == payload['optimizer_updates_since_reset']
    assert torch.equal(saved['stream_state'], payload['stream_state'])
    assert torch.equal(saved['torch_rng'], payload['torch_rng'])
    assert all(torch.equal(a, b) for a, b in zip(saved['cuda_rng'], payload['cuda_rng']))


def commit_checkpoint(directory, name, payload):
    path = directory / name
    if path.exists():
        raise RuntimeError('Refusing to overwrite a checkpoint: ' + str(path))
    temporary = path.with_suffix('.pt.tmp')
    if temporary.exists():
        raise RuntimeError('Uncommitted checkpoint already exists: ' + str(temporary))
    with temporary.open('xb') as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    saved = torch.load(temporary, map_location='cpu', weights_only=False)
    verify_payload(payload, saved)
    del saved
    digest = sha(temporary)
    os.replace(temporary, path)
    metadata = dict(path=str(path), step=payload['step'], bytes=path.stat().st_size,
                    sha256=digest, full_recovery=True, committed_at=time.time(),
                    optimizer_updates_since_reset=payload['optimizer_updates_since_reset'])
    dump(path.with_suffix('.json'), metadata)
    dump(directory / 'latest_checkpoint.json', metadata)
    print('CHECKPOINT', json.dumps(metadata), flush=True)
    return metadata


def restore(payload, model, ema, opt, stream):
    model.load_state_dict(payload['model'])
    ema.load_state_dict(payload['ema_model'])
    opt.load_state_dict(payload['optimizer'])
    stream.set_state(payload['stream_state'])
    torch.set_rng_state(payload['torch_rng'])
    torch.cuda.set_rng_state_all(payload['cuda_rng'])
    np.random.set_state(payload['numpy_rng'])
    random.setstate(payload['python_rng'])


def update(model, ema, opt, stream, ref, cfg):
    z = ref[torch.randint(len(ref), (cfg['batch_size'],), device='cuda', generator=stream)]
    eps = torch.randn(z.shape, device='cuda', generator=stream)
    t = torch.rand(len(z), device='cuda', generator=stream)
    x = (1-t[:, None])*eps + t[:, None]*z
    opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        pred = model(x, t)
        loss = (pred.float() - (z-eps)).square().mean()
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite training loss')
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'])
    if not torch.isfinite(grad):
        raise RuntimeError('Nonfinite gradient norm')
    opt.step()
    with torch.no_grad():
        for ep, p in zip(ema.parameters(), model.parameters()):
            ep.mul_(cfg['ema_decay']).add_(p, alpha=1-cfg['ema_decay'])
        for eb, b in zip(ema.buffers(), model.buffers()):
            eb.copy_(b)
    return float(loss.detach()), float(grad)


def load_needed_data(task, data):
    """Load only this training set and two fixed panels; retain no full CPU set.

    The original uint16 -> float32 -> division operations and exact raw indices
    are preserved. In particular, S2 still uses its last N images.
    """
    manifest = json.loads((data/'manifest.json').read_text())
    with np.load(data/'paper_partitions.npz') as p:
        groups = [p['group_a_raw_indices'].copy(), p['group_b_raw_indices'].copy()]
    assert len(groups[0]) == len(groups[1]) == 98023
    assert not np.intersect1d(*groups).size
    gi = 0 if task['group'] == 's1' else 1
    raw_train = groups[gi][:task['n']] if gi == 0 else groups[gi][-task['n']:]
    gen = torch.Generator().manual_seed(2026101799)
    train_positions = torch.randperm(task['n'], generator=gen)[:512]
    other_positions = torch.randperm(98023, generator=gen)[:512]
    raw_other = groups[1-gi][other_positions.numpy()]
    wanted = np.concatenate([raw_train, raw_other])
    assert not np.intersect1d(raw_train, raw_other).size
    selected = torch.empty(len(wanted), 6400, dtype=torch.float32)
    seen = np.zeros(len(wanted), np.uint8)
    for shard in manifest['shards']:
        hit = (wanted >= shard['start']) & (wanted < shard['start']+shard['count'])
        if not hit.any():
            continue
        positions = np.flatnonzero(hit)
        with np.load(data/shard['filename']) as z:
            raw = z['gray_rgb_sum'].reshape(shard['count'], 6400)
            values = torch.from_numpy(np.ascontiguousarray(raw[wanted[hit]-shard['start']])).float()
            values.div_(manifest['decode_divisor'])
            selected[positions] = values
        seen[positions] += 1
    assert np.all(seen == 1)
    ref = selected[:task['n']].cuda()
    # clone is essential: panels must not retain the large selected storage.
    train_panel = selected[train_positions].clone()
    other_panel = selected[task['n']:].clone()
    panel_info = dict(train_positions=train_positions.numpy(), other_positions=other_positions.numpy(),
                      train_raw_indices=raw_train[train_positions.numpy()], other_raw_indices=raw_other)
    return ref, train_panel, other_panel, panel_info


@torch.inference_mode()
def evaluate(model, ema, train_cpu, other_cpu, directory, label, step, task, panel_info):
    generator = torch.Generator().manual_seed(2026101799)
    count = 512
    train_ids = torch.randperm(task['n'], generator=generator)[:count]
    other_ids = torch.randperm(98023, generator=generator)[:count]
    assert np.array_equal(train_ids.numpy(), panel_info['train_positions'])
    assert np.array_equal(other_ids.numpy(), panel_info['other_positions'])
    eps = torch.randn(count, 6400, generator=generator)
    uniform = torch.rand(count, generator=generator)
    arrays = {'train_panel_indices': train_ids.numpy(), 'other_panel_indices': other_ids.numpy(),
              'train_raw_indices': panel_info['train_raw_indices'], 'other_raw_indices': panel_info['other_raw_indices']}
    reports = {}
    for name, net in [('raw', model), ('ema', ema)]:
        net.eval()
        by_time = {}
        for t in ['uniform', .1, .3, .5, .8, .95, .99]:
            ts = uniform if t == 'uniform' else torch.full((count,), t)
            pair = []
            for group_name, ref, ids in [('train', train_cpu, train_ids), ('other', other_cpu, other_ids)]:
                losses = []
                for b in range(0, count, 32):
                    z = ref[b:b+32].cuda()
                    e, tt = eps[b:b+32].cuda(), ts[b:b+32].cuda()
                    v = net((1-tt[:, None])*e + tt[:, None]*z, tt)
                    losses.append((v-(z-e)).square().mean(1).double().cpu().numpy())
                values = np.concatenate(losses)
                assert np.isfinite(values).all()
                arrays[f'{name}_{t}_{group_name}_velocity_mse'] = values
                arrays[f'{name}_{t}_{group_name}_clean_mse'] = values*(1-ts.numpy())**2
                pair.append(values)
            a, b = pair
            by_time[str(t)] = dict(train_velocity_mse=float(a.mean()), other_velocity_mse=float(b.mean()),
                                  gap_velocity_mse=float((b-a).mean()),
                                  gap_standard_error=float((b-a).std(ddof=1)/np.sqrt(count)),
                                  train_clean_mse=float((a*(1-ts.numpy())**2).mean()),
                                  other_clean_mse=float((b*(1-ts.numpy())**2).mean()))
        reports[name] = by_time
    model.train()
    np.savez_compressed(directory / f'evaluation_{label}.npz', **arrays)
    dump(directory / f'evaluation_{label}.json', dict(step=step, count=count, seed=2026101799,
        selected_by='fixed seed, distinct diagnostic RNG; unchanged before/after',
        reports=reports, finished_at=time.time()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--key', required=True)
    args = parser.parse_args()
    root = Path(args.root)
    cfg = json.loads((root / 'protocol.json').read_text())
    task = next(t for t in cfg['tasks'] if t['key'] == args.key)
    directory = root / task['key']
    directory.mkdir(exist_ok=True)
    sys.path.insert(0, str(root / 'original_code'))
    from models import make_model
    from core import DATA
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(task['stream_seed'])
    np.random.seed(task['stream_seed'] % (2**32))
    random.seed(task['stream_seed'])
    interrupt = []
    signal.signal(signal.SIGUSR1, lambda signum, frame: interrupt.append('watchdog_deadline'))
    signal.signal(signal.SIGTERM, lambda signum, frame: interrupt.append('external_SIGTERM'))
    signal.signal(signal.SIGINT, lambda signum, frame: interrupt.append('external_SIGINT'))
    dump(directory / 'status.json', dict(state='loading', pid=os.getpid(), time=time.time()))
    assert sha(task['source']) == task['source_sha256'], 'Source SHA256 changed'
    source = torch.load(task['source'], map_location='cpu', weights_only=False)
    assert source['step'] == task['start_step']
    assert 'optimizer' not in source
    model = make_model('mlp').cuda()
    assert sum(p.numel() for p in model.parameters()) == cfg['parameters']
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    model.load_state_dict(source['model'])
    ema.load_state_dict(source['ema_model'])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    stream = torch.Generator(device='cuda').manual_seed(task['stream_seed'])
    dump(directory / 'status.json', dict(state='loading_selected_data', pid=os.getpid(), time=time.time()))
    ref, ref_cpu, other_cpu, panel_info = load_needed_data(task, DATA)
    assert len(ref) == task['n'] and len(ref_cpu) == len(other_cpu) == 512
    model.train()
    dump(directory / 'status.json', dict(state='waiting_recovery_lock', pid=os.getpid(), time=time.time()))
    preflight_lock = (root/'preflight_recovery.lock').open('a')
    fcntl.flock(preflight_lock, fcntl.LOCK_EX)
    if interrupt:
        raise RuntimeError('Interrupted during data loading')
    dump(directory / 'status.json', dict(state='checking_recovery', pid=os.getpid(), time=time.time()))
    # Two-update disk round trip, before the official ten-hour interval.
    original_stream = stream.get_state().clone()
    original_rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), np.random.get_state(), random.getstate())
    loss, _ = update(model, ema, opt, stream, ref, cfg)
    recovery = snapshot(model, ema, opt, stream, task['start_step']+1, cfg, task, {}, loss)
    previous = Path(cfg['reuse_recovery_root'])/task['key']/'preflight_recovery.pt'
    if not previous.exists():
        previous = previous.with_suffix('.pt.tmp')
    if previous.exists():
        recovered = torch.load(previous, map_location='cpu', weights_only=False)
        # Exact equality also checks the new data loader against the original
        # loader: its first update must give the identical weights and moments.
        verify_payload(recovery, recovered)
        dump(directory/'preflight_recovery_reused.json', dict(path=str(previous),
            sha256=sha(previous), step=recovery['step'], exact_first_update_match=True))
    else:
        commit_checkpoint(directory, 'preflight_recovery.pt', recovery)
        recovered = torch.load(directory/'preflight_recovery.pt', map_location='cpu', weights_only=False)
    expected_loss, _ = update(model, ema, opt, stream, ref, cfg)
    expected_model = cpu_tree(model.state_dict())
    expected_ema = cpu_tree(ema.state_dict())
    restore(recovered, model, ema, opt, stream)
    replay_loss, _ = update(model, ema, opt, stream, ref, cfg)
    assert expected_loss == replay_loss, ('Replay loss differs', expected_loss, replay_loss)
    assert all(torch.equal(v.cpu(), expected_model[k]) for k, v in model.state_dict().items())
    assert all(torch.equal(v.cpu(), expected_ema[k]) for k, v in ema.state_dict().items())
    dump(directory / 'replay_verified.json', dict(bitwise_model_and_ema=True,
        loss=expected_loss, optimizer_restored=True, stream_restored=True,
        source_step=task['start_step'], official_training_not_started=True))
    del recovered, recovery, expected_model, expected_ema
    # Restore the untouched original source after preflight, including empty AdamW.
    model.load_state_dict(source['model'])
    ema.load_state_dict(source['ema_model'])
    del opt
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    stream.set_state(original_stream)
    torch.set_rng_state(original_rng[0])
    torch.cuda.set_rng_state_all(original_rng[1])
    np.random.set_state(original_rng[2])
    random.setstate(original_rng[3])
    del source
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    fcntl.flock(preflight_lock, fcntl.LOCK_UN)
    preflight_lock.close()
    dump(directory / 'status.json', dict(state='evaluating_before', pid=os.getpid(), time=time.time()))
    evaluate(model, ema, ref_cpu, other_cpu, directory, 'before', task['start_step'], task, panel_info)
    torch.cuda.synchronize()
    dump(directory / 'ready.json', dict(pid=os.getpid(), source_step=task['start_step'],
        source_sha256=task['source_sha256'], ready_at=time.time(),
        gpu_peak_bytes=torch.cuda.max_memory_allocated(), optimizer_reset=True))
    dump(directory / 'status.json', dict(state='ready_at_barrier', pid=os.getpid(), time=time.time()))
    while not (root / 'start.json').exists():
        if interrupt:
            raise RuntimeError('Interrupted before start: ' + repr(interrupt))
        time.sleep(.25)
    timing = json.loads((root / 'start.json').read_text())
    while time.time() < timing['start_epoch']:
        time.sleep(max(0., min(.05, timing['start_epoch']-time.time())))
    remaining = timing['deadline_epoch'] - time.time()
    if remaining <= 0:
        raise RuntimeError('Common start interval already expired')
    # monotonic deadline is immune to later wall-clock corrections.
    begin = time.monotonic()
    deadline = begin + remaining
    step = task['start_step']
    losses = []
    last_status = begin
    last_loss = None
    writer = ThreadPoolExecutor(max_workers=1)
    pending = None
    first_step_epoch = None
    last_update_epoch = None
    while time.monotonic() < deadline and not interrupt:
        last_loss, grad = update(model, ema, opt, stream, ref, cfg)
        step += 1
        last_update_epoch = time.time()
        if first_step_epoch is None:
            first_step_epoch = last_update_epoch
        losses.append(last_loss)
        now = time.monotonic()
        if now-last_status >= 10 or step % 1000 == 0:
            dump(directory / 'status.json', dict(state='training', pid=os.getpid(), step=step,
                start_step=task['start_step'], updates=step-task['start_step'],
                elapsed_seconds=now-begin, last_loss=last_loss, gradient_norm=grad,
                deadline_epoch=timing['deadline_epoch'], first_step_epoch=first_step_epoch,
                last_update_epoch=last_update_epoch, time=time.time()))
            last_status = now
        if step % 1000 == 0:
            row = dict(step=step, mean_loss=float(np.mean(losses)), last_loss=last_loss,
                       gradient_norm=grad, elapsed_seconds=now-begin, time=time.time())
            append_csv(directory / 'training.csv', row)
            print('TRAIN', json.dumps(row), flush=True)
            losses = []
        if pending is not None and pending.done():
            pending.result()
            pending = None
        if step % cfg['checkpoint_every'] == 0:
            if pending is not None:
                pending.result()
            payload = snapshot(model, ema, opt, stream, step, cfg, task, timing, last_loss)
            pending = writer.submit(commit_checkpoint, directory, f'checkpoint_step_{step:09d}.pt', payload)
    torch.cuda.synchronize()
    stopped_epoch = time.time()
    elapsed = time.monotonic()-begin
    stop_reason = interrupt[-1] if interrupt else 'ten_hour_deadline'
    dump(directory / 'status.json', dict(state='training_stopped_saving', pid=os.getpid(),
        step=step, stopped_epoch=stopped_epoch, stop_reason=stop_reason,
        last_update_epoch=last_update_epoch, elapsed_seconds=elapsed))
    if pending is not None:
        pending.result()
    final = snapshot(model, ema, opt, stream, step, cfg, task, timing, last_loss)
    final['timing'].update(first_step_epoch=first_step_epoch, last_update_epoch=last_update_epoch,
                           stopped_epoch=stopped_epoch, worker_elapsed_seconds=elapsed)
    metadata = commit_checkpoint(directory, f'final_step_{step:09d}.pt', final)
    del final
    writer.shutdown(wait=True)
    if losses:
        append_csv(directory / 'training.csv', dict(step=step, mean_loss=float(np.mean(losses)),
            last_loss=last_loss, gradient_norm=grad, elapsed_seconds=elapsed, time=time.time()))
    evaluate(model, ema, ref_cpu, other_cpu, directory, 'after', step, task, panel_info)
    complete = dict(state='complete' if not interrupt else 'interrupted', key=task['key'],
        start_step=task['start_step'], final_step=step, updates=step-task['start_step'],
        common_start_epoch=timing['start_epoch'], deadline_epoch=timing['deadline_epoch'],
        first_step_epoch=first_step_epoch, last_update_epoch=last_update_epoch,
        training_stopped_epoch=stopped_epoch, worker_elapsed_seconds=elapsed,
        stop_reason=stop_reason, checkpoint=metadata, finished_at=time.time())
    dump(directory / 'complete.json', complete)
    dump(directory / 'status.json', complete)
    print('COMPLETE', json.dumps(complete), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        info = dict(state='error', pid=os.getpid(), traceback=traceback.format_exc(), time=time.time())
        try:
            args_root = Path(sys.argv[sys.argv.index('--root')+1])
            args_key = sys.argv[sys.argv.index('--key')+1]
            dump(args_root / args_key / 'status.json', info)
        except Exception:
            pass
        traceback.print_exc()
        raise
