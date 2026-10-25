"""Continue the completed ten-hour stage for the user-authorized extra hour."""
from pathlib import Path
import argparse
import copy
import csv
import fcntl
import json
import os
import signal
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import train_worker as w


def restore_saved(saved, model, ema, opt, stream):
    # AdamW may retain its CPU scalar step tensor by reference on load.
    # Keep the immutable CPU checkpoint separate from the live optimizer.
    isolated = dict(saved, optimizer=copy.deepcopy(saved['optimizer']))
    w.restore(isolated, model, ema, opt, stream)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--key', required=True)
    args = p.parse_args()
    root = Path(args.root)
    cfg = json.loads((root / 'protocol.json').read_text())
    task = next(t for t in cfg['tasks'] if t['key'] == args.key)
    source_directory = root / task['key']
    extension = root / 'extension_01'
    directory = extension / task['key']
    directory.mkdir(exist_ok=True)
    request = json.loads((root / 'extension_request.json').read_text())
    timing = json.loads((extension / 'start.json').read_text())
    previous = json.loads((source_directory / 'complete.json').read_text())
    source = Path(previous['checkpoint']['path'])
    assert request['user_authorized'] and request['extension_seconds'] == 3600
    assert previous['state'] == 'complete' and previous['stop_reason'] == 'ten_hour_deadline'
    assert source.is_file() and time.time() < timing['deadline_epoch']
    sys.path.insert(0, str(root / 'original_code'))
    from models import make_model
    from core import DATA
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    interrupted = []
    signal.signal(signal.SIGUSR1, lambda s, f: interrupted.append('watchdog_deadline'))
    signal.signal(signal.SIGTERM, lambda s, f: interrupted.append('external_SIGTERM'))
    signal.signal(signal.SIGINT, lambda s, f: interrupted.append('external_SIGINT'))
    w.dump(directory / 'status.json', dict(state='loading_resume_data', pid=os.getpid(), time=time.time()))
    model = make_model('mlp').cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    stream = torch.Generator(device='cuda')
    ref, train_cpu, other_cpu, panel_info = w.load_needed_data(task, DATA)
    model.train()
    # Serialize full-state recovery tests to keep memory below the cgroup cap.
    with (root / 'resume_recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert w.sha(source) == previous['checkpoint']['sha256']
        saved = torch.load(source, map_location='cpu', weights_only=False)
        assert saved['task'] == task and saved['protocol'] == cfg
        assert saved['step'] == previous['final_step']
        assert saved['optimizer_updates_since_reset'] == saved['step'] - task['start_step']
        restore_saved(saved, model, ema, opt, stream)
        expected_loss, expected_grad = w.update(model, ema, opt, stream, ref, cfg)
        expected = w.snapshot(model, ema, opt, stream, saved['step'] + 1, cfg, task, timing, expected_loss)
        restore_saved(saved, model, ema, opt, stream)
        replay_loss, replay_grad = w.update(model, ema, opt, stream, ref, cfg)
        replay = w.snapshot(model, ema, opt, stream, saved['step'] + 1, cfg, task, timing, replay_loss)
        assert expected_loss == replay_loss and expected_grad == replay_grad
        w.verify_payload(expected, replay)
        restore_saved(saved, model, ema, opt, stream)
        step = saved['step']
        first_step_epoch = saved['timing']['first_step_epoch']
        last_update_epoch = saved['timing']['last_update_epoch']
        last_loss = saved['last_training_loss']
        del expected, replay, saved
        model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        w.dump(directory / 'resume_replay_verified.json', dict(
            checkpoint=str(source), sha256=previous['checkpoint']['sha256'], step=step,
            next_step_loss=expected_loss, bitwise_model_ema_optimizer_and_rng=True,
            optimizer_reset=False, verified_at=time.time()))
    w.dump(directory / 'resume_ready.json', dict(pid=os.getpid(), step=step, ready_at=time.time()))
    w.dump(directory / 'ready.json', dict(pid=os.getpid(), source_step=step,
        full_state_resume=True, optimizer_reset=False, ready_at=time.time()))
    w.dump(directory / 'status.json', dict(state='resume_ready', pid=os.getpid(), step=step, time=time.time()))
    while not (extension / 'resume_start.json').exists():
        if interrupted:
            raise RuntimeError('Interrupted before resume barrier')
        time.sleep(.2)
    resume_start = json.loads((extension / 'resume_start.json').read_text())
    while time.time() < resume_start['resume_epoch']:
        time.sleep(.02)
    resume_epoch = time.time()
    # Restore the unfinished 1,000-update mean, without losing logged updates.
    with (source_directory / 'training.csv').open() as f:
        old_rows = list(csv.DictReader(f))
    assert int(old_rows[-1]['step']) == step
    losses = [float(old_rows[-1]['mean_loss'])] * (step % 1000)
    begin = time.monotonic() - (time.time() - timing['start_epoch'])
    deadline = time.monotonic() + (timing['deadline_epoch'] - time.time())
    last_status = time.monotonic()
    pending = None
    writer = ThreadPoolExecutor(max_workers=1)
    segment_updates = 0
    while time.monotonic() < deadline and not interrupted:
        last_loss, grad = w.update(model, ema, opt, stream, ref, cfg)
        step += 1
        segment_updates += 1
        last_update_epoch = time.time()
        losses.append(last_loss)
        now = time.monotonic()
        if now - last_status >= 10 or step % 1000 == 0:
            w.dump(directory / 'status.json', dict(state='training', pid=os.getpid(), step=step,
                start_step=task['start_step'], updates=step-task['start_step'],
                elapsed_seconds=now-begin, last_loss=last_loss, gradient_norm=grad,
                deadline_epoch=timing['deadline_epoch'], first_step_epoch=first_step_epoch,
                last_update_epoch=last_update_epoch, resumed=True, resume_epoch=resume_epoch, time=time.time()))
            last_status = now
        if step % 1000 == 0:
            w.append_csv(directory / 'training.csv', dict(step=step, mean_loss=float(np.mean(losses)),
                last_loss=last_loss, gradient_norm=grad, elapsed_seconds=now-begin, time=time.time()))
            losses = []
        if pending is not None and pending.done():
            pending.result()
            pending = None
        if step % cfg['checkpoint_every'] == 0:
            if pending is not None:
                pending.result()
            payload = w.snapshot(model, ema, opt, stream, step, cfg, task, timing, last_loss)
            payload['extension_request'] = request
            payload['resumed_from_checkpoint'] = previous['checkpoint']
            pending = writer.submit(w.commit_checkpoint, directory, f'checkpoint_step_{step:09d}.pt', payload)
    torch.cuda.synchronize()
    stopped_epoch = time.time()
    elapsed = time.monotonic() - begin
    stop_reason = interrupted[-1] if interrupted else 'extended_deadline'
    w.dump(directory / 'status.json', dict(state='training_stopped_saving', pid=os.getpid(),
        step=step, stopped_epoch=stopped_epoch, stop_reason=stop_reason, elapsed_seconds=elapsed))
    if pending is not None:
        pending.result()
    transition_seconds = resume_epoch - previous['training_stopped_epoch']
    final = w.snapshot(model, ema, opt, stream, step, cfg, task, timing, last_loss)
    final['extension_request'] = request
    final['resumed_from_checkpoint'] = previous['checkpoint']
    final['timing'].update(first_step_epoch=first_step_epoch, last_update_epoch=last_update_epoch,
        stopped_epoch=stopped_epoch, worker_elapsed_seconds=elapsed,
        authorized_preview_pause_seconds=previous['authorized_preview_pause_seconds'],
        extension_transition_seconds=transition_seconds, extension_resume_epoch=resume_epoch)
    metadata = w.commit_checkpoint(directory, f'final_step_{step:09d}.pt', final)
    del final
    writer.shutdown(wait=True)
    if losses:
        w.append_csv(directory / 'training.csv', dict(step=step, mean_loss=float(np.mean(losses)),
            last_loss=last_loss, gradient_norm=grad, elapsed_seconds=elapsed, time=time.time()))
    w.evaluate(model, ema, train_cpu, other_cpu, directory, 'after', step, task, panel_info)
    complete = dict(state='complete' if not interrupted else 'interrupted', key=task['key'],
        start_step=task['start_step'], final_step=step, updates=step-task['start_step'],
        common_start_epoch=timing['start_epoch'], deadline_epoch=timing['deadline_epoch'],
        first_step_epoch=first_step_epoch, last_update_epoch=last_update_epoch,
        training_stopped_epoch=stopped_epoch, worker_elapsed_seconds=elapsed,
        stop_reason=stop_reason, checkpoint=metadata, finished_at=time.time(),
        authorized_preview_pause_seconds=previous['authorized_preview_pause_seconds'],
        extension_transition_seconds=transition_seconds, extension_resume_epoch=resume_epoch,
        resumed_from_step=previous['final_step'], resumed_segment_updates=segment_updates,
        user_authorized_preview_pause=True, optimizer_reset_on_resume=False,
        user_authorized_extension_seconds=3600)
    w.dump(directory / 'complete.json', complete)
    w.dump(directory / 'status.json', complete)
    print('RESUMED_COMPLETE', json.dumps(complete), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        try:
            root_arg = Path(sys.argv[sys.argv.index('--root') + 1])
            key_arg = sys.argv[sys.argv.index('--key') + 1]
            w.dump(root_arg / 'extension_01' / key_arg / 'resume_error.json', dict(
                pid=os.getpid(), time=time.time(), traceback=traceback.format_exc()))
        finally:
            raise
