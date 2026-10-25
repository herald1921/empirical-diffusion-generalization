"""Exhaustive pixel-RMSE search without norm-expansion cancellation.

All reference rows are visited. Subtraction and reduction use float64 via
cdist's direct Euclidean kernel; matrix-multiplication expansion is disabled.
Returned distances are float64. Equal distances choose the first global row.
"""
import math
import torch


@torch.inference_mode()
def nearest_rmse(q, ref, chunk=4096, query_chunk=32):
    if q.ndim != 2 or ref.ndim != 2 or q.shape[1] != ref.shape[1]:
        raise ValueError('Expected query/reference matrices with equal feature dimensions')
    if not len(ref) or not q.shape[1] or chunk < 1 or query_chunk < 1:
        raise ValueError('Reference, feature dimension and chunk sizes must be nonempty/positive')
    if q.device != ref.device:
        raise ValueError('Query and reference must use the same device')
    if not torch.isfinite(q).all() or not torch.isfinite(ref).all():
        raise ValueError('Nearest-neighbor inputs must be finite')
    best = torch.full((len(q),), float('inf'), device=q.device, dtype=torch.float64)
    index = torch.full((len(q),), -1, device=q.device, dtype=torch.long)
    scale = math.sqrt(q.shape[1])
    for r0 in range(0, len(ref), chunk):
        r = ref[r0:r0 + chunk].to(torch.float64)
        for q0 in range(0, len(q), query_chunk):
            stop = min(q0 + query_chunk, len(q))
            x = q[q0:stop].to(torch.float64)
            distance = torch.cdist(x, r, p=2, compute_mode='donot_use_mm_for_euclid_dist') / scale
            value, local_index = distance.min(dim=1)
            wins = value < best[q0:stop]
            best[q0:stop] = torch.where(wins, value, best[q0:stop])
            index[q0:stop] = torch.where(wins, local_index + r0, index[q0:stop])
    return best, index
