"""Exhaustive cumulative-positive/same-parent-negative, tie-aware metrics."""
import numpy as np
import torch


def sorted_pair_metrics(positive, negative, chunk=1_000_000, device='cuda'):
    """Exactly the reference stepwise AP and half-credit tied AUROC."""
    p, n = len(positive), len(negative)
    if not p or not n:
        return None, None
    auc = torch.zeros((), dtype=torch.float64, device=device)
    ap = torch.zeros_like(auc)
    start = 0
    while start < p:
        stop = min(start + chunk, p)
        if stop < p:
            stop = int(torch.searchsorted(positive, positive[stop-1], right=True))
        values, counts = torch.unique_consecutive(positive[start:stop], return_counts=True)
        left = torch.searchsorted(negative, values, right=False)
        right = torch.searchsorted(negative, values, right=True)
        c = counts.double()
        auc += (c * (left.double() + right.double()) * .5).sum()
        tp = p - (start + counts.cumsum(0) - counts)
        fp = n - left
        ap += (c * tp.double() / (tp + fp).double()).sum()
        start = stop
    return float(auc) / (p*n), float(ap) / p


def pair_metrics(scores, codes, counts, levels, emit, device='cuda'):
    """Create only the five upper-triangle strata, without quadratic int64 indices."""
    n = len(codes)
    device_codes = torch.as_tensor(codes, device=device, dtype=torch.int32)
    columns = torch.arange(n, device=device)
    banks = [torch.empty(int(v), device=device, dtype=torch.float32) for v in counts]
    positions = [0]*5
    for first in range(0, n, 256):
        last = min(first+256, n)
        block = torch.as_tensor(np.array(scores[first:last], copy=True), device=device)
        depth = torch.zeros(block.shape, device=device, dtype=torch.uint8)
        for level in range(4):
            depth += device_codes[first:last, level, None] == device_codes[None, :, level]
        upper = columns[None, :] > torch.arange(first, last, device=device)[:, None]
        for k in range(5):
            v = block[upper & (depth == k)]
            banks[k][positions[k]:positions[k]+len(v)] = v
            positions[k] += len(v)
    assert positions == list(map(int, counts))
    del block, depth, upper, device_codes, columns
    rows = []
    for level in levels:
        positive = torch.sort(torch.cat(banks[level+1:])).values
        negative = torch.sort(banks[level]).values
        auc, ap = sorted_pair_metrics(positive, negative, device=device)
        rows.append(dict(level_index=level, positive_count=len(positive), negative_count=len(negative), AUROC=auc, AP=ap))
        emit('pair_level', level_index=level, AUROC=auc, AP=ap)
        del positive, negative
    del banks
    return rows


def query_metrics(scores, codes, levels, emit, batch=96, device='cuda'):
    """Sort each query once; preserve score ties after parent/self filtering."""
    n = len(codes)
    labels = torch.as_tensor(codes, device=device, dtype=torch.int32)
    positions = torch.arange(n, device=device, dtype=torch.int32)[None, :]
    results = {level: np.full(n, np.nan, dtype=np.float64) for level in levels}
    pair_counts = {level: [0, 0] for level in levels}
    for first in range(0, n, batch):
        last = min(first+batch, n)
        x = torch.as_tensor(np.array(scores[first:last], copy=True), device=device)
        values, order = torch.sort(x, dim=1, descending=True)
        ends = torch.ones_like(values, dtype=torch.bool)
        ends[:, :-1] = values[:, :-1] != values[:, 1:]
        next_end = torch.cummin(torch.where(ends, positions, n-1).flip(1), dim=1).values.flip(1).long()
        not_self = order != torch.arange(first, last, device=device)[:, None]
        for level in levels:
            valid = not_self if level == 0 else not_self & (labels[order, level-1] == labels[first:last, level-1, None])
            positive = valid & (labels[order, level] == labels[first:last, level, None])
            tp = positive.cumsum(1, dtype=torch.int32)
            seen = valid.cumsum(1, dtype=torch.int32)
            p = tp[:, -1]
            neg = seen[:, -1] - p
            eligible = (p > 0) & (neg > 0)
            pair_counts[level][0] += int(p.sum())
            pair_counts[level][1] += int(neg.sum())
            precision_at_end = tp.gather(1, next_end).double() / seen.gather(1, next_end).clamp_min(1).double()
            ap = (precision_at_end * positive).sum(1) / p.clamp_min(1).double()
            ap[~eligible] = torch.nan
            results[level][first:last] = ap.cpu().numpy()
        if first == 0 or last == n or (first//batch) % 50 == 0:
            emit('query_progress', completed=last, total=n)
    return results, pair_counts


def all_metrics(scores, codes, levels=range(4), emit=lambda *a, **k: None, device='cuda'):
    n = len(codes)
    within = []
    for k in range(4):
        counts = np.unique(codes[:, k], return_counts=True)[1]
        within.append(int((counts*(counts-1)//2).sum()))
    strata = [n*(n-1)//2-within[0], *[within[k]-within[k+1] for k in range(3)], within[3]]
    rows = pair_metrics(scores, codes, strata, levels, emit, device=device)
    aps, counted = query_metrics(scores, codes, levels, emit, device=device)
    for row in rows:
        k = row['level_index']; a = aps[k]; good = np.isfinite(a)
        row.update(query_macro_AP=float(a[good].mean()) if good.any() else None,
                   eligible_queries=int(good.sum()), ineligible_queries=int((~good).sum()))
    return rows, aps
