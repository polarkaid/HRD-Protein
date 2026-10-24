"""Main-once sampling, with five exclusive support relationships per main."""
import numpy as np


class Sampler:
    def __init__(self, codes, train, sequence_hashes, seed=0, batch_size=64):
        self.codes, self.train = codes, np.asarray(train)
        self.batch_size = batch_size
        self.main_rng, self.support_rng = np.random.default_rng(seed), np.random.default_rng(seed)
        _, self.alias = np.unique(sequence_hashes, return_inverse=True)
        self.duplicates = {}
        _, counts = np.unique(self.alias, return_counts=True)
        for group in np.flatnonzero(counts > 1):
            members = np.flatnonzero(self.alias == group)
            for member in members:
                self.duplicates[int(member)] = members
        self.groups, self.candidates = [], {}
        for depth in range(5):
            key = np.zeros(len(train), np.int64) if depth == 0 else codes[train, depth-1]
            order = np.argsort(key, kind='stable')
            starts = np.r_[0, np.flatnonzero(np.diff(key[order]))+1]
            groups = {int(key[order[s]]): self.train[order[s:t]]
                      for s, t in zip(starts, np.r_[starts[1:], len(order)])}
            self.groups.append(groups)
        for d in range(4):
            for child in self.groups[d+1].values():
                i = int(child[0])
                parent = self.groups[d][0 if d == 0 else int(codes[i, d-1])]
                self.candidates[d, int(codes[i, d])] = parent[codes[parent, d] != codes[i, d]]

    def epoch(self):
        order = self.main_rng.permutation(self.train)
        # Keep the published main-order RNG stream, including the old per-main draws.
        # Supports themselves are drawn below from the independent no-replacement stream.
        for a in order:
            for d in range(4):
                pool = self.candidates[d, int(self.codes[a, d])]
                if len(pool):
                    self.main_rng.integers(len(pool))
            leaf = self.groups[4][int(self.codes[a, 3])]
            if len(leaf) > 1:
                self.main_rng.integers(len(leaf)-1)
        size = self.batch_size
        # Move exact-string aliases into different main batches without dropping records.
        for first in range(0, len(order), size):
            end, seen = min(first+size, len(order)), set()
            for p in range(first, end):
                group = self.alias[order[p]]
                if group in seen:
                    present = set(self.alias[order[first:end]])
                    j = next((j for j in range(end, len(order)) if self.alias[order[j]] not in present), None)
                    if j is None:
                        j = next((j for j in range(first) if self.alias[order[j]] not in present and
                                  group not in self.alias[order[j//size*size:(j//size+1)*size]]), None)
                    if j is None:
                        raise ValueError('Cannot keep identical sequences in separate main batches')
                    order[p], order[j] = order[j], order[p]
                seen.add(self.alias[order[p]])
        bank = np.full((len(order), 6), -1, np.int64)
        bank[:, 0] = order
        for first in range(0, len(order), size):
            group = bank[first:first+size]
            blocked = np.isin(self.alias, self.alias[group[:, 0]])
            for row in group:
                a = row[0]
                for d in range(5):
                    pool = (self.candidates[d, int(self.codes[a, d])] if d < 4
                            else self.groups[4][int(self.codes[a, 3])])
                    chosen = -1
                    if len(pool) > 64:
                        for _ in range(8):
                            candidate = int(pool[self.support_rng.integers(len(pool))])
                            if not blocked[candidate]:
                                chosen = candidate
                                break
                    if chosen < 0:
                        allowed = pool[~blocked[pool]]
                        if len(allowed):
                            chosen = int(allowed[self.support_rng.integers(len(allowed))])
                    if chosen >= 0:
                        row[d+1] = chosen
                        blocked[self.duplicates.get(chosen, chosen)] = True
        return bank


def unpack(groups):
    ids = np.unique(groups[groups >= 0])
    a, b, r = [], [], []
    for depth in range(5):
        keep = groups[:, depth+1] >= 0
        a.extend(np.searchsorted(ids, groups[keep, 0]))
        b.extend(np.searchsorted(ids, groups[keep, depth+1]))
        r.extend([depth] * int(keep.sum()))
    return ids, np.asarray(a), np.asarray(b), np.asarray(r)
