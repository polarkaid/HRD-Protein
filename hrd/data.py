"""Read the external processed split and a ragged, FP32 residue cache."""
import csv
from pathlib import Path
import numpy as np
import torch


def fasta_records(path):
    records, name, pieces = [], None, []
    for line in Path(path).read_text(encoding='utf8').splitlines():
        if line.startswith('>'):
            if name is not None:
                records.append((name, ''.join(pieces)))
            name, pieces = line[1:].split()[0], []
        elif line.strip():
            pieces.append(line.strip())
    if name is not None:
        records.append((name, ''.join(pieces)))
    return records


class Dataset:
    def __init__(self, folder):
        self.folder = Path(folder)
        with np.load(self.folder / 'split.npz', allow_pickle=False) as f:
            self.split = {k: f[k] for k in f.files}
        self.ids, self.codes = self.split['ids'], self.split['codes']
        self.hashes = self.split['sequence_sha256']

    def sequences(self):
        records = fasta_records(self.folder / 'all.fasta')
        by_id = dict(records)
        return [(str(i), by_id[str(i)]) for i in self.ids]


class Residues:
    def __init__(self, data, folder, device='cuda'):
        folder = Path(folder)
        self.device = torch.device(device)
        with (folder / 'domains.tsv').open(encoding='utf8', newline='') as f:
            records = list(csv.DictReader(f, delimiter='\t'))
        lookup = {r['domain_id']: i for i, r in enumerate(records)}
        self.rows = np.array([lookup[str(i)] for i in data.ids], dtype=np.int64)
        # ID lookup handles both active-row caches and historical full-dataset caches.
        self.offsets = np.load(folder / 'offsets.npy', allow_pickle=False)
        self.raw = np.load(folder / 'embeddings.npy', mmap_mode='r', allow_pickle=False)
        done = np.load(folder / 'done.npy', allow_pickle=False)
        if not done[self.rows].all():
            raise ValueError('Finish precompute before using this residue cache')
        self.lengths = np.diff(self.offsets)[self.rows]

    def batch(self, ids):
        length = int(self.lengths[ids].max())
        x = np.zeros((len(ids), length, 1024), dtype=np.float32)
        valid = np.arange(length)[None, :] < self.lengths[ids, None]
        for j, i in enumerate(ids):
            r = self.rows[i]
            x[j, :self.lengths[i]] = self.raw[self.offsets[r]:self.offsets[r+1]]
        # Match the manuscript's BF16 execution copy; keep the stored cache FP32.
        dtype = torch.bfloat16 if self.device.type == 'cuda' else torch.float32
        return (torch.from_numpy(x).to(self.device, dtype=dtype),
                torch.from_numpy(valid).to(self.device))
