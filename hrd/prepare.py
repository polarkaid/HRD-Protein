"""Official CATHS40/SCOPe40 sequences -> the fixed seed-0 benchmark split."""
from collections import defaultdict
import csv
import hashlib
from pathlib import Path
import shutil
import tarfile

import numpy as np

from .data import fasta_records
from .download import download_file, digest
from .paths import ROOT, save_json

CATH_BASE = 'https://download.cathdb.info/cath/releases/all-releases/v4_4_0/'
CATH_FASTA = 'cath-dataset-nonredundant-S40-v4_4_0.fa'
CATH_LABELS = 'cath-domain-list-v4_4_0.txt'
SCOPE_FASTA = 'astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa'
SOURCES = {
    CATH_FASTA: (CATH_BASE + 'non-redundant-data-sets/' + CATH_FASTA,
                '2b9335c395e94c9bfe51e91d4f4a6147821c859a8430b1a7363a4993be07aa03'),
    CATH_LABELS: (CATH_BASE + 'cath-classification-data/' + CATH_LABELS,
                 '2f87544bb8844ff37ad1d9348dfe0a23a7c9e4837e0d207733935148e088cc81'),
    SCOPE_FASTA: ('https://scop.berkeley.edu/downloads/scopeseq-2.08/' + SCOPE_FASTA,
                 'e6d3213b38225f617966686809fa5531f0de6a7d00d01692477476a1493ab037'),
}
LEVELS = {'cath40': ['class', 'architecture', 'topology', 'superfamily'],
          'scope40': ['class', 'fold', 'superfamily', 'family']}
PARTS = ('train', 'validation', 'test')


def source_file(name, raw, offline=False):
    url, checksum = SOURCES[name]
    try:
        return download_file(url, raw / name, checksum, offline=offline)
    except (RuntimeError, FileNotFoundError):
        if name != SCOPE_FASTA or offline:
            raise
    # The authors' archival release remains available when the SCOPe server is down.
    print('SCOPe server unavailable; using the authors\' Zenodo release (315 MB).', flush=True)
    archive = download_file('https://zenodo.org/records/5829561/files/scopeseq-2.08.tgz?download=1',
                            raw / 'scopeseq-2.08.tgz', 'd595ba9998e3029ede93f8c7f612e228', 'md5')
    path = raw / name
    with tarfile.open(archive, 'r:gz') as tar:
        member = next(m for m in tar if m.isfile() and Path(m.name).name == name)
        with tar.extractfile(member) as src, path.with_suffix('.tmp').open('wb') as out:
            shutil.copyfileobj(src, out)
    if digest(path.with_suffix('.tmp')) != checksum:
        raise ValueError('Archived SCOPe FASTA differs from the fixed release')
    path.with_suffix('.tmp').replace(path)
    return path


def make_records(dataset, raw):
    if dataset == 'cath40':
        labels = {}
        for line in (raw / CATH_LABELS).read_text().splitlines():
            if line.strip() and not line.startswith('#'):
                fields = line.split()
                labels[fields[0]] = fields[1:5]
        sequences = {name.split('|')[-1].split('/')[0]: seq.upper()
                     for name, seq in fasta_records(raw / CATH_FASTA)}
    else:
        labels = {line[1:].split()[0]: line.split()[1].split('.')
                  for line in (raw / SCOPE_FASTA).read_text().splitlines() if line.startswith('>')}
        sequences = {name: seq.upper() for name, seq in fasta_records(raw / SCOPE_FASTA)}
    ids = sorted(sequences, key=lambda name: tuple(int(p) if p.isdigit() else p
                                                   for p in labels[name]) + (name,))
    source_rows = np.array([i for i, name in enumerate(ids)
                           if not (dataset == 'cath40' and labels[name][0] == '6')], dtype=np.int64)
    ids = np.array(ids)[source_rows]
    paths = [['.'.join(labels[name][:k]) for k in range(1, 5)] for name in ids]
    return ids, sequences, np.asarray(paths), source_rows


def split_records(ids, sequences, labels):
    hashes = np.array([hashlib.sha256(sequences[name].encode()).hexdigest() for name in ids])
    groups, strata = defaultdict(list), defaultdict(list)
    for i, h in enumerate(hashes):
        groups[str(h)].append(i)
    for h, indices in groups.items():
        classes = set(labels[indices, 0])
        if len(classes) != 1:
            raise ValueError('Identical input sequences span Classes; cannot reproduce the fixed split')
        strata[next(iter(classes))].append(h)
    rng = np.random.default_rng(0)
    partition = np.full(len(ids), -1, np.int8)
    for _, names in sorted(strata.items()):
        names = np.array(sorted(names))
        names = names[rng.permutation(len(names))]
        raw = np.array([.7, .1, .2]) * len(names)
        sizes = np.floor(raw).astype(int)
        for j in np.argsort(-(raw - sizes), kind='stable')[:len(names) - sizes.sum()]:
            sizes[j] += 1
        start = 0
        for p, size in enumerate(sizes):
            for h in names[start:start + size]:
                partition[groups[str(h)]] = p
            start += size
    codes = np.column_stack([np.unique(labels[:, k], return_inverse=True)[1]
                             for k in range(4)]).astype(np.int32)
    return hashes, codes, partition


def prepare_dataset(dataset, offline=False):
    dest, raw = ROOT / 'data' / dataset, ROOT / 'data' / 'raw'
    if all((dest / name).is_file() for name in ('manifest.json', 'split.npz', 'all.fasta')):
        print(f'{dataset}: processed data already available.', flush=True)
        return dest
    names = [CATH_FASTA, CATH_LABELS] if dataset == 'cath40' else [SCOPE_FASTA]
    for name in names:
        source_file(name, raw, offline)
    ids, sequences, labels, source_rows = make_records(dataset, raw)
    hashes, codes, partition = split_records(ids, sequences, labels)
    parts = {p: np.flatnonzero(partition == k) for k, p in enumerate(PARTS)}
    dest.mkdir(parents=True, exist_ok=True)
    np.savez(dest / 'split.npz', ids=ids, codes=codes, sequence_sha256=hashes,
             source_rows=source_rows, **parts)
    with (dest / 'all.fasta').open('w', encoding='utf8', newline='\n') as f:
        for name in ids:
            f.write(f'>{name}\n{sequences[name]}\n')
    with (dest / 'partitions.tsv').open('w', encoding='utf8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['domain_id', 'partition', *LEVELS[dataset]])
        writer.writerows((name, PARTS[int(p)], *path) for name, p, path in zip(ids, partition, labels))
    save_json(dest / 'manifest.json', dict(dataset=dataset, version='4.4' if dataset == 'cath40' else '2.08',
              split_seed=0, records=len(ids), counts={p: len(ix) for p, ix in parts.items()},
              sources={name: dict(url=SOURCES[name][0], sha256=SOURCES[name][1]) for name in names}))
    print(f'{dataset}: prepared {len(ids)} domains; ' + ', '.join(f'{p}={len(ix)}' for p, ix in parts.items()), flush=True)
    return dest
