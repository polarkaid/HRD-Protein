"""Project-local preparation, repeated training and released-checkpoint testing."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .data import Dataset, Residues
from .download import digest, ensure_plm, ensure_weights, PLM_REVISION
from .model import HRD
from .paths import ROOT, save_json
from .precompute import pending_indices, precompute
from .prepare import prepare_dataset
from .run import METRICS, evaluate, setup, train


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def run_settings(data, cfg, device):
    """Keep reused results tied to the same data, parameters and precision."""
    return dict(split_sha256=digest(data.folder / 'split.npz'), config=cfg,
                precision='bf16-fp32' if torch.device(device).type == 'cuda' else 'fp32',
                plm_revision=PLM_REVISION)


def csv_rows(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.tmp')
    with partial.open('w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    partial.replace(path)


def prepare(dataset, mode, device, offline=False):
    folder = prepare_dataset(dataset, offline)
    data = Dataset(folder)
    needed = np.arange(len(data.ids)) if mode == 'train' else data.split['test']
    residue_dir = ROOT / 'cache' / 'residues' / f'prott5-{PLM_REVISION[:8]}' / dataset
    pending = pending_indices(data, residue_dir, needed)
    if len(pending):
        sequences = data.sequences()
        gib = sum(len(s) for _, s in sequences) * 1024 * 4 / 1024**3
        print(f'{dataset}: {len(pending)} missing sequences; full FP32 cache allocation {gib:.1f} GiB.', flush=True)
        model_path = ensure_plm(offline)
        precompute(data, residue_dir, model_path, device, indices=needed)
    else:
        print(f'{dataset}: reusing residue embeddings; no PLM download needed.', flush=True)
    return data, Residues(data, residue_dir, device, required=needed)


def load_model(checkpoint, device):
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = HRD().to(device)
    model.load_state_dict(saved['model'])
    return model, saved


def test_checkpoint(data, cache, cfg, checkpoint, output, settings):
    output = Path(output)
    checkpoint_hash = digest(checkpoint)
    status = output / 'metrics.json'
    if status.exists():
        previous = read_json(status)
        if previous.get('settings') == settings and previous.get('checkpoint_sha256') == checkpoint_hash:
            print(f'{cfg["dataset"]}: reusing completed test at {output}', flush=True)
            return previous
    setup(0)
    model, saved = load_model(checkpoint, cache.device)
    saved_dataset = saved.get('config', {}).get('dataset')
    if saved_dataset and saved_dataset != cfg['dataset']:
        raise ValueError(f'Checkpoint is for {saved_dataset}, not {cfg["dataset"]}')
    print(f'{cfg["dataset"]}: evaluating {Path(checkpoint).name} on the test partition.', flush=True)
    result = evaluate(model, cache, data, cfg, 'test', output)
    result.update(settings=settings, checkpoint_sha256=checkpoint_hash,
                  selected_epoch=saved.get('epoch'), validation_mean12=saved.get('selection_mean12'))
    csv_rows(output / 'metrics.csv', result['metrics'], ['level', *METRICS])
    save_json(status, result)
    for row in result['metrics']:
        values = ', '.join(f'{key}={row[key]:.6f}' if row[key] is not None else f'{key}=NA' for key in METRICS)
        print(f'  {row["level"]}: {values}', flush=True)
    del model
    if cache.device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def archive_run(folder):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    target = ROOT / 'runs' / 'archive' / f'{folder.parent.name}_{folder.name}_{stamp}'
    target.parent.mkdir(parents=True, exist_ok=True)
    folder.rename(target)
    print(f'Previous run preserved at {target}', flush=True)


def train_seed(data, cache, cfg, seed, settings, force=False):
    output = ROOT / 'runs' / 'train' / cfg['dataset'] / f'seed{seed}'
    if force and output.exists():
        archive_run(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / 'run.json'
    if metadata.exists():
        state = read_json(metadata)
        if state['settings'] != settings or state['seed'] != seed:
            raise ValueError(f'{output} uses different data/settings. Use --force-retrain to start a new run.')
    else:
        if any((output / name).exists() for name in ('best.pt', 'last.pt')):
            raise ValueError(f'{output} contains an older run. Use --force-retrain to preserve it and start again.')
        state = dict(settings=settings, seed=seed, training_complete=False)
        save_json(metadata, state)
    if state['training_complete'] and (output / 'best.pt').exists():
        print(f'{cfg["dataset"]} seed {seed}: training already complete; skipping.', flush=True)
    else:
        print(f'{cfg["dataset"]} seed {seed}: training (resume from last completed epoch if available).', flush=True)
        model = train(data, cache, cfg, seed, output, resume=True)
        del model
        state['training_complete'] = True
        save_json(metadata, state)
        if cache.device.type == 'cuda':
            torch.cuda.empty_cache()
    return test_checkpoint(data, cache, cfg, output / 'best.pt', output / 'test', settings)


def summarize(dataset, settings):
    folder = ROOT / 'runs' / 'train' / dataset
    completed = {}
    for path in folder.glob('seed*/test/metrics.json'):
        run_dir = path.parent.parent
        if not (run_dir / 'run.json').exists():
            continue
        metadata = read_json(run_dir / 'run.json')
        test_result = read_json(path)
        if metadata['training_complete'] and test_result.get('settings') == settings:
            completed[metadata['seed']] = test_result
    if not completed:
        return
    seeds = sorted(completed)

    def aggregate(selected):
        rows = []
        for level in range(4):
            row = dict(level=completed[selected[0]]['metrics'][level]['level'], n_seeds=len(selected))
            for metric in METRICS:
                values = [completed[seed]['metrics'][level][metric] for seed in selected]
                row[metric + '_mean'] = float(np.mean(values)) if all(v is not None for v in values) else None
                row[metric + '_std'] = (float(np.std(values, ddof=1))
                                        if len(values) > 1 and all(v is not None for v in values) else None)
            rows.append(row)
        return rows

    rows = [dict(seed=seed, **row) for seed in seeds for row in completed[seed]['metrics']]
    stats = aggregate(seeds)
    save_json(folder / 'summary.json', dict(dataset=dataset, completed_seeds=seeds,
              per_seed=completed, aggregate=stats,
              five_seed_0_to_4=aggregate(list(range(5))) if set(range(5)) <= completed.keys() else None))
    csv_rows(folder / 'per_seed.csv', rows, ['seed', 'level', *METRICS])
    csv_rows(folder / 'summary.csv', stats,
             ['level', 'n_seeds', *[f'{metric}_{stat}' for metric in METRICS for stat in ('mean', 'std')]])
    print(f'{dataset}: summary updated for completed seeds {seeds}: {folder / "summary.csv"}', flush=True)


def local_checkpoint(path):
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.is_relative_to(ROOT):
        return source
    target = ROOT / 'weights' / 'local' / f'{digest(source)[:12]}_{source.name}'
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = target.with_name(target.name + '.tmp')
        shutil.copyfile(source, temporary)
        temporary.replace(target)
    return target


def main(mode):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['all', 'cath40', 'scope40'], default='all')
    parser.add_argument('--device', default='cuda', help='cuda (default), cuda:0, or cpu')
    parser.add_argument('--offline', action='store_true', help='Reuse local files only; never contact download servers')
    parser.add_argument('--prepare-only', action='store_true', help='Download/prepare data and residues, then stop')
    if mode == 'train':
        parser.add_argument('--seed', type=int, help='Train one seed; default trains seeds 0 through 4')
        parser.add_argument('--force-retrain', action='store_true', help='Archive existing selected runs and train from scratch')
    else:
        parser.add_argument('--checkpoint', help='Local weights; requires --dataset cath40 or scope40')
    args = parser.parse_args()
    if mode == 'train' and args.seed is not None and args.seed < 0:
        parser.error('--seed must be nonnegative')
    if mode == 'test' and args.checkpoint and args.dataset == 'all':
        parser.error('Choose --dataset cath40 or scope40 when supplying --checkpoint')
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Install the CUDA PyTorch build from README, or explicitly use --device cpu.')
    setup(0)
    if device.type == 'cuda':
        torch.cuda.set_device(device if device.index is not None else 0)
    datasets = ['cath40', 'scope40'] if args.dataset == 'all' else [args.dataset]
    checkpoint = local_checkpoint(args.checkpoint) if mode == 'test' and args.checkpoint else None
    for dataset in datasets:
        cfg = read_json(ROOT / 'configs' / f'{dataset}.json')
        if mode == 'test' and not args.prepare_only:
            weight = checkpoint if checkpoint else ensure_weights(dataset, args.offline)
        data, cache = prepare(dataset, mode, args.device, args.offline)
        if args.prepare_only:
            continue
        settings = run_settings(data, cfg, args.device)
        if mode == 'train':
            seeds = range(5) if args.seed is None else [args.seed]
            for seed in seeds:
                train_seed(data, cache, cfg, seed, settings, args.force_retrain)
                summarize(dataset, settings)
        else:
            result_dir = ROOT / 'runs' / 'test' / dataset / f'{weight.stem}_{digest(weight)[:12]}'
            test_checkpoint(data, cache, cfg, weight, result_dir, settings)
        del cache
