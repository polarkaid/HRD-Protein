"""HRD training, checkpoint selection, embedding and evaluation workflow."""
import argparse
import json
import os
from pathlib import Path
import random

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import numpy as np
import torch
from .data import Dataset, Residues
from .model import HRD
from .loss import Objective
from .sampling import Sampler, unpack
from .metrics import all_metrics

METRICS = ('AUROC', 'AP', 'query_macro_AP')


def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf8')


def forward(model, cache, ids):
    x, valid = cache.batch(ids)
    with torch.autocast(cache.device.type, dtype=torch.bfloat16, enabled=cache.device.type == 'cuda'):
        return model(x, valid)


@torch.inference_mode()
def encode(model, cache, ids, batch_size=128):
    model.eval()
    vectors = np.empty((len(ids), 512), np.float32)
    order = np.argsort(cache.lengths[ids], kind='stable')
    for start in range(0, len(order), batch_size):
        selected = order[start:start+batch_size]
        vectors[selected] = forward(model, cache, ids[selected]).cpu().numpy()
    return vectors


def evaluate(model, cache, data, cfg, partition, output=None):
    ids = data.split[partition]
    z = encode(model, cache, ids, cfg.get('inference_batch_size', 128))
    # Preserve the exact FP32 cosine scoring path used for the paper tables.
    scores = z @ z.T
    norm = np.linalg.norm(z, axis=1)
    scores /= norm[:, None] * norm[None, :]
    scores = (scores + scores.T) / 2
    rows, aps = all_metrics(scores, data.codes[ids], device=str(cache.device))
    for row in rows:
        row['level'] = cfg['levels'][row['level_index']]
    result = dict(dataset=cfg['dataset'], partition=partition, records=len(ids), metrics=rows)
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez(output / 'vectors.npz', vectors=z, indices=ids, ids=data.ids[ids])
        np.savez(output / 'query_AP.npz', **{str(k): v for k, v in aps.items()})
        save_json(output / 'metrics.json', result)
    return result


def train(data, cache, cfg, seed, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'best.pt').exists():
        raise FileExistsError('Choose a new output directory to preserve the existing run')
    setup(seed)
    model = HRD(seed).to(cache.device)
    objective = Objective(cfg['centers'], cfg['temperature'], cfg['distance_coefficient']).to(cache.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    sampler = Sampler(data.codes, data.split['train'], data.hashes, seed, cfg['batch_anchors'])
    codes = torch.as_tensor(data.codes, device=cache.device)
    history, best, bad = [], -1.0, 0
    save_json(output / 'config.json', dict(cfg, seed=seed))
    for epoch in range(1, cfg['maximum_epochs']+1):
        model.train()
        bank, losses = sampler.epoch(), []
        for first in range(0, len(bank), cfg['batch_anchors']):
            ids, a, b, r = unpack(bank[first:first+cfg['batch_anchors']])
            optimizer.zero_grad(set_to_none=True)
            z = forward(model, cache, ids)
            a, b, r = [torch.as_tensor(v, dtype=torch.long, device=cache.device) for v in (a, b, r)]
            total, rank, distance = objective(z, codes[ids], a, b, r)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip_norm'], error_if_nonfinite=True)
            optimizer.step()
            losses.append([float(t.detach()) for t in (total, rank, distance)])
        val = evaluate(model, cache, data, cfg, 'validation')
        value = float(np.mean([r[k] for r in val['metrics'] for k in METRICS]))
        if value > best + cfg['minimum_improvement']:
            best, bad = value, 0
            torch.save(dict(model=model.state_dict(), epoch=epoch, selection_mean12=best,
                            config=cfg, seed=seed), output / 'best.pt')
        else:
            bad += 1
        row = dict(epoch=epoch, losses=np.mean(losses, axis=0).tolist(), validation=val, selection_mean12=value)
        history.append(row)
        save_json(output / 'history.json', history)
        print(f'Epoch {epoch}: validation mean12={value:.8f}; best={best:.8f}', flush=True)
        if epoch >= cfg['early_stopping_from_epoch'] and bad >= cfg['patience']:
            break
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['precompute', 'train', 'embed', 'evaluate'])
    parser.add_argument('--data', required=True, help='External processed dataset directory')
    parser.add_argument('--cache', required=True, help='External ProtT5 residue cache directory')
    parser.add_argument('--config', help='configs/cath40.json or configs/scope40.json')
    parser.add_argument('--output', help='New run or result directory')
    parser.add_argument('--checkpoint', help='best.pt for embed/evaluate')
    parser.add_argument('--plm', help='Local ProtT5-XL UniRef50 model directory for precompute')
    parser.add_argument('--partition', choices=['train', 'validation', 'test'], default='test')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    setup(args.seed)
    data = Dataset(args.data)
    if args.command == 'precompute':
        if not args.plm:
            parser.error('precompute requires --plm')
        from .precompute import precompute
        precompute(data, args.cache, args.plm, args.device)
        return
    if not args.config or not args.output:
        parser.error('train/embed/evaluate require --config and --output')
    cfg = json.loads(Path(args.config).read_text(encoding='utf8'))
    cache = Residues(data, args.cache, args.device)
    if args.command == 'train':
        train(data, cache, cfg, args.seed, args.output)
        return
    if not args.checkpoint:
        parser.error('embed/evaluate require --checkpoint')
    model = HRD().to(cache.device)
    ck = torch.load(args.checkpoint, map_location=cache.device, weights_only=True)
    model.load_state_dict(ck['model'])
    if args.command == 'evaluate':
        result = evaluate(model, cache, data, cfg, args.partition, args.output)
        print(json.dumps(result, indent=2), flush=True)
    else:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        ids = data.split[args.partition]
        np.savez(output / 'vectors.npz', vectors=encode(model, cache, ids, cfg['inference_batch_size']),
                 indices=ids, ids=data.ids[ids])


if __name__ == '__main__':
    main()
