"""HRD training, checkpoint selection, embedding and evaluation workflow."""
from pathlib import Path
import random

from .paths import initialize, save_json

initialize()

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


def save_checkpoint(path, value):
    path = Path(path)
    partial = path.with_name(path.name + '.tmp')
    torch.save(value, partial)
    partial.replace(path)


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


def train(data, cache, cfg, seed, output, resume=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'best.pt').exists() and not resume:
        raise FileExistsError('Choose a new output directory to preserve the existing run')
    setup(seed)
    model = HRD(seed).to(cache.device)
    objective = Objective(cfg['centers'], cfg['temperature'], cfg['distance_coefficient']).to(cache.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    sampler = Sampler(data.codes, data.split['train'], data.hashes, seed, cfg['batch_anchors'])
    codes = torch.as_tensor(data.codes, device=cache.device)
    history, best, bad = [], -1.0, 0
    start_epoch = 1
    if resume and (output / 'last.pt').exists():
        state = torch.load(output / 'last.pt', map_location=cache.device, weights_only=True)
        if state['config'] != cfg or state['seed'] != seed:
            raise ValueError('Existing checkpoint has different training settings')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        sampler.main_rng.bit_generator.state = state['main_rng']
        sampler.support_rng.bit_generator.state = state['support_rng']
        torch.set_rng_state(state['torch_rng'].cpu())
        if cache.device.type == 'cuda':
            torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda_rng']])
        history, best, bad = state['history'], state['best'], state['bad']
        start_epoch = state['epoch'] + 1
        print(f'Resuming after epoch {state["epoch"]}.', flush=True)
        if state['finished']:
            save_json(output / 'history.json', history)
            return model
    for epoch in range(start_epoch, cfg['maximum_epochs']+1):
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
            save_checkpoint(output / 'best.pt', dict(model=model.state_dict(), epoch=epoch,
                            selection_mean12=best, config=cfg, seed=seed))
        else:
            bad += 1
        row = dict(epoch=epoch, losses=np.mean(losses, axis=0).tolist(), validation=val, selection_mean12=value)
        history.append(row)
        finished = (epoch == cfg['maximum_epochs'] or
                    (epoch >= cfg['early_stopping_from_epoch'] and bad >= cfg['patience']))
        save_checkpoint(output / 'last.pt', dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
            config=cfg, seed=seed, epoch=epoch, best=best, bad=bad, history=history,
            main_rng=sampler.main_rng.bit_generator.state, support_rng=sampler.support_rng.bit_generator.state,
            torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if cache.device.type == 'cuda' else [],
            finished=finished))
        save_json(output / 'history.json', history)
        print(f'Epoch {epoch}: validation mean12={value:.8f}; best={best:.8f}', flush=True)
        if finished:
            break
    return model
