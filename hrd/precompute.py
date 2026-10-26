"""Frozen local ProtT5 encoder: singleton FP32 forward, all residues retained."""
import csv
import re
from pathlib import Path
import numpy as np
import torch
from .paths import save_json


def pending_indices(data, folder, indices):
    folder = Path(folder)
    indices = np.asarray(indices, dtype=np.int64)
    if not (folder / 'manifest.json').exists():
        return indices
    with (folder / 'domains.tsv').open(encoding='utf8', newline='') as f:
        records = list(csv.DictReader(f, delimiter='\t'))
    if ([r['domain_id'] for r in records] != data.ids.tolist() or
            [r['sequence_sha256'] for r in records] != data.hashes.tolist()):
        raise ValueError(f'Residue cache belongs to different sequences: {folder}')
    done = np.load(folder / 'done.npy', mmap_mode='r')
    return indices[~done[indices]]


@torch.inference_mode()
def precompute(data, folder, model_path, device, indices=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    records = data.sequences()
    offsets = np.r_[0, np.cumsum([len(s) for _, s in records])].astype(np.int64)
    indices = np.arange(len(records)) if indices is None else np.asarray(indices)
    pending = pending_indices(data, folder, indices)
    if (folder / 'manifest.json').exists():
        x = np.load(folder / 'embeddings.npy', mmap_mode='r+')
        done = np.load(folder / 'done.npy', mmap_mode='r+')
    else:
        x = np.lib.format.open_memmap(folder / 'embeddings.npy', mode='w+',
                                     dtype='float32', shape=(int(offsets[-1]), 1024))
        done = np.lib.format.open_memmap(folder / 'done.npy', mode='w+', dtype='bool', shape=(len(records),))
        done[:] = False
        done.flush()
        np.save(folder / 'offsets.npy', offsets)
        with (folder / 'domains.tsv').open('w', encoding='utf8', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['domain_id', 'sequence_sha256'])
            writer.writerows(zip(data.ids, data.hashes))
        save_json(folder / 'manifest.json', dict(model=str(model_path), dtype='float32',
            dimension=1024, preprocessing='UZOB to X; all residues; exclude EOS and padding'))
    if not len(pending):
        print('Requested residue cache already complete.', flush=True)
        return
    from transformers import T5EncoderModel, T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False, local_files_only=True)
    model = T5EncoderModel.from_pretrained(model_path, torch_dtype=torch.float32,
                                         local_files_only=True).eval().to(device)
    model.requires_grad_(False)
    print(f'Computing {len(pending)} missing residue embeddings on {device}.', flush=True)
    for count, i in enumerate(pending, 1):
        _, sequence = records[i]
        sequence = re.sub('[UZOB]', 'X', sequence)
        tokens = tokenizer([' '.join(sequence)], add_special_tokens=True, return_tensors='pt')
        hidden = model(**{k: v.to(device) for k, v in tokens.items()}).last_hidden_state
        x[offsets[i]:offsets[i+1]] = hidden[0, :len(sequence)].cpu().numpy()
        # Publish completion only after the residue values have reached the file.
        x.flush()
        done[i] = True
        done.flush()
        if count == 1 or count % 100 == 0 or count == len(pending):
            print(f'Cached {count}/{len(pending)} missing sequences', flush=True)
    del model, x, done
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()
