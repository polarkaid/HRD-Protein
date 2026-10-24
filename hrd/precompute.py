"""Frozen local ProtT5 encoder: singleton FP32 forward, all residues retained."""
import csv
import json
import re
from pathlib import Path
import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer


@torch.inference_mode()
def precompute(data, folder, model_path, device):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    records = data.sequences()
    offsets = np.r_[0, np.cumsum([len(s) for _, s in records])].astype(np.int64)
    if (folder / 'done.npy').exists():
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
        (folder / 'manifest.json').write_text(json.dumps(dict(model=str(model_path), dtype='float32',
            dimension=1024, preprocessing='UZOB to X; all residues; exclude EOS and padding'), indent=2))
    if done.all():
        print('Residue cache already complete.', flush=True)
        return
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False, local_files_only=True)
    model = T5EncoderModel.from_pretrained(model_path, torch_dtype=torch.float32,
                                         local_files_only=True).eval().to(device)
    model.requires_grad_(False)
    for i, (_, sequence) in enumerate(records):
        if done[i]:
            continue
        sequence = re.sub('[UZOB]', 'X', sequence)
        tokens = tokenizer([' '.join(sequence)], add_special_tokens=True, return_tensors='pt')
        hidden = model(**{k: v.to(device) for k, v in tokens.items()}).last_hidden_state
        x[offsets[i]:offsets[i+1]] = hidden[0, :len(sequence)].cpu().numpy()
        # Publish completion only after the residue values have reached the file.
        x.flush()
        done[i] = True
        done.flush()
        if i == 0 or (i+1) % 100 == 0 or i+1 == len(records):
            print(f'Cached {i+1}/{len(records)} sequences', flush=True)
