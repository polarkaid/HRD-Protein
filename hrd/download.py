"""Resumable downloads, with pinned public model versions and local storage."""
import hashlib
from pathlib import Path
import time

from .paths import ROOT

PLM_REPO = 'Rostlab/prot_t5_xl_half_uniref50-enc'
PLM_REVISION = '94a6abc029ae13029317b140b7424e012bf8dfbf'
WEIGHT_REPO = 'polarkaid/HRD-Protein'
WEIGHT_REVISION = '31b24045bd9b212f0d96e5f582deaa36725110f3'
WEIGHT_NAMES = {'cath40': 'HRD_CATHS40.pt', 'scope40': 'HRD_SCOPe40.pt'}


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download_file(url, path, checksum=None, algorithm='sha256', offline=False):
    path = Path(path)
    if path.is_file():
        if checksum and digest(path, algorithm) != checksum:
            raise ValueError(f'File differs from the published dataset: {path}')
        return path
    if offline:
        raise FileNotFoundError(f'Offline mode: place the downloaded file at {path}')
    import requests
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.part')
    for attempt in range(3):
        try:
            start = partial.stat().st_size if partial.exists() else 0
            headers = {'Accept-Encoding': 'identity', 'User-Agent': 'HRD-Protein/1.0'}
            if start:
                headers['Range'] = f'bytes={start}-'
            with requests.get(url, headers=headers, stream=True, timeout=(10, 45)) as response:
                if response.status_code == 416:
                    if checksum and digest(partial, algorithm) == checksum:
                        partial.replace(path)
                        return path
                    partial.unlink(missing_ok=True)
                    raise OSError('Server rejected the partial download; retrying from the start')
                response.raise_for_status()
                append = response.status_code == 206 and start > 0
                if append and not response.headers.get('Content-Range', '').startswith(f'bytes {start}-'):
                    raise OSError('Unexpected download range')
                total = int(response.headers.get('Content-Length', 0)) + (start if append else 0)
                received, last = start if append else 0, time.monotonic()
                print(f'Downloading {path.name} ({received / 1e6:.1f}/{total / 1e6:.1f} MB)', flush=True)
                with partial.open('ab' if append else 'wb') as f:
                    for block in response.iter_content(1024 * 1024):
                        f.write(block)
                        received += len(block)
                        if time.monotonic() - last >= 5:
                            print(f'  {path.name}: {received / 1e6:.1f}/{total / 1e6:.1f} MB', flush=True)
                            last = time.monotonic()
                if total and partial.stat().st_size != total:
                    raise OSError('Incomplete download')
            if checksum and digest(partial, algorithm) != checksum:
                partial.unlink()
                raise OSError('Downloaded file checksum differs from the fixed release')
            partial.replace(path)
            return path
        except (requests.RequestException, OSError) as exc:
            if attempt == 2:
                raise RuntimeError(f'Could not download {url}. Partial data remain at {partial}; rerun to retry. {exc}') from exc
            print(f'Download interrupted; retry {attempt + 1}/2: {exc}', flush=True)
            time.sleep(1)


def hub_files(repo, revision, names, folder, offline=False):
    folder = Path(folder)
    missing = [name for name in names if not (folder / name).is_file()]
    if not missing:
        return folder
    if offline:
        raise FileNotFoundError(f'Offline mode: missing files in {folder}: {missing}')
    from huggingface_hub import hf_hub_download
    for name in missing:
        print(f'Downloading {repo}/{name}', flush=True)
        # The Hub client already retries requests and resumes partial downloads.
        hf_hub_download(repo, name, revision=revision, local_dir=folder,
                        cache_dir=ROOT / 'cache' / 'huggingface' / 'hub')
    return folder


def ensure_plm(offline=False):
    return hub_files(PLM_REPO, PLM_REVISION,
                     ['config.json', 'pytorch_model.bin', 'spiece.model',
                      'tokenizer_config.json', 'special_tokens_map.json'],
                     ROOT / 'models' / 'prott5', offline)


def ensure_weights(dataset, offline=False):
    name = WEIGHT_NAMES[dataset]
    return hub_files(WEIGHT_REPO, WEIGHT_REVISION, [name],
                     ROOT / 'weights', offline) / name
