"""All generated data and library caches live inside this checkout."""
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def initialize():
    cache = ROOT / 'cache'
    temporary = cache / 'tmp'
    temporary.mkdir(parents=True, exist_ok=True)
    for key, path in {
        'HF_HOME': cache / 'huggingface',
        'HF_HUB_CACHE': cache / 'huggingface' / 'hub',
        'HUGGINGFACE_HUB_CACHE': cache / 'huggingface' / 'hub',
        'HF_XET_CACHE': cache / 'huggingface' / 'xet',
        'TORCH_HOME': cache / 'torch',
        'XDG_CACHE_HOME': cache,
        'TMP': temporary, 'TEMP': temporary, 'TMPDIR': temporary,
    }.items():
        os.environ[key] = str(path)
    # Let Transformers use HF_HUB_CACHE, even if a global legacy override exists.
    os.environ.pop('TRANSFORMERS_CACHE', None)
    tempfile.tempdir = str(temporary)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.tmp')
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')
    partial.replace(path)
