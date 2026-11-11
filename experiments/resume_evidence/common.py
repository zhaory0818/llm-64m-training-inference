"""Shared reproducible assets and isolated output for efficiency experiments."""
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
POINTER = ROOT / 'docs/learning/generated/resume_evidence/latest_run.json'
WEIGHTS = {'base': ('out/full_sft_768.pth', '2334de2f14733fffc4ef4f71fb1d318d5940a1a295bcdfb0549e49d17538946f'),
    'adapter': ('out/lora_medical_768.pth', 'a81c64d452367746d1d65297d154d5edb66d049bea01e806dd80b0cadb2e00c9'),
    'merged': ('out/merge_medical_768.pth', '5423859cd89922f91c12935e1570f5c494e4c0ffb91102b7e94dc2cb2e4bbf65')}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def run_dir():
    return Path(read(POINTER)['folder'])


def initialize_torch():
    os.environ.update(TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
        HF_HOME=str(ROOT / '.cache/huggingface'), CUBLAS_WORKSPACE_CONFIG=':4096:8')
    import datasets  # Windows Arrow must load before torch.
    import torch
    assert Path(sys.prefix).resolve() == (ROOT / '.venv').resolve()
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.80)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    return torch


def load_model(kind='base', dtype=None, manual_attention=False):
    import torch
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from model.model_lora import apply_lora, load_lora
    key = 'merged' if kind == 'merged' else 'base'
    path, expected = WEIGHTS[key]
    assert sha(ROOT / path) == expected
    model = MiniMindForCausalLM(MiniMindConfig(flash_attn=not manual_attention))
    model.load_state_dict(torch.load(ROOT / path, map_location='cpu', weights_only=True), strict=True)
    if kind == 'lora':
        assert sha(ROOT / WEIGHTS['adapter'][0]) == WEIGHTS['adapter'][1]
        apply_lora(model)
        load_lora(model, str(ROOT / WEIGHTS['adapter'][0]))
    return model.to(dtype=dtype or torch.float32).eval()


def verify_sources(folder):
    records = read(folder / 'protected_before.json')
    for rel, expected in records.items():
        assert sha(ROOT / rel) == expected, rel
    return True


def environment(torch):
    return {'at': datetime.now().astimezone().isoformat(), 'python': sys.version,
        'packages': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'datasets', 'numpy']},
        'gpu': torch.cuda.get_device_name(), 'gpu_bytes': torch.cuda.get_device_properties(0).total_memory,
        'num_threads': torch.get_num_threads(), 'tf32': False,
        'nvidia_smi': subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw',
            '--format=csv,noheader'], text=True).strip()}
