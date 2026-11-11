"""Exercise actual SFT loop, CLI, HTTP and SSE with opt-in features enabled."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from urllib.request import Request, urlopen

from common import ROOT, run_dir, dump, read, sha, initialize_torch, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path)
    args = parser.parse_args()
    folder = args.run_dir or run_dir()
    result = {}
    # The unchanged original train_epoch is exercised with real backward and Adam
    # updates. Only the "main process saves checkpoint" predicate is disabled;
    # all temporary optimization state is discarded, protected files are untouched.
    torch = initialize_torch()
    from torch.utils.data import DataLoader, TensorDataset
    import trainer.train_full_sft as sft
    from dataset.batch_collator import DynamicRightPadding
    payload = torch.load(folder / 'frozen_sft.pt', map_location='cpu', weights_only=True)
    indices = [48, 1, 49, 2]
    ds = TensorDataset(payload['input_ids'][indices], payload['labels'][indices])
    loader = DataLoader(ds, batch_size=2, collate_fn=DynamicRightPadding(multiple=8))
    sft.args = SimpleNamespace(device='cuda', epochs=1, learning_rate=1e-5,
        accumulation_steps=1, grad_clip=1., log_interval=1, save_interval=1000)
    sft.model = load_model().cuda().train()
    sft.optimizer = torch.optim.AdamW(sft.model.parameters(), lr=1e-5)
    sft.scaler = torch.amp.GradScaler('cuda', enabled=False)
    sft.autocast_ctx = torch.autocast('cuda', dtype=torch.bfloat16)
    sft.is_main_process = lambda: False  # Suppress writes, not any training operation.
    before = sft.model.model.layers[0].self_attn.q_proj.weight.detach().clone()
    sft.train_epoch(0, loader, len(loader))
    steps = sorted({int(state['step'].item()) for state in sft.optimizer.state.values()})
    delta = (before - sft.model.model.layers[0].self_attn.q_proj.weight).abs().max().item()
    assert steps == [2] and delta > 0
    result['actual_sft_train_epoch'] = {'adam_steps': steps, 'q_proj_max_abs_delta': delta,
        'batches': 2, 'batch': 2, 'widths': [x.shape[1] for x, _ in loader],
        'checkpoint_saving_suppressed_for_test': True}
    del sft.optimizer, sft.model, before
    torch.cuda.empty_cache()
    command = [sys.executable, '-X', 'utf8', str(ROOT / 'tools/project_demo.py')]
    outputs = []
    for flags in [[], ['--last-logits', '--cached-sdpa']]:
        completed = subprocess.run(command + ['--device', 'cuda', '--max-new-tokens', '16',
            '--prompt', '请介绍一下自己。'] + flags, cwd=ROOT, text=True,
            encoding='utf-8', capture_output=True, timeout=120, check=True)
        parsed = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith('{')]
        assert parsed[0]['strict_loaded'] and parsed[-1]['new_tokens'] > 0
        outputs.append(parsed)
    assert outputs[0][-1] == outputs[1][-1]
    result['cli'] = {'original_and_optimized_greedy_equal': True, 'outputs': outputs}
    port = 18998
    with socket.socket() as probe:
        assert probe.connect_ex(('127.0.0.1', port)) != 0, 'Preserve existing server; choose a free port.'
    log_path = folder / 'integration_api.log'
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command + ['--serve', '--device', 'cuda', '--port', str(port),
            '--last-logits', '--cached-sdpa'], cwd=ROOT, stdout=log, stderr=log)
        try:
            base = f'http://127.0.0.1:{port}'
            deadline = time.monotonic() + 90
            health = None
            while time.monotonic() < deadline:
                assert process.poll() is None, log_path.read_text(encoding='utf-8')
                try:
                    with urlopen(base + '/health', timeout=1) as response:
                        health = json.load(response)
                    break
                except OSError:
                    time.sleep(.5)
            assert health and health['last_logits'] and health['cached_sdpa']
            responses = []
            for stream in [False, True]:
                body = {'model': 'minimind', 'messages': [{'role': 'user', 'content': '你好'}],
                    'max_tokens': 16, 'temperature': .7, 'top_p': .92, 'stream': stream}
                request = Request(base + '/v1/chat/completions',
                    data=json.dumps(body).encode('utf-8'), headers={'Content-Type': 'application/json'})
                with urlopen(request, timeout=90) as response:
                    content = response.read().decode('utf-8')
                    content_type = response.headers.get('Content-Type')
                if stream:
                    events = [json.loads(line[6:]) for line in content.splitlines() if line.startswith('data: ')]
                    assert events and not any('error' in event for event in events)
                    assert events[-1]['choices'][0]['finish_reason'] in ['stop', 'tool_calls']
                    responses.append({'stream': True, 'content_type': content_type, 'events': events})
                else:
                    parsed = json.loads(content)
                    assert parsed['choices'][0]['message']['role'] == 'assistant'
                    responses.append({'stream': False, 'response': parsed})
            result['api'] = {'health': health, 'responses': responses}
        finally:
            # Windows venv launcher may have a child Python server. Stop exactly
            # this test-owned process tree, never a port-wide or name-wide kill.
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                capture_output=True, text=True)
            process.wait(timeout=20)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', port)) != 0:
                break
        time.sleep(.2)
    else:
        raise AssertionError('Test server still listening')
    result['test_server_stopped'] = True
    allowed = {'trainer/train_full_sft.py', 'tools/project_demo.py'}
    audit = []
    for rel, expected in read(folder / 'protected_before.json').items():
        actual = sha(ROOT / rel)
        assert actual == expected or rel in allowed, rel
        audit.append({'path': rel, 'before': expected, 'after': actual,
            'changed': actual != expected, 'intentional_optional_entry_integration': rel in allowed})
    result['protected_assets'] = audit
    result['passed'] = True
    dump(folder / 'integration_result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
