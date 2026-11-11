"""Portable native Dense inference/API entry for this MiniMind reproduction project."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=Path('out/full_sft_768.pth'))
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--prompt', default='请介绍一下自己。')
    parser.add_argument('--mode', choices=['chat', 'pretrain'], default='chat')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--last-logits', action='store_true', help='Project only the last hidden position during generation.')
    parser.add_argument('--cached-sdpa', action='store_true', help='Use guarded SDPA for unmasked single-token cached decode.')
    parser.add_argument('--port', type=int, default=8998)
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 512:
        parser.error('Demo output budget must be between 1 and 512 new tokens.')
    if not 1024 <= args.port <= 65535:
        parser.error('Port must be between 1024 and 65535.')
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    if not checkpoint.is_file():
        parser.error(f'Checkpoint missing: {checkpoint}. See docs/project/REPRODUCE.md; weights are not bundled with source.')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
    sys.path.insert(0, str(ROOT))
    import datasets  # Windows Arrow DLL compatibility: load before torch.
    import torch
    from transformers import AutoTokenizer
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
    from trainer.trainer_utils import setup_seed

    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device == 'auto':
        device = 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable.')
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model', local_files_only=True)
    model = MiniMindForCausalLM(MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False))
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    if args.adapter:
        from model.model_lora import apply_lora, load_lora
        adapter = args.adapter if args.adapter.is_absolute() else ROOT / args.adapter
        if not adapter.is_file():
            parser.error(f'Adapter missing: {adapter}')
        apply_lora(model)
        load_lora(model, str(adapter))
    model = model.half().to(device) if device == 'cuda' else model.float()
    model.eval()
    from model.inference_options import configure_inference
    configure_inference(model, last_logits=args.last_logits, cached_sdpa=args.cached_sdpa)
    with checkpoint.open('rb') as stream:
        fingerprint = hashlib.file_digest(stream, 'sha256').hexdigest()
    print(json.dumps({'checkpoint': str(checkpoint), 'sha256': fingerprint,
        'device': device, 'parameters': sum(p.numel() for p in model.parameters()),
        'strict_loaded': True, 'training_started': False,
        'last_logits': args.last_logits, 'cached_sdpa': args.cached_sdpa}, ensure_ascii=False), flush=True)
    if args.check_only:
        return
    if args.serve:
        if args.mode != 'chat':
            parser.error('API demo requires chat mode.')
        with socket.socket() as probe:
            if probe.connect_ex(('127.0.0.1', args.port)) == 0:
                parser.error('Port already in use; existing service preserved.')
        import uvicorn
        import scripts.serve_openai_api as original
        original.model, original.tokenizer, original.device = model, tokenizer, device

        @original.app.get('/health')
        async def health():
            return {'status': 'ready', 'model': 'minimind-reproduction', 'sha256': fingerprint,
                'last_logits': args.last_logits, 'cached_sdpa': args.cached_sdpa}

        print(f'Local API: http://127.0.0.1:{args.port}/v1 ; Ctrl+C stops this process.', flush=True)
        uvicorn.run(original.app, host='127.0.0.1', port=args.port, log_level='info')
        return
    prompt = (tokenizer.bos_token + args.prompt if args.mode == 'pretrain'
        else tokenizer.apply_chat_template([{'role': 'user', 'content': args.prompt}],
            tokenize=False, add_generation_prompt=True, open_thinking=False))
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').to(device)
    if inputs.input_ids.shape[1] > 640:
        parser.error('Demo input exceeds 640 tokens; shorten it explicitly.')
    setup_seed(args.seed)
    with torch.inference_mode():
        output = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
            max_new_tokens=args.max_new_tokens, do_sample=args.sample,
            temperature=.85 if args.sample else 1., top_p=.95 if args.sample else 1.,
            top_k=50 if args.sample else 0, repetition_penalty=1.,
            eos_token_id=tokenizer.eos_token_id, use_cache=True)
    new = output[0, inputs.input_ids.shape[1]:].tolist()
    print(json.dumps({'response': tokenizer.decode(new, skip_special_tokens=True),
        'new_tokens': len(new), 'finish_reason': 'eos' if new and new[-1] == 2 else 'length',
        'seed': args.seed, 'sample': args.sample}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
