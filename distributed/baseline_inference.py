"""
Baseline Inference — Single Machine (No Disaggregation)

Runs full prefill + decode on ONE machine for comparison against
the disaggregated setup.

Run on M5:
    python distributed/baseline_inference.py --machine M5

Run on M4 Pro:
    python distributed/baseline_inference.py --machine M4_Pro
"""

import argparse
import json
import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str, dtype=torch.float16):
    """Load model onto MPS."""
    print(f"[Baseline] Loading {model_name} in {dtype}...")
    t0 = time.perf_counter()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="mps",
    )
    model.eval()
    
    load_time = time.perf_counter() - t0
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[Baseline] Loaded: {param_count:.1f}B params in {load_time:.1f}s")
    
    return model, tokenizer, load_time


def run_full_inference(model, tokenizer, prompt: str, max_new_tokens: int = 128, device: str = "mps"):
    """
    Run complete inference (prefill + decode) on a single machine.
    Measures prefill and decode separately for apples-to-apples comparison.
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]
    print(f"[Baseline] Prompt: {input_len} tokens, generating up to {max_new_tokens} tokens")
    
    # ---- PREFILL PHASE ----
    torch.mps.synchronize()
    t_prefill_start = time.perf_counter()
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            use_cache=True,
            return_dict=True,
        )
    
    torch.mps.synchronize()
    t_prefill_end = time.perf_counter()
    prefill_time = t_prefill_end - t_prefill_start
    
    # First token
    next_logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(next_logits, dim=-1, keepdim=True)
    kv_cache = outputs.past_key_values
    
    print(f"[Baseline] Prefill: {prefill_time*1000:.1f}ms")
    
    # ---- DECODE PHASE ----
    generated_ids = [next_token_id.item()]
    per_token_times = []
    current_token = next_token_id
    
    torch.mps.synchronize()
    t_decode_start = time.perf_counter()
    
    for i in range(max_new_tokens - 1):  # -1 because we already have first token
        torch.mps.synchronize()
        t_tok_start = time.perf_counter()
        
        with torch.no_grad():
            outputs = model(
                input_ids=current_token,
                past_key_values=kv_cache,
                use_cache=True,
                return_dict=True,
            )
        
        torch.mps.synchronize()
        t_tok_end = time.perf_counter()
        per_token_times.append((t_tok_end - t_tok_start) * 1000)
        
        next_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_logits, dim=-1, keepdim=True)
        
        generated_ids.append(next_token_id.item())
        kv_cache = outputs.past_key_values
        current_token = next_token_id
        
        if next_token_id.item() == tokenizer.eos_token_id:
            print(f"[Baseline] EOS at token {i+2}")
            break
        
        if (i + 2) % 32 == 0:
            elapsed = time.perf_counter() - t_decode_start
            tps = (i + 2) / elapsed
            print(f"[Baseline] Token {i+2}/{max_new_tokens} ({tps:.1f} tok/s)")
    
    torch.mps.synchronize()
    t_decode_end = time.perf_counter()
    
    decode_time = t_decode_end - t_decode_start
    generated_tokens = len(generated_ids)
    tokens_per_sec = generated_tokens / decode_time if decode_time > 0 else 0
    
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    ttft = prefill_time * 1000 + per_token_times[0] if per_token_times else prefill_time * 1000
    total_e2e = (t_decode_end - t_prefill_start)
    
    return {
        "prefill_time_ms": prefill_time * 1000,
        "decode_time_ms": decode_time * 1000,
        "generated_tokens": generated_tokens,
        "tokens_per_sec": tokens_per_sec,
        "ttft_ms": ttft,
        "total_e2e_ms": total_e2e * 1000,
        "avg_token_latency_ms": sum(per_token_times) / len(per_token_times) if per_token_times else 0,
        "per_token_times_ms": per_token_times[:10],
        "generated_text": generated_text[:500],
        "prompt_tokens": input_len,
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline Inference (Single Machine)")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--prompt", default="Explain the theory of general relativity in simple terms. Start with")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--machine", default="unknown", help="Machine name (M5 or M4_Pro)")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--runs", type=int, default=3, help="Number of runs to average")
    args = parser.parse_args()
    
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    # Load model
    model, tokenizer, load_time = load_model(args.model, dtype)
    
    # Warmup run
    print(f"\n[Baseline] Warmup run...")
    _ = run_full_inference(model, tokenizer, args.prompt, max_new_tokens=8)
    
    # Benchmark runs
    all_results = []
    for run_idx in range(args.runs):
        print(f"\n{'='*60}")
        print(f"[Baseline] Run {run_idx + 1}/{args.runs} on {args.machine}")
        print(f"{'='*60}")
        
        result = run_full_inference(model, tokenizer, args.prompt, args.max_tokens)
        all_results.append(result)
        
        print(f"  Prefill:     {result['prefill_time_ms']:.1f}ms")
        print(f"  Decode:      {result['decode_time_ms']:.1f}ms ({result['generated_tokens']} tokens)")
        print(f"  Tok/s:       {result['tokens_per_sec']:.1f}")
        print(f"  TTFT:        {result['ttft_ms']:.1f}ms")
        print(f"  Total E2E:   {result['total_e2e_ms']:.1f}ms")
    
    # Average results
    avg = {}
    for key in ['prefill_time_ms', 'decode_time_ms', 'tokens_per_sec', 'ttft_ms', 
                 'total_e2e_ms', 'avg_token_latency_ms', 'generated_tokens']:
        values = [r[key] for r in all_results]
        avg[key] = sum(values) / len(values)
    
    print(f"\n{'='*60}")
    print(f"[Baseline] AVERAGED RESULTS ({args.runs} runs) on {args.machine}:")
    print(f"{'='*60}")
    print(f"  Prefill:     {avg['prefill_time_ms']:.1f}ms")
    print(f"  Decode:      {avg['decode_time_ms']:.1f}ms ({avg['generated_tokens']:.0f} tokens)")
    print(f"  Tok/s:       {avg['tokens_per_sec']:.1f}")
    print(f"  TTFT:        {avg['ttft_ms']:.1f}ms")
    print(f"  Total E2E:   {avg['total_e2e_ms']:.1f}ms")
    print(f"  Avg/token:   {avg['avg_token_latency_ms']:.2f}ms")
    print(f"{'='*60}")
    
    # Save
    output = {
        "role": "baseline",
        "machine": args.machine,
        "model": args.model,
        "prompt": args.prompt,
        "max_new_tokens": args.max_tokens,
        "dtype": args.dtype,
        "num_runs": args.runs,
        "load_time_s": load_time,
        "averaged": avg,
        "all_runs": all_results,
        "generated_text": all_results[-1]['generated_text'],
    }
    
    os.makedirs("results/raw/distributed", exist_ok=True)
    output_file = f"results/raw/distributed/baseline_{args.machine}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[Baseline] Results saved to {output_file}")


if __name__ == "__main__":
    main()
