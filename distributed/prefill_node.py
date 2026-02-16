"""
Prefill Node — Runs on M5 (169.254.1.1)

Loads Mistral-7B, warms up MPS shaders, processes the prompt (prefill phase),
and sends the KV cache to the Decode Node over Thunderbolt.

WARMUP: MPS does lazy shader compilation — the first forward pass for any new
tensor shape compiles Metal kernels (~800ms overhead). We run a warmup pass
before timing to ensure fair measurement.

Usage:
    python distributed/prefill_node.py
    python distributed/prefill_node.py --runs 3
"""

import argparse
import json
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_transfer import (
    KVServer, serialize_kv_from_dynamic_cache, send_kv_cache, M5_IP, KV_PORT
)
from experiment_config import MODEL_NAME, MAX_NEW_TOKENS, NUM_RUNS, LONG_PROMPT

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str = MODEL_NAME, dtype=torch.float16):
    """Load model and tokenizer onto MPS."""
    print(f"[Prefill] Loading {model_name} in {dtype}...")
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
    print(f"[Prefill] Model loaded: {param_count:.1f}B params in {load_time:.1f}s")
    
    return model, tokenizer, load_time


def warmup_model(model, tokenizer, device="mps"):
    """
    Warmup MPS shader compilation.
    
    WHY: MPS lazily compiles Metal GPU kernels the first time it sees a
    forward pass with a given tensor shape. This adds ~800ms of one-time
    overhead that would unfairly inflate the first real measurement.
    
    We run 2 warmup passes:
      1. Short sequence (8 tokens) — compiles basic GEMM/attention kernels
      2. Medium sequence (64 tokens) — compiles for larger attention patterns
    """
    print(f"[Prefill] Warming up MPS shaders...")
    
    # Warmup 1: Short sequence
    short_input = tokenizer("Hello world", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**short_input, use_cache=True, return_dict=True)
    torch.mps.synchronize()
    
    # Warmup 2: Medium sequence
    medium_text = "The transformer architecture introduced self-attention " * 8
    medium_input = tokenizer(medium_text, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**medium_input, use_cache=True, return_dict=True)
    torch.mps.synchronize()
    
    print(f"[Prefill] Warmup complete — MPS shaders compiled")


def run_prefill(model, tokenizer, prompt: str, device: str = "mps"):
    """
    Run prefill phase: encode the prompt and compute KV cache.
    
    Returns:
        kv_cache: The DynamicCache with pre-computed keys/values
        input_ids: The tokenized prompt
        next_token_id: First predicted token
        prefill_time: Time taken for prefill (seconds)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]
    print(f"[Prefill] Prompt: {input_len} tokens")
    
    torch.mps.synchronize()
    t0 = time.perf_counter()
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            use_cache=True,
            return_dict=True,
        )
    
    torch.mps.synchronize()
    prefill_time = time.perf_counter() - t0
    
    next_token_logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    first_token = tokenizer.decode(next_token_id[0])
    
    print(f"[Prefill] Prefill complete in {prefill_time*1000:.1f}ms")
    print(f"[Prefill] First predicted token: '{first_token}'")
    
    return outputs.past_key_values, inputs.input_ids, next_token_id, prefill_time


def main():
    parser = argparse.ArgumentParser(description="Prefill Node (M5)")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--prompt", default=LONG_PROMPT)
    parser.add_argument("--host", default=M5_IP)
    parser.add_argument("--port", type=int, default=KV_PORT)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help="Number of runs to average")
    args = parser.parse_args()
    
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    # Step 1: Load model
    model, tokenizer, load_time = load_model(args.model, dtype)
    
    # Step 2: WARMUP — compile MPS shaders before any timing
    warmup_model(model, tokenizer)
    
    # Quick check: how many tokens in the prompt?
    test_tokens = tokenizer(args.prompt, return_tensors="pt")
    prompt_token_count = test_tokens.input_ids.shape[1]
    print(f"\n[Prefill] Experiment config:")
    print(f"  Prompt tokens:  {prompt_token_count}")
    print(f"  Runs:           {args.runs}")
    
    # Step 3: Start KV server
    server = KVServer(host=args.host, port=args.port)
    server.start()
    
    print(f"\n{'='*60}")
    print(f"[Prefill] Waiting for Decode Node to connect...")
    print(f"[Prefill] On M4 Pro, run:")
    print(f"  cd ~/mac-ml-benchmark && source venv/bin/activate && python distributed/decode_node.py --host {args.host} --runs {args.runs}")
    print(f"{'='*60}\n")
    
    conn = server.wait_for_client()
    
    # Step 4: Run experiment (multiple runs)
    all_results = []
    
    for run_idx in range(args.runs):
        print(f"\n{'='*60}")
        print(f"[Prefill] RUN {run_idx + 1}/{args.runs}")
        print(f"{'='*60}")
        
        # Prefill
        kv_cache, input_ids, next_token_id, prefill_time = run_prefill(
            model, tokenizer, args.prompt
        )
        
        # Serialize
        t_ser_start = time.perf_counter()
        kv_bytes = serialize_kv_from_dynamic_cache(kv_cache)
        t_ser_end = time.perf_counter()
        serialize_time = t_ser_end - t_ser_start
        kv_size_mb = len(kv_bytes) / (1024 * 1024)
        
        print(f"[Prefill] KV cache: {kv_size_mb:.1f} MB serialized in {serialize_time*1000:.1f}ms")
        
        # Send metadata + KV cache
        metadata = {
            "model": args.model,
            "prompt": args.prompt[:200],  # truncate for transfer
            "input_len": input_ids.shape[1],
            "next_token_id": next_token_id.cpu().tolist(),
            "prefill_time_ms": prefill_time * 1000,
            "serialize_time_ms": serialize_time * 1000,
            "kv_size_mb": kv_size_mb,
            "dtype": args.dtype,
            "run_idx": run_idx,
            "total_runs": args.runs,
        }
        
        print(f"[Prefill] Sending KV cache to Decode Node...")
        transfer_timings = send_kv_cache(conn, kv_bytes, metadata)
        
        run_result = {
            "run": run_idx + 1,
            "prompt_tokens": input_ids.shape[1],
            "prefill_time_ms": prefill_time * 1000,
            "serialize_time_ms": serialize_time * 1000,
            "transfer_time_ms": transfer_timings['transfer_time_s'] * 1000,
            "kv_size_mb": kv_size_mb,
            "tb_throughput_gbps": transfer_timings['throughput_gbps'],
            "tb_throughput_gbs": transfer_timings['throughput_gbs'],
        }
        all_results.append(run_result)
        
        print(f"  Prefill:      {prefill_time*1000:.1f}ms")
        print(f"  Serialize:    {serialize_time*1000:.1f}ms")
        print(f"  Transfer:     {transfer_timings['transfer_time_s']*1000:.1f}ms ({kv_size_mb:.1f} MB)")
        print(f"  TB:           {transfer_timings['throughput_gbps']:.1f} Gbps")
        
        # Wait for decode node to signal run complete before next run
        if run_idx < args.runs - 1:
            print(f"[Prefill] Waiting for decode node to finish run {run_idx + 1}...")
            ack = conn.recv(1024)
            if ack:
                ack_data = json.loads(ack.decode())
                print(f"[Prefill] Decode run {run_idx + 1}: {ack_data.get('decode_tokens_per_sec', '?'):.1f} tok/s")
    
    # Wait for final results from decode node
    print(f"\n[Prefill] Waiting for final results from Decode Node...")
    try:
        final_signal = conn.recv(4096)
        if final_signal:
            decode_final = json.loads(final_signal.decode())
    except Exception as e:
        decode_final = {}
        print(f"[Prefill] Could not receive decode results: {e}")
    
    # Compute averages
    avg_prefill = sum(r['prefill_time_ms'] for r in all_results) / len(all_results)
    avg_serialize = sum(r['serialize_time_ms'] for r in all_results) / len(all_results)
    avg_transfer = sum(r['transfer_time_ms'] for r in all_results) / len(all_results)
    avg_tb_gbps = sum(r['tb_throughput_gbps'] for r in all_results) / len(all_results)
    
    print(f"\n{'='*60}")
    print(f"[Prefill] AVERAGED RESULTS ({args.runs} runs):")
    print(f"{'='*60}")
    print(f"  Prompt tokens:  {all_results[0]['prompt_tokens']}")
    print(f"  KV cache size:  {all_results[0]['kv_size_mb']:.1f} MB")
    print(f"  Prefill:        {avg_prefill:.1f}ms")
    print(f"  Serialize:      {avg_serialize:.1f}ms")
    print(f"  Transfer:       {avg_transfer:.1f}ms")
    print(f"  TB throughput:  {avg_tb_gbps:.1f} Gbps")
    if decode_final:
        print(f"  --- Decode (from M4 Pro) ---")
        for k, v in decode_final.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.1f}")
            else:
                print(f"  {k}: {v}")
    print(f"{'='*60}")
    
    # Save
    results = {
        "role": "prefill",
        "machine": "M5",
        "model": args.model,
        "prompt_tokens": all_results[0]['prompt_tokens'],
        "max_new_tokens": MAX_NEW_TOKENS,
        "num_runs": args.runs,
        "warmup": True,
        "load_time_s": load_time,
        "averaged": {
            "prefill_time_ms": avg_prefill,
            "serialize_time_ms": avg_serialize,
            "transfer_time_ms": avg_transfer,
            "kv_size_mb": all_results[0]['kv_size_mb'],
            "tb_throughput_gbps": avg_tb_gbps,
        },
        "all_runs": all_results,
        "decode_results": decode_final,
    }
    
    os.makedirs("results/raw/distributed", exist_ok=True)
    with open("results/raw/distributed/prefill_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[Prefill] Results saved to results/raw/distributed/prefill_results.json")
    
    conn.close()
    server.close()
    print("[Prefill] Done.")


if __name__ == "__main__":
    main()
