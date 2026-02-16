"""
Prefill Node — Runs on M5 (169.254.1.1)

Loads Mistral-7B, processes the prompt (prefill phase), and sends
the KV cache to the Decode Node over Thunderbolt.

Usage:
    python distributed/prefill_node.py --prompt "What is the meaning of life?"
    python distributed/prefill_node.py --prompt "Explain quantum computing" --seq-len 512
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

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str = "mistralai/Mistral-7B-v0.1", dtype=torch.float16):
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


def run_prefill(model, tokenizer, prompt: str, device: str = "mps"):
    """
    Run prefill phase: encode the prompt and compute KV cache.
    
    Returns:
        kv_cache: The DynamicCache with pre-computed keys/values
        input_ids: The tokenized prompt
        last_token_logits: Logits for the last token (to get first generated token)
        prefill_time: Time taken for prefill
    """
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]
    print(f"[Prefill] Prompt: {input_len} tokens")
    
    # Run prefill
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
    
    # Get the first predicted token
    next_token_logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    first_token = tokenizer.decode(next_token_id[0])
    
    print(f"[Prefill] Prefill complete in {prefill_time*1000:.1f}ms")
    print(f"[Prefill] First predicted token: '{first_token}'")
    
    return outputs.past_key_values, inputs.input_ids, next_token_id, prefill_time


def main():
    parser = argparse.ArgumentParser(description="Prefill Node (M5)")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--prompt", default="Explain the theory of general relativity in simple terms. Start with")
    parser.add_argument("--host", default=M5_IP)
    parser.add_argument("--port", type=int, default=KV_PORT)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = parser.parse_args()
    
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    # Step 1: Load model
    model, tokenizer, load_time = load_model(args.model, dtype)
    
    # Step 2: Start KV server
    server = KVServer(host=args.host, port=args.port)
    server.start()
    
    print(f"\n{'='*60}")
    print(f"[Prefill] Waiting for Decode Node to connect...")
    print(f"[Prefill] On M4 Pro, run:")
    print(f"  python distributed/decode_node.py --host {args.host}")
    print(f"{'='*60}\n")
    
    conn = server.wait_for_client()
    
    # Step 3: Run prefill
    print(f"\n[Prefill] Running prefill for prompt: '{args.prompt[:60]}...'")
    kv_cache, input_ids, next_token_id, prefill_time = run_prefill(
        model, tokenizer, args.prompt
    )
    
    # Step 4: Serialize KV cache
    t_ser_start = time.perf_counter()
    kv_bytes = serialize_kv_from_dynamic_cache(kv_cache)
    t_ser_end = time.perf_counter()
    serialize_time = t_ser_end - t_ser_start
    
    print(f"[Prefill] KV cache serialized: {len(kv_bytes)/(1024*1024):.1f} MB in {serialize_time*1000:.1f}ms")
    
    # Step 5: Send metadata + KV cache
    metadata = {
        "model": args.model,
        "prompt": args.prompt,
        "input_len": input_ids.shape[1],
        "next_token_id": next_token_id.cpu().tolist(),
        "prefill_time_ms": prefill_time * 1000,
        "serialize_time_ms": serialize_time * 1000,
        "kv_size_mb": len(kv_bytes) / (1024 * 1024),
        "dtype": args.dtype,
    }
    
    print(f"[Prefill] Sending KV cache to Decode Node...")
    transfer_timings = send_kv_cache(conn, kv_bytes, metadata)
    
    print(f"\n{'='*60}")
    print(f"[Prefill] RESULTS:")
    print(f"  Model load:     {load_time:.1f}s")
    print(f"  Prefill:        {prefill_time*1000:.1f}ms ({input_ids.shape[1]} tokens)")
    print(f"  KV serialize:   {serialize_time*1000:.1f}ms")
    print(f"  KV transfer:    {transfer_timings['transfer_time_s']*1000:.1f}ms ({transfer_timings['kv_size_mb']:.1f} MB)")
    print(f"  TB throughput:   {transfer_timings['throughput_gbps']:.1f} Gbps ({transfer_timings['throughput_gbs']:.2f} GB/s)")
    print(f"{'='*60}")
    
    # Save results
    results = {
        "role": "prefill",
        "machine": "M5",
        "model": args.model,
        "prompt_tokens": input_ids.shape[1],
        "load_time_s": load_time,
        "prefill_time_ms": prefill_time * 1000,
        "serialize_time_ms": serialize_time * 1000,
        "transfer_time_ms": transfer_timings['transfer_time_s'] * 1000,
        "kv_size_mb": transfer_timings['kv_size_mb'],
        "tb_throughput_gbps": transfer_timings['throughput_gbps'],
        "tb_throughput_gbs": transfer_timings['throughput_gbs'],
    }
    
    os.makedirs("results/raw/distributed", exist_ok=True)
    with open("results/raw/distributed/prefill_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[Prefill] Results saved. Waiting for Decode Node to finish...")
    
    # Wait for decode node to signal completion
    try:
        done_signal = conn.recv(1024)
        if done_signal:
            decode_results = json.loads(done_signal.decode())
            print(f"\n[Prefill] Decode Node finished:")
            print(f"  Decode tokens:  {decode_results.get('generated_tokens', '?')}")
            print(f"  Decode tok/s:   {decode_results.get('decode_tokens_per_sec', '?'):.1f}")
            print(f"  Total E2E:      {decode_results.get('total_e2e_ms', '?'):.1f}ms")
    except Exception as e:
        print(f"[Prefill] Could not receive decode results: {e}")
    
    conn.close()
    server.close()
    print("[Prefill] Done.")


if __name__ == "__main__":
    main()
