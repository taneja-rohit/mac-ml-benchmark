"""
Decode Node — Runs on M4 Pro (169.254.1.2)

Loads Mistral-7B, receives the KV cache from the Prefill Node,
and runs autoregressive token generation (decode phase).

Usage:
    python distributed/decode_node.py
    python distributed/decode_node.py --max-tokens 128 --host 169.254.1.1
"""

import argparse
import json
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_transfer import (
    KVClient, recv_kv_cache, deserialize_kv_cache, M5_IP, KV_PORT
)

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str = "mistralai/Mistral-7B-v0.1", dtype=torch.float16):
    """Load model and tokenizer onto MPS."""
    print(f"[Decode] Loading {model_name} in {dtype}...")
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
    print(f"[Decode] Model loaded: {param_count:.1f}B params in {load_time:.1f}s")
    
    return model, tokenizer, load_time


def run_decode(model, tokenizer, kv_cache, first_token_id: torch.Tensor,
               max_new_tokens: int = 128, device: str = "mps"):
    """
    Run decode phase: generate tokens one at a time using the KV cache.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        kv_cache: DynamicCache from prefill node
        first_token_id: The first token predicted by prefill
        max_new_tokens: Maximum tokens to generate
        device: Device to run on
    
    Returns:
        generated_text: The decoded text
        generated_tokens: Number of tokens generated
        decode_time: Total decode time
        tokens_per_sec: Decode throughput
        per_token_times: List of per-token latencies
    """
    generated_ids = []
    per_token_times = []
    current_token = first_token_id.to(device)
    
    print(f"[Decode] Starting decode (max {max_new_tokens} tokens)...")
    
    torch.mps.synchronize()
    t_decode_start = time.perf_counter()
    
    for i in range(max_new_tokens):
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
        per_token_times.append((t_tok_end - t_tok_start) * 1000)  # ms
        
        # Get next token
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        generated_ids.append(next_token_id.item())
        kv_cache = outputs.past_key_values
        current_token = next_token_id
        
        # Check for EOS
        if next_token_id.item() == tokenizer.eos_token_id:
            print(f"[Decode] EOS at token {i+1}")
            break
        
        # Progress
        if (i + 1) % 32 == 0:
            elapsed = time.perf_counter() - t_decode_start
            tps = (i + 1) / elapsed
            print(f"[Decode] Token {i+1}/{max_new_tokens} ({tps:.1f} tok/s)")
    
    torch.mps.synchronize()
    t_decode_end = time.perf_counter()
    
    decode_time = t_decode_end - t_decode_start
    generated_tokens = len(generated_ids)
    tokens_per_sec = generated_tokens / decode_time if decode_time > 0 else 0
    
    # Decode the generated text
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"\n[Decode] Generated {generated_tokens} tokens in {decode_time*1000:.1f}ms")
    print(f"[Decode] Throughput: {tokens_per_sec:.1f} tok/s")
    print(f"[Decode] Avg per-token: {sum(per_token_times)/len(per_token_times):.2f}ms")
    
    return generated_text, generated_tokens, decode_time, tokens_per_sec, per_token_times


def main():
    parser = argparse.ArgumentParser(description="Decode Node (M4 Pro)")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--host", default=M5_IP, help="Prefill node IP")
    parser.add_argument("--port", type=int, default=KV_PORT)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    args = parser.parse_args()
    
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    # Step 1: Load model (can happen while prefill node processes)
    model, tokenizer, load_time = load_model(args.model, dtype)
    
    # Step 2: Connect to prefill node
    client = KVClient(server_host=args.host, port=args.port)
    client.connect()
    
    # Step 3: Receive KV cache
    print(f"\n[Decode] Waiting for KV cache from Prefill Node...")
    t_total_start = time.perf_counter()
    
    kv_data, metadata, transfer_timings = recv_kv_cache(client.sock)
    
    print(f"[Decode] KV cache received: {transfer_timings['kv_size_mb']:.1f} MB in {transfer_timings['transfer_time_s']*1000:.1f}ms")
    print(f"[Decode] TB throughput: {transfer_timings['throughput_gbps']:.1f} Gbps ({transfer_timings['throughput_gbs']:.2f} GB/s)")
    print(f"[Decode] Prefill took: {metadata.get('prefill_time_ms', '?')}ms on M5")
    
    # Step 4: Deserialize KV cache
    t_deser_start = time.perf_counter()
    kv_cache = deserialize_kv_cache(kv_data, device="mps")
    t_deser_end = time.perf_counter()
    deserialize_time = t_deser_end - t_deser_start
    
    print(f"[Decode] KV cache deserialized in {deserialize_time*1000:.1f}ms")
    
    # Step 5: Run decode
    first_token_id = torch.tensor(metadata['next_token_id'], device="mps")
    
    generated_text, generated_tokens, decode_time, tokens_per_sec, per_token_times = run_decode(
        model, tokenizer, kv_cache, first_token_id, 
        max_new_tokens=args.max_tokens
    )
    
    t_total_end = time.perf_counter()
    total_e2e = t_total_end - t_total_start
    
    # Print generated text
    prompt = metadata.get('prompt', '')
    print(f"\n{'='*60}")
    print(f"GENERATED TEXT:")
    print(f"{'='*60}")
    print(f"{prompt}{generated_text}")
    print(f"{'='*60}")
    
    # Compute total end-to-end including prefill on M5
    prefill_time_ms = metadata.get('prefill_time_ms', 0)
    serialize_time_ms = metadata.get('serialize_time_ms', 0)
    ttft = prefill_time_ms + serialize_time_ms + transfer_timings['transfer_time_s'] * 1000 + deserialize_time * 1000 + per_token_times[0]
    
    print(f"\n{'='*60}")
    print(f"[Decode] COMPLETE TIMING BREAKDOWN:")
    print(f"{'='*60}")
    print(f"  [M5]  Prefill:           {prefill_time_ms:.1f}ms")
    print(f"  [M5]  KV serialize:      {serialize_time_ms:.1f}ms")
    print(f"  [TB]  KV transfer:       {transfer_timings['transfer_time_s']*1000:.1f}ms ({transfer_timings['kv_size_mb']:.1f} MB)")
    print(f"  [M4]  KV deserialize:    {deserialize_time*1000:.1f}ms")
    print(f"  [M4]  Decode:            {decode_time*1000:.1f}ms ({generated_tokens} tokens)")
    print(f"  ---")
    print(f"  TTFT (time-to-first):    {ttft:.1f}ms")
    print(f"  Decode tok/s:            {tokens_per_sec:.1f}")
    print(f"  Total E2E:               {total_e2e*1000:.1f}ms")
    print(f"  TB throughput:           {transfer_timings['throughput_gbps']:.1f} Gbps")
    print(f"{'='*60}")
    
    # Send results back to prefill node
    decode_results = {
        "generated_tokens": generated_tokens,
        "decode_tokens_per_sec": tokens_per_sec,
        "total_e2e_ms": total_e2e * 1000,
        "ttft_ms": ttft,
        "decode_time_ms": decode_time * 1000,
        "deserialize_time_ms": deserialize_time * 1000,
    }
    
    try:
        client.sock.sendall(json.dumps(decode_results).encode())
    except Exception:
        pass
    
    # Save full results
    results = {
        "role": "decode",
        "machine": "M4_Pro",
        "model": args.model,
        "prompt": metadata.get('prompt', ''),
        "prompt_tokens": metadata.get('input_len', 0),
        "generated_tokens": generated_tokens,
        "load_time_s": load_time,
        "prefill_time_ms": prefill_time_ms,
        "serialize_time_ms": serialize_time_ms,
        "transfer_time_ms": transfer_timings['transfer_time_s'] * 1000,
        "kv_size_mb": transfer_timings['kv_size_mb'],
        "tb_throughput_gbps": transfer_timings['throughput_gbps'],
        "tb_throughput_gbs": transfer_timings['throughput_gbs'],
        "deserialize_time_ms": deserialize_time * 1000,
        "decode_time_ms": decode_time * 1000,
        "tokens_per_sec": tokens_per_sec,
        "ttft_ms": ttft,
        "total_e2e_ms": total_e2e * 1000,
        "avg_token_latency_ms": sum(per_token_times) / len(per_token_times),
        "per_token_times_ms": per_token_times[:10],  # First 10 for analysis
        "generated_text": generated_text[:500],  # Truncate for storage
    }
    
    os.makedirs("results/raw/distributed", exist_ok=True)
    with open("results/raw/distributed/decode_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[Decode] Results saved to results/raw/distributed/decode_results.json")
    
    client.close()
    print("[Decode] Done.")


if __name__ == "__main__":
    main()
