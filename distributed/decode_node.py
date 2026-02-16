"""
Decode Node — Runs on M4 Pro (169.254.1.2)

Loads Mistral-7B, warms up MPS shaders, receives the KV cache from the
Prefill Node over Thunderbolt, and runs autoregressive decode.

WARMUP: MPS does lazy shader compilation — the first forward pass for any new
tensor shape compiles Metal kernels (~800ms overhead). We run a warmup pass
before timing to ensure fair measurement.

Usage:
    python distributed/decode_node.py
    python distributed/decode_node.py --host 169.254.1.1 --runs 3 --max-tokens 256
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
from experiment_config import MODEL_NAME, MAX_NEW_TOKENS, NUM_RUNS

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str = MODEL_NAME, dtype=torch.float16):
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


def warmup_model(model, tokenizer, device="mps"):
    """
    Warmup MPS shader compilation.
    
    WHY: MPS lazily compiles Metal GPU kernels the first time it sees a
    forward pass with a given tensor shape. This adds ~800ms overhead.
    We run 2 warmup passes to compile:
      1. Prefill kernels (short sequence → model(**inputs))
      2. Decode kernels (single token + KV cache → model(input_ids, past_key_values))
    """
    print(f"[Decode] Warming up MPS shaders...")
    
    # Warmup 1: Prefill-style (compile attention/FFN kernels)
    short_input = tokenizer("Hello world", return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**short_input, use_cache=True, return_dict=True)
    torch.mps.synchronize()
    
    # Warmup 2: Decode-style (single token + KV cache — this is what decode_node does)
    next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    with torch.no_grad():
        _ = model(input_ids=next_id, past_key_values=out.past_key_values,
                  use_cache=True, return_dict=True)
    torch.mps.synchronize()
    
    print(f"[Decode] Warmup complete — MPS shaders compiled (prefill + decode kernels)")


def run_decode(model, tokenizer, kv_cache, first_token_id: torch.Tensor,
               max_new_tokens: int = MAX_NEW_TOKENS, device: str = "mps"):
    """
    Run decode phase: generate tokens one at a time using the KV cache.
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
        per_token_times.append((t_tok_end - t_tok_start) * 1000)
        
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        generated_ids.append(next_token_id.item())
        kv_cache = outputs.past_key_values
        current_token = next_token_id
        
        if next_token_id.item() == tokenizer.eos_token_id:
            print(f"[Decode] EOS at token {i+1}")
            break
        
        if (i + 1) % 64 == 0:
            elapsed = time.perf_counter() - t_decode_start
            tps = (i + 1) / elapsed
            print(f"[Decode] Token {i+1}/{max_new_tokens} ({tps:.1f} tok/s)")
    
    torch.mps.synchronize()
    t_decode_end = time.perf_counter()
    
    decode_time = t_decode_end - t_decode_start
    generated_tokens = len(generated_ids)
    tokens_per_sec = generated_tokens / decode_time if decode_time > 0 else 0
    
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print(f"[Decode] Generated {generated_tokens} tokens in {decode_time*1000:.1f}ms ({tokens_per_sec:.1f} tok/s)")
    
    return generated_text, generated_tokens, decode_time, tokens_per_sec, per_token_times


def main():
    parser = argparse.ArgumentParser(description="Decode Node (M4 Pro)")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--host", default=M5_IP, help="Prefill node IP")
    parser.add_argument("--port", type=int, default=KV_PORT)
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--runs", type=int, default=NUM_RUNS, help="Number of runs to average")
    args = parser.parse_args()
    
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    
    # Step 1: Load model
    model, tokenizer, load_time = load_model(args.model, dtype)
    
    # Step 2: WARMUP — compile MPS shaders before any timing
    warmup_model(model, tokenizer)
    
    # Step 3: Connect to prefill node
    client = KVClient(server_host=args.host, port=args.port)
    client.connect()
    
    print(f"\n[Decode] Waiting for KV cache from Prefill Node...")
    
    # Step 4: Run experiment (multiple runs)
    all_results = []
    
    for run_idx in range(args.runs):
        print(f"\n{'='*60}")
        print(f"[Decode] RUN {run_idx + 1}/{args.runs}")
        print(f"{'='*60}")
        
        # Receive KV cache
        t_run_start = time.perf_counter()
        
        kv_data, metadata, transfer_timings = recv_kv_cache(client.sock)
        
        print(f"[Decode] KV cache received: {transfer_timings['kv_size_mb']:.1f} MB in {transfer_timings['transfer_time_s']*1000:.1f}ms")
        print(f"[Decode] TB: {transfer_timings['throughput_gbps']:.1f} Gbps | Prefill on M5: {metadata.get('prefill_time_ms', '?'):.1f}ms")
        
        # Deserialize
        t_deser_start = time.perf_counter()
        kv_cache = deserialize_kv_cache(kv_data, device="mps")
        t_deser_end = time.perf_counter()
        deserialize_time = t_deser_end - t_deser_start
        
        print(f"[Decode] KV deserialized in {deserialize_time*1000:.1f}ms")
        
        # Decode
        first_token_id = torch.tensor(metadata['next_token_id'], device="mps")
        
        generated_text, generated_tokens, decode_time, tokens_per_sec, per_token_times = run_decode(
            model, tokenizer, kv_cache, first_token_id, max_new_tokens=args.max_tokens
        )
        
        t_run_end = time.perf_counter()
        
        # Timing breakdown
        prefill_time_ms = metadata.get('prefill_time_ms', 0)
        serialize_time_ms = metadata.get('serialize_time_ms', 0)
        transfer_time_ms = transfer_timings['transfer_time_s'] * 1000
        deser_time_ms = deserialize_time * 1000
        first_tok_ms = per_token_times[0] if per_token_times else 0
        
        # TTFT = time from "prompt submitted" to "first token appears"
        # In disaggregated: prefill(M5) + serialize + transfer + deserialize + first_decode_step
        ttft = prefill_time_ms + serialize_time_ms + transfer_time_ms + deser_time_ms + first_tok_ms
        
        # Total E2E = everything from prefill start to last token
        total_e2e_ms = prefill_time_ms + serialize_time_ms + transfer_time_ms + deser_time_ms + decode_time * 1000
        
        run_result = {
            "run": run_idx + 1,
            "prompt_tokens": metadata.get('input_len', 0),
            "generated_tokens": generated_tokens,
            "prefill_time_ms": prefill_time_ms,
            "serialize_time_ms": serialize_time_ms,
            "transfer_time_ms": transfer_time_ms,
            "kv_size_mb": transfer_timings['kv_size_mb'],
            "tb_throughput_gbps": transfer_timings['throughput_gbps'],
            "deserialize_time_ms": deser_time_ms,
            "decode_time_ms": decode_time * 1000,
            "tokens_per_sec": tokens_per_sec,
            "ttft_ms": ttft,
            "total_e2e_ms": total_e2e_ms,
            "avg_token_latency_ms": sum(per_token_times) / len(per_token_times) if per_token_times else 0,
            "per_token_times_ms": per_token_times[:10],
        }
        all_results.append(run_result)
        
        print(f"\n  [M5]  Prefill:         {prefill_time_ms:.1f}ms")
        print(f"  [M5]  Serialize:       {serialize_time_ms:.1f}ms")
        print(f"  [TB]  Transfer:        {transfer_time_ms:.1f}ms ({transfer_timings['kv_size_mb']:.1f} MB)")
        print(f"  [M4]  Deserialize:     {deser_time_ms:.1f}ms")
        print(f"  [M4]  Decode:          {decode_time*1000:.1f}ms ({generated_tokens} tokens, {tokens_per_sec:.1f} tok/s)")
        print(f"  ---")
        print(f"  TTFT:                  {ttft:.1f}ms")
        print(f"  Total E2E:             {total_e2e_ms:.1f}ms")
        
        # Signal run complete to prefill node (so it starts next run)
        if run_idx < args.runs - 1:
            ack = {"run": run_idx + 1, "decode_tokens_per_sec": tokens_per_sec, "status": "done"}
            try:
                client.sock.sendall(json.dumps(ack).encode())
            except Exception:
                pass
    
    # Compute averages
    def avg(key):
        return sum(r[key] for r in all_results) / len(all_results)
    
    avg_result = {
        "prefill_time_ms": avg("prefill_time_ms"),
        "serialize_time_ms": avg("serialize_time_ms"),
        "transfer_time_ms": avg("transfer_time_ms"),
        "deserialize_time_ms": avg("deserialize_time_ms"),
        "decode_time_ms": avg("decode_time_ms"),
        "tokens_per_sec": avg("tokens_per_sec"),
        "ttft_ms": avg("ttft_ms"),
        "total_e2e_ms": avg("total_e2e_ms"),
        "avg_token_latency_ms": avg("avg_token_latency_ms"),
        "generated_tokens": avg("generated_tokens"),
    }
    
    print(f"\n{'='*60}")
    print(f"[Decode] AVERAGED RESULTS ({args.runs} runs) — DISAGGREGATED")
    print(f"{'='*60}")
    print(f"  Prompt tokens:      {all_results[0]['prompt_tokens']}")
    print(f"  Generated tokens:   {avg_result['generated_tokens']:.0f}")
    print(f"  KV cache:           {all_results[0]['kv_size_mb']:.1f} MB")
    print(f"  ---")
    print(f"  [M5] Prefill:       {avg_result['prefill_time_ms']:.1f}ms")
    print(f"  [M5] Serialize:     {avg_result['serialize_time_ms']:.1f}ms")
    print(f"  [TB] Transfer:      {avg_result['transfer_time_ms']:.1f}ms")
    print(f"  [M4] Deserialize:   {avg_result['deserialize_time_ms']:.1f}ms")
    print(f"  [M4] Decode:        {avg_result['decode_time_ms']:.1f}ms")
    print(f"  ---")
    print(f"  TTFT:               {avg_result['ttft_ms']:.1f}ms")
    print(f"  Decode tok/s:       {avg_result['tokens_per_sec']:.1f}")
    print(f"  Total E2E:          {avg_result['total_e2e_ms']:.1f}ms")
    print(f"{'='*60}")
    
    # Send final averaged results back to prefill node
    try:
        client.sock.sendall(json.dumps(avg_result).encode())
    except Exception:
        pass
    
    # Save full results
    results = {
        "role": "disaggregated_decode",
        "machine": "M4_Pro",
        "prefill_machine": "M5",
        "model": args.model,
        "prompt_tokens": all_results[0]['prompt_tokens'],
        "max_new_tokens": args.max_tokens,
        "num_runs": args.runs,
        "warmup": True,
        "load_time_s": load_time,
        "averaged": avg_result,
        "all_runs": all_results,
        "generated_text": generated_text[:500] if 'generated_text' in dir() else "",
    }
    
    os.makedirs("results/raw/distributed", exist_ok=True)
    with open("results/raw/distributed/disaggregated_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[Decode] Results saved to results/raw/distributed/disaggregated_results.json")
    
    client.close()
    print("[Decode] Done.")


if __name__ == "__main__":
    main()
