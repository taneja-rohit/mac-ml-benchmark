"""
Ray Disaggregated Inference — M5 (prefill) → M4 Pro (decode)

Compares Ray's object store transport vs our raw TCP socket implementation.
Ray uses Apache Arrow serialization (not pickle) for tensors, and manages
its own data plane for cross-node transfers.

Architecture:
  M5 (head node):   Loads model → prefill → ray.put(KV cache)
  M4 Pro (worker):   DecodeActor receives KV cache via Ray → decode tokens

Setup (run these BEFORE this script):
  On M5:
    ray start --head --port=6379 --node-ip-address=169.254.1.1

  On M4 Pro:
    ray start --address='169.254.1.1:6379' --node-ip-address=169.254.1.2

  Then on M5:
    python distributed/ray_disaggregated.py

Teardown:
    ray stop  (on both machines)
"""

import argparse
import json
import os
import platform
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_config import MODEL_NAME, MAX_NEW_TOKENS, NUM_RUNS, LONG_PROMPT, DTYPE

import ray


# ============================================================
# DECODE ACTOR — Runs on M4 Pro worker node
# ============================================================

@ray.remote
class DecodeActor:
    """
    Remote actor scheduled on M4 Pro.
    Loads Mistral-7B onto MPS, warms up shaders, then decodes
    tokens from a KV cache received via Ray's object store.
    """

    def __init__(self, model_name: str, dtype_str: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.hostname = platform.node()
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"[DecodeActor@{self.hostname}] Loading {model_name} on {self.device}...")
        t0 = time.perf_counter()

        dtype = torch.float16 if dtype_str == "float16" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map=self.device,
        )
        self.model.eval()

        self.load_time = time.perf_counter() - t0
        params = sum(p.numel() for p in self.model.parameters()) / 1e9
        print(f"[DecodeActor@{self.hostname}] Loaded: {params:.1f}B params in {self.load_time:.1f}s")

        self._warmup()

    def _warmup(self):
        """Compile MPS Metal shaders for both prefill and decode kernels."""
        import torch
        print(f"[DecodeActor@{self.hostname}] Warming up MPS shaders...")

        # Warmup 1: prefill kernel
        short = self.tokenizer("Hello world", return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**short, use_cache=True, return_dict=True)
        torch.mps.synchronize()

        # Warmup 2: decode kernel (single token + KV cache)
        nid = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
        with torch.no_grad():
            _ = self.model(input_ids=nid, past_key_values=out.past_key_values,
                           use_cache=True, return_dict=True)
        torch.mps.synchronize()
        print(f"[DecodeActor@{self.hostname}] Warmup complete")

    def ready(self):
        """Health check — blocks until model is loaded and warmed up."""
        return {"hostname": self.hostname, "device": self.device, "load_time_s": self.load_time}

    def decode_with_kv(self, cpu_kv_list, next_token_id_list, max_new_tokens):
        """
        Receive KV cache as list of (key, value) CPU tensors (transferred by
        Ray's object store), move to MPS, and run autoregressive decode.

        Timing breakdown returned:
          to_mps_time_ms:     CPU → MPS tensor copy
          decode_time_ms:     Autoregressive token generation
          total_worker_time_ms: Everything inside this function
        """
        import torch
        from transformers.cache_utils import DynamicCache

        t_func_start = time.perf_counter()

        # --- CPU → MPS ---
        t_to_mps_start = time.perf_counter()
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(cpu_kv_list):
            cache.update(k.to(self.device), v.to(self.device), layer_idx)
        torch.mps.synchronize()
        t_to_mps_end = time.perf_counter()
        to_mps_ms = (t_to_mps_end - t_to_mps_start) * 1000

        # --- Decode ---
        first_token_id = torch.tensor(next_token_id_list, device=self.device)
        current_token = first_token_id
        generated_ids = []
        per_token_times = []

        torch.mps.synchronize()
        t_decode_start = time.perf_counter()

        for i in range(max_new_tokens):
            torch.mps.synchronize()
            t_tok = time.perf_counter()

            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_token,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )

            torch.mps.synchronize()
            per_token_times.append((time.perf_counter() - t_tok) * 1000)

            next_logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(next_logits, dim=-1, keepdim=True)
            generated_ids.append(next_token_id.item())
            cache = outputs.past_key_values
            current_token = next_token_id

            if next_token_id.item() == self.tokenizer.eos_token_id:
                break

            if (i + 1) % 64 == 0:
                elapsed = time.perf_counter() - t_decode_start
                print(f"[DecodeActor] Token {i+1}/{max_new_tokens} ({(i+1)/elapsed:.1f} tok/s)")

        torch.mps.synchronize()
        t_decode_end = time.perf_counter()

        decode_time = t_decode_end - t_decode_start
        gen_tokens = len(generated_ids)
        tok_s = gen_tokens / decode_time if decode_time > 0 else 0
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return {
            "to_mps_time_ms": to_mps_ms,
            "decode_time_ms": decode_time * 1000,
            "generated_tokens": gen_tokens,
            "tokens_per_sec": tok_s,
            "avg_token_latency_ms": sum(per_token_times) / len(per_token_times) if per_token_times else 0,
            "per_token_times_ms": per_token_times[:10],
            "total_worker_time_ms": (time.perf_counter() - t_func_start) * 1000,
            "generated_text": text[:300],
            "hostname": self.hostname,
        }


# ============================================================
# LOCAL HELPERS — Run on M5 (head node)
# ============================================================

def load_prefill_model(model_name, dtype):
    """Load model on M5 for prefill."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Prefill@M5] Loading {model_name}...")
    t0 = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="mps",
    )
    model.eval()

    load_time = time.perf_counter() - t0
    params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[Prefill@M5] Loaded: {params:.1f}B params in {load_time:.1f}s")
    return model, tokenizer, load_time


def warmup_prefill(model, tokenizer, device="mps"):
    """Compile MPS shaders for prefill kernels."""
    print("[Prefill@M5] Warming up MPS shaders...")

    short = tokenizer("Hello world", return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**short, use_cache=True, return_dict=True)
    torch.mps.synchronize()

    med = tokenizer("The transformer architecture " * 8, return_tensors="pt").to(device)
    with torch.no_grad():
        _ = model(**med, use_cache=True, return_dict=True)
    torch.mps.synchronize()

    print("[Prefill@M5] Warmup complete")


def run_prefill(model, tokenizer, prompt, device="mps"):
    """Forward pass to produce KV cache."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs.input_ids.shape[1]

    torch.mps.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, return_dict=True)

    torch.mps.synchronize()
    prefill_time = time.perf_counter() - t0

    next_token_logits = outputs.logits[:, -1, :]
    next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    return outputs.past_key_values, next_token_id, prefill_time, input_len


def extract_cpu_kv(kv_cache):
    """
    Move KV cache from MPS → CPU as a list of (key, value) tensor tuples.
    This is the data that Ray will serialize and transfer to M4 Pro.
    """
    cpu_kv = []
    if hasattr(kv_cache, 'layers') and len(kv_cache.layers) > 0:
        for layer in kv_cache.layers:
            cpu_kv.append((layer.keys.cpu(), layer.values.cpu()))
    elif hasattr(kv_cache, 'key_cache'):
        for i in range(len(kv_cache.key_cache)):
            cpu_kv.append((kv_cache.key_cache[i].cpu(), kv_cache.value_cache[i].cpu()))
    else:
        raise ValueError(f"Unknown cache format: {type(kv_cache)}")
    return cpu_kv


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Ray Disaggregated Inference")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--prompt", default=LONG_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--dtype", default=DTYPE, choices=["float16", "float32"])
    parser.add_argument("--runs", type=int, default=NUM_RUNS)
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # ---- Connect to Ray cluster ----
    print("=" * 60)
    print("[Ray] Connecting to cluster...")
    ray.init(address="auto")

    nodes = [n for n in ray.nodes() if n['Alive']]
    print(f"[Ray] Cluster: {len(nodes)} alive nodes")
    for n in nodes:
        print(f"  - {n['NodeManagerAddress']} ({n.get('NodeName', '?')})")

    if len(nodes) < 2:
        print("\n[Ray] ERROR: Need 2 nodes for disaggregated experiment.")
        print("  On M4 Pro, run:")
        print("    cd ~/mac-ml-benchmark && source venv/bin/activate")
        print("    ray start --address='169.254.1.1:6379' --node-ip-address=169.254.1.2")
        ray.shutdown()
        sys.exit(1)

    print(f"[Ray] ✓ Two nodes detected — proceeding")
    print("=" * 60)

    # ---- Create DecodeActor on worker (M4 Pro) ----
    # We schedule it to any non-head node. Since there are only 2 nodes,
    # it will land on M4 Pro.
    print("\n[Ray] Creating DecodeActor on worker node...")
    t_actor = time.perf_counter()
    decode_actor = DecodeActor.remote(args.model, args.dtype)

    # Wait for model load + warmup
    actor_info = ray.get(decode_actor.ready.remote())
    actor_time = time.perf_counter() - t_actor
    print(f"[Ray] DecodeActor ready on '{actor_info['hostname']}' in {actor_time:.1f}s")

    # ---- Load prefill model on M5 ----
    model, tokenizer, load_time = load_prefill_model(args.model, dtype)
    warmup_prefill(model, tokenizer)

    prompt_tokens = tokenizer(args.prompt, return_tensors="pt").input_ids.shape[1]
    print(f"\n[Ray] Experiment config:")
    print(f"  Model:          {args.model}")
    print(f"  Prompt tokens:  {prompt_tokens}")
    print(f"  Max new tokens: {args.max_tokens}")
    print(f"  Runs:           {args.runs}")
    print(f"  Transport:      Ray Object Store (Arrow serialization)")

    # ---- Experiment runs ----
    all_results = []

    for run_idx in range(args.runs):
        print(f"\n{'='*60}")
        print(f"[Ray] RUN {run_idx + 1}/{args.runs}")
        print(f"{'='*60}")

        # 1. Prefill on M5 (local)
        kv_cache, next_token_id, prefill_time, input_len = run_prefill(
            model, tokenizer, args.prompt
        )
        print(f"  [M5] Prefill:          {prefill_time*1000:.1f}ms ({input_len} tokens)")

        # 2. MPS → CPU copy
        torch.mps.synchronize()
        t_cpu_start = time.perf_counter()
        cpu_kv = extract_cpu_kv(kv_cache)
        torch.mps.synchronize()
        t_cpu_end = time.perf_counter()
        to_cpu_ms = (t_cpu_end - t_cpu_start) * 1000

        # KV cache size
        kv_bytes = sum(k.nelement() * k.element_size() + v.nelement() * v.element_size()
                       for k, v in cpu_kv)
        kv_mb = kv_bytes / (1024 * 1024)
        print(f"  [M5] MPS→CPU:          {to_cpu_ms:.1f}ms ({kv_mb:.1f} MB raw tensors)")

        # 3. ray.put() — Arrow serialization + local Plasma store
        t_put_start = time.perf_counter()
        kv_ref = ray.put(cpu_kv)
        t_put_end = time.perf_counter()
        ray_put_ms = (t_put_end - t_put_start) * 1000
        print(f"  [M5] ray.put():        {ray_put_ms:.1f}ms (serialize + store)")

        # 4. Remote decode — includes: object transfer + deserialize + CPU→MPS + decode
        next_tok_list = next_token_id.cpu().tolist()

        t_remote_start = time.perf_counter()
        result_ref = decode_actor.decode_with_kv.remote(kv_ref, next_tok_list, args.max_tokens)
        decode_result = ray.get(result_ref)
        t_remote_end = time.perf_counter()
        remote_total_ms = (t_remote_end - t_remote_start) * 1000

        # 5. Breakdown
        to_mps_ms = decode_result['to_mps_time_ms']
        decode_ms = decode_result['decode_time_ms']
        # Ray overhead = everything that isn't CPU→MPS or decode
        #   = object transfer over Thunderbolt + Arrow deserialization + Ray scheduling
        ray_overhead_ms = remote_total_ms - decode_result['total_worker_time_ms']
        ray_deser_ms = decode_result['total_worker_time_ms'] - to_mps_ms - decode_ms

        first_tok_ms = decode_result['per_token_times_ms'][0] if decode_result['per_token_times_ms'] else 0

        # TTFT = prefill + MPS→CPU + put + ray_overhead + ray_deser + CPU→MPS + first_token
        ttft = (prefill_time * 1000 + to_cpu_ms + ray_put_ms +
                ray_overhead_ms + ray_deser_ms + to_mps_ms + first_tok_ms)
        total_e2e = prefill_time * 1000 + to_cpu_ms + ray_put_ms + remote_total_ms

        print(f"  [Ray] Network+Sched:   {ray_overhead_ms:.1f}ms (Ray transfer + scheduling)")
        print(f"  [Ray] Deserialization: {ray_deser_ms:.1f}ms (Arrow → CPU tensors)")
        print(f"  [M4] CPU→MPS:          {to_mps_ms:.1f}ms")
        print(f"  [M4] Decode:           {decode_ms:.1f}ms ({decode_result['generated_tokens']} tok, {decode_result['tokens_per_sec']:.1f} tok/s)")
        print(f"  ---")
        print(f"  TTFT:                  {ttft:.1f}ms")
        print(f"  Total E2E:             {total_e2e:.1f}ms")

        run_result = {
            "run": run_idx + 1,
            "prompt_tokens": input_len,
            "generated_tokens": decode_result['generated_tokens'],
            "prefill_time_ms": prefill_time * 1000,
            "mps_to_cpu_ms": to_cpu_ms,
            "ray_put_ms": ray_put_ms,
            "ray_overhead_ms": ray_overhead_ms,
            "ray_deser_ms": ray_deser_ms,
            "to_mps_time_ms": to_mps_ms,
            "decode_time_ms": decode_ms,
            "tokens_per_sec": decode_result['tokens_per_sec'],
            "ttft_ms": ttft,
            "total_e2e_ms": total_e2e,
            "avg_token_latency_ms": decode_result['avg_token_latency_ms'],
            "kv_size_mb": kv_mb,
            "remote_total_ms": remote_total_ms,
            "worker_total_ms": decode_result['total_worker_time_ms'],
        }
        all_results.append(run_result)

    # ---- Averages ----
    def avg(key):
        return sum(r[key] for r in all_results) / len(all_results)

    keys = [
        "prefill_time_ms", "mps_to_cpu_ms", "ray_put_ms", "ray_overhead_ms",
        "ray_deser_ms", "to_mps_time_ms", "decode_time_ms", "tokens_per_sec",
        "ttft_ms", "total_e2e_ms", "avg_token_latency_ms", "generated_tokens",
    ]
    avg_result = {k: avg(k) for k in keys}

    # Also compute combined "transfer overhead" for comparison with TCP
    # TCP had: serialize + transfer + deserialize
    # Ray has: MPS→CPU + ray.put + ray_overhead + ray_deser + CPU→MPS
    avg_result["total_transfer_overhead_ms"] = (
        avg_result["mps_to_cpu_ms"] + avg_result["ray_put_ms"] +
        avg_result["ray_overhead_ms"] + avg_result["ray_deser_ms"] +
        avg_result["to_mps_time_ms"]
    )

    print(f"\n{'='*60}")
    print(f"[Ray] AVERAGED RESULTS ({args.runs} runs) — RAY DISAGGREGATED")
    print(f"{'='*60}")
    print(f"  Prompt tokens:         {all_results[0]['prompt_tokens']}")
    print(f"  Generated tokens:      {avg_result['generated_tokens']:.0f}")
    print(f"  KV cache:              {all_results[0]['kv_size_mb']:.1f} MB")
    print(f"  ---")
    print(f"  [M5]  Prefill:         {avg_result['prefill_time_ms']:.1f}ms")
    print(f"  [M5]  MPS→CPU:         {avg_result['mps_to_cpu_ms']:.1f}ms")
    print(f"  [M5]  ray.put():       {avg_result['ray_put_ms']:.1f}ms")
    print(f"  [Ray] Network+Sched:   {avg_result['ray_overhead_ms']:.1f}ms")
    print(f"  [Ray] Deserialization: {avg_result['ray_deser_ms']:.1f}ms")
    print(f"  [M4]  CPU→MPS:         {avg_result['to_mps_time_ms']:.1f}ms")
    print(f"  [M4]  Decode:          {avg_result['decode_time_ms']:.1f}ms")
    print(f"  ---")
    print(f"  Transfer overhead:     {avg_result['total_transfer_overhead_ms']:.1f}ms (everything between prefill and decode)")
    print(f"  TTFT:                  {avg_result['ttft_ms']:.1f}ms")
    print(f"  Decode tok/s:          {avg_result['tokens_per_sec']:.1f}")
    print(f"  Total E2E:             {avg_result['total_e2e_ms']:.1f}ms")
    print(f"{'='*60}")

    # ---- Save ----
    results = {
        "role": "ray_disaggregated",
        "transport": "ray_object_store",
        "prefill_machine": "M5",
        "decode_machine": "M4_Pro",
        "decode_hostname": actor_info['hostname'],
        "model": args.model,
        "prompt_tokens": all_results[0]['prompt_tokens'],
        "max_new_tokens": args.max_tokens,
        "dtype": args.dtype,
        "num_runs": args.runs,
        "warmup": True,
        "ray_version": ray.__version__,
        "averaged": avg_result,
        "all_runs": all_results,
    }

    os.makedirs("results/raw/distributed", exist_ok=True)
    out_path = "results/raw/distributed/ray_disaggregated_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Ray] Results saved to {out_path}")

    ray.shutdown()
    print("[Ray] Done.")


if __name__ == "__main__":
    main()
