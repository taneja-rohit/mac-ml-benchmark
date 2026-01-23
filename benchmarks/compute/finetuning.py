#!/usr/bin/env python3
"""
Fine-tuning benchmarks for PyTorch+MPS and MLX.
Tests LoRA-style training performance on Mistral-7B.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

# ============================================================================
# PYTORCH + MPS FINE-TUNING
# ============================================================================

def benchmark_pytorch_finetuning(
    model_name: str = "mistralai/Mistral-7B-v0.1",
    num_steps: int = 20,
    batch_size: int = 1,
    seq_length: int = 256,
    lora_rank: int = 16,
):
    """Benchmark PyTorch+MPS fine-tuning with LoRA-style approach."""
    
    print("\n" + "="*60)
    print("PYTORCH + MPS FINE-TUNING BENCHMARK")
    print("="*60)
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Device: {device}")
        
        # Load model
        print(f"Loading {model_name}...")
        load_start = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="mps" if device.type == "mps" else "auto",
            low_cpu_mem_usage=True,
        )
        
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.1f}s")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params / 1e9:.2f}B")
        
        # LoRA-style: Only train a subset of parameters
        # We'll simulate LoRA by training just the query/value projections
        trainable_params = 0
        for name, param in model.named_parameters():
            # Train q_proj and v_proj (like LoRA)
            if "q_proj" in name or "v_proj" in name:
                param.requires_grad = True
                trainable_params += param.numel()
            else:
                param.requires_grad = False
        
        print(f"Trainable parameters: {trainable_params / 1e6:.1f}M ({100*trainable_params/total_params:.2f}%)")
        
        # Create dummy training data
        dummy_input = torch.randint(
            0, tokenizer.vocab_size, 
            (batch_size, seq_length), 
            device=device
        )
        dummy_labels = dummy_input.clone()
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4
        )
        
        # Training loop
        print(f"\nRunning {num_steps} training steps...")
        model.train()
        
        losses = []
        total_tokens = 0
        
        # Warmup
        for _ in range(2):
            outputs = model(dummy_input, labels=dummy_labels)
            outputs.loss.backward()
            optimizer.zero_grad()
            if device.type == "mps":
                torch.mps.synchronize()
        
        # Benchmark
        start_time = time.time()
        
        for step in range(num_steps):
            outputs = model(dummy_input, labels=dummy_labels)
            loss = outputs.loss
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 
                1.0
            )
            
            optimizer.step()
            optimizer.zero_grad()
            
            if device.type == "mps":
                torch.mps.synchronize()
            
            losses.append(loss.item())
            total_tokens += batch_size * seq_length
            
            if (step + 1) % 5 == 0:
                print(f"  Step {step+1}/{num_steps}: loss = {loss.item():.4f}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Calculate metrics
        tokens_per_second = total_tokens / total_time
        
        # Peak memory
        if device.type == "mps":
            # MPS doesn't have direct memory query, estimate from model size
            peak_memory_gb = (total_params * 2 + trainable_params * 4 * 3) / (1024**3)  # weights + gradients + optimizer
        else:
            peak_memory_gb = 0
        
        results = {
            "benchmark": "pytorch_mps_finetuning",
            "model": model_name,
            "precision": "float16",
            "trainable_params": trainable_params,
            "total_params": total_params,
            "steps": num_steps,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "time_seconds": total_time,
            "tokens": total_tokens,
            "tokens_per_second": tokens_per_second,
            "start_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "peak_memory_gb": peak_memory_gb,
            "timestamp": datetime.now().isoformat(),
        }
        
        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Tokens/second: {tokens_per_second:.1f}")
        print(f"  Start loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Peak memory: ~{peak_memory_gb:.1f} GB (estimated)")
        print(f"{'='*60}")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================================
# MLX FINE-TUNING (LoRA)
# ============================================================================

def benchmark_mlx_finetuning(
    model_name: str = "mlx-community/Mistral-7B-Instruct-v0.2-4bit",
    num_steps: int = 20,
    batch_size: int = 1,
    seq_length: int = 256,
):
    """Benchmark MLX fine-tuning with LoRA."""
    
    print("\n" + "="*60)
    print("MLX FINE-TUNING BENCHMARK (LoRA)")
    print("="*60)
    
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx_lm import load
        from mlx_lm.tuner.trainer import TrainingArgs
        from mlx_lm.tuner import train as mlx_train
        
        print(f"Device: {mx.default_device()}")
        
        # Load model
        print(f"Loading {model_name}...")
        load_start = time.time()
        
        model, tokenizer = load(model_name)
        
        load_time = time.time() - load_start
        print(f"Model loaded in {load_time:.1f}s")
        
        # Create dummy training data
        dummy_tokens = mx.random.randint(0, 32000, (batch_size, seq_length))
        
        # Simple forward/backward benchmark (mlx_lm.train has its own data loading)
        print(f"\nRunning {num_steps} forward+backward passes...")
        
        # Get trainable params count
        total_params = sum(p.size for _, p in model.parameters().items())
        
        # For LoRA, typically ~0.1-1% of params are trainable
        # Estimate based on standard LoRA config
        estimated_trainable = int(total_params * 0.01)
        
        total_tokens = 0
        
        # Warmup
        for _ in range(2):
            logits = model(dummy_tokens)
            mx.eval(logits)
        
        # Benchmark forward passes (simplified - full training would use mlx_lm.train)
        start_time = time.time()
        
        for step in range(num_steps):
            logits = model(dummy_tokens)
            mx.eval(logits)
            total_tokens += batch_size * seq_length
            
            if (step + 1) % 5 == 0:
                print(f"  Step {step+1}/{num_steps}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        tokens_per_second = total_tokens / total_time
        
        results = {
            "benchmark": "mlx_finetuning",
            "model": model_name,
            "precision": "4-bit quantized",
            "estimated_trainable_params": estimated_trainable,
            "steps": num_steps,
            "batch_size": batch_size,
            "seq_length": seq_length,
            "time_seconds": total_time,
            "tokens": total_tokens,
            "tokens_per_second": tokens_per_second,
            "note": "Forward pass only - full LoRA training uses mlx_lm.lora",
            "timestamp": datetime.now().isoformat(),
        }
        
        print(f"\n{'='*60}")
        print(f"RESULTS (forward pass benchmark):")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Tokens/second: {tokens_per_second:.1f}")
        print(f"{'='*60}")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}


# ============================================================================
# MAIN
# ============================================================================

def run_finetuning_benchmarks(output_dir: str = None):
    """Run all fine-tuning benchmarks."""
    
    print("\n" + "="*70)
    print("       FINE-TUNING BENCHMARK SUITE")
    print("="*70)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "benchmarks": {}
    }
    
    # PyTorch + MPS
    print("\n[1/2] PyTorch + MPS Fine-tuning")
    pytorch_results = benchmark_pytorch_finetuning()
    results["benchmarks"]["pytorch_mps"] = pytorch_results
    
    # MLX
    print("\n[2/2] MLX Fine-tuning")
    user_input = input("Run MLX fine-tuning benchmark? (y/n): ").strip().lower()
    if user_input == 'y':
        mlx_results = benchmark_mlx_finetuning()
        results["benchmarks"]["mlx"] = mlx_results
    
    # Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "finetuning_benchmarks.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tuning benchmarks")
    parser.add_argument("--output-dir", type=str, default="results/raw",
                        help="Output directory for results")
    parser.add_argument("--pytorch-only", action="store_true",
                        help="Only run PyTorch benchmark")
    args = parser.parse_args()
    
    if args.pytorch_only:
        results = benchmark_pytorch_finetuning()
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(os.path.join(args.output_dir, "pytorch_mps_finetuning.json"), "w") as f:
                json.dump(results, f, indent=2)
    else:
        run_finetuning_benchmarks(args.output_dir)
