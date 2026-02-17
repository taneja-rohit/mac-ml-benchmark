# Distributed Inference: What We Learned

**Two MacBook Pros. One Thunderbolt 5 cable. Mistral-7B. Three experiments.**

We built disaggregated inference from scratch — raw TCP sockets, then Ray — to understand what actually happens when you split LLM inference across machines. This document captures every lesson.

---

## The Experiment

| Setup | Description |
|:---|:---|
| **M5** (head) | Prefill node — 871-token prompt, produces 109 MB KV cache |
| **M4 Pro** (worker) | Decode node — receives KV cache, generates 256 tokens |
| **Interconnect** | Thunderbolt 5 via `bridge0` — measured 37.6 Gbps with `iperf3` |
| **Model** | Mistral-7B, float16, MPS backend |

Three transport methods tested:
1. **Raw TCP sockets** — hand-rolled `socket.send/recv` with `torch.save` (pickle)
2. **Ray Object Store** — `ray.put/get` with Arrow serialization + Plasma
3. **Baselines** — single-machine inference on M5 and M4 Pro independently

---

## Results: Ray vs TCP vs Baseline

### Head-to-Head (averaged over 3 runs)

| Phase | TCP (pickle) | Ray (Arrow) | What's happening |
|:---|---:|---:|:---|
| **Prefill (M5)** | 5,389ms | 3,620ms | Same model, same prompt — variance from MPS memory pressure |
| **Serialize** | 93ms | 228ms¹ | TCP: `torch.save` (pickle). Ray: MPS→CPU (122ms) + `ray.put` Arrow (106ms) |
| **Network transfer** | 572ms | 1,261ms | TCP: raw socket. Ray: Plasma object manager protocol over TCP |
| **Deserialize** | 1,370ms² | 1,436ms³ | TCP: `torch.load` (pickle + CPU→MPS). Ray: Arrow (12ms!) + CPU→MPS (1,424ms) |
| **Decode (M4 Pro)** | 26,742ms | 30,095ms | Same decode loop — variance from memory pressure |
| **Transfer overhead** | **2,035ms** | **2,924ms** | Everything between prefill end and decode start |
| **TTFT** | 7,553ms | 6,642ms | Time to first token |
| **Total E2E** | 34,198ms | 36,639ms | End-to-end |
| **Decode tok/s** | 8.7 | 8.8 | Identical — decode is purely local |

¹ Ray splits this: 122ms MPS→CPU copy + 106ms Arrow serialization into Plasma store.
² TCP's 1,370ms bundles pickle deserialization + CPU→MPS copy together.
³ Ray splits this: 12ms Arrow deserialization + 1,424ms CPU→MPS copy.

### Single-Machine Baselines

| Metric | M5 alone | M4 Pro alone | Disaggregated (TCP) | Disaggregated (Ray) |
|:---|---:|---:|---:|---:|
| Total E2E | 42,817ms | ~34,000ms | 34,198ms | 36,639ms |
| TTFT | 1,521ms | ~1,300ms | 7,553ms | 6,642ms |
| Decode tok/s | 5.8 | ~11 | 8.7 | 8.8 |

**Verdict:** Disaggregation didn't help. The 2-3 second transfer overhead wiped out any benefit from using M4 Pro's faster decode. Single-machine M4 Pro was the best option.

---

## Why: The macOS Networking Tax

### The Fundamental Problem

Apple exposes Thunderbolt as a **network device** (`bridge0`), not a PCIe bus extension. Every byte between machines must traverse:

```
MPS GPU (M5) → CPU (M5) → TCP/IP stack → Thunderbolt NIC → Thunderbolt NIC → TCP/IP stack → CPU (M4) → MPS GPU (M4)
```

Seven copies. On Linux with NVIDIA GPUs and RDMA:

```
GPU (Node 1) → NIC (GPUDirect RDMA) → NIC → GPU (Node 2)
```

One copy. That's why vLLM transfers 109 MB in **~3ms** on InfiniBand vs our **~2,000ms** on Thunderbolt.

### Why TCP was faster than Ray for raw transfer

| Layer | TCP (ours) | Ray |
|:---|:---|:---|
| Serialization | `torch.save` → pickle → bytes | Arrow C++ → Plasma store (mmap) |
| Framing | 4-byte length header | Plasma object manager protocol (object ID, metadata, chunks) |
| Transport | `socket.send(raw_bytes)` | Plasma object manager → TCP |
| Deserialization | `torch.load` → pickle → tensors | Arrow zero-copy → numpy → torch |
| Overhead | ~0 (just raw TCP) | gRPC scheduling + object resolution + protocol headers |

TCP won because we had **zero protocol overhead** — just bytes on a wire. Ray's object manager adds request/response handshakes, object ID resolution, and scheduling latency. For one large transfer, that overhead is pure cost.

### What `iperf3` told us vs what we achieved

| Metric | iperf3 | TCP experiment | Ray experiment |
|:---|---:|---:|---:|
| Throughput (Gbps) | 37.6 | 4.8 | ~2.8 (estimated) |
| Utilization | 100% | 13% | 7% |

The gap: `iperf3` sends from kernel buffers using optimized C. We send from Python userspace, going through GIL, `memoryview` chunking, and TCP congestion control. Ray adds another layer of C++ protocol on top.

---

## Arrow vs Pickle — The Serialization Story

### What pickle does (torch.save)

```python
torch.save(kv_cache, buffer)
# For each of 32 layers × 2 tensors (key + value):
#   1. Python pickle: write opcode "REDUCE" 
#   2. Write class name "torch._utils._rebuild_tensor_v2"
#   3. Write storage type, dtype, shape, strides as Python objects
#   4. Write raw tensor data bytes
#   5. Write pickle "STOP" opcode
# Result: 109 MB of pickle-encoded data

torch.load(buffer)
# For each tensor:
#   1. Python interpreter parses pickle opcodes (SLOW — Python bytecode)
#   2. Calls torch._utils._rebuild_tensor_v2() to create tensor
#   3. Allocates memory, copies raw data
#   4. Moves to MPS device
# Result: 1,370ms — dominated by Python object creation
```

### What Arrow does (ray.put/get)

```python
ray.put(cpu_kv_list)
# Arrow C++ library (GIL RELEASED):
#   1. For each tensor: record dtype + shape in Arrow schema (metadata)
#   2. Point to raw data buffer — NO COPY for contiguous tensors
#   3. Write Arrow IPC format to Plasma shared memory
# Result: 106ms — C++ speed, minimal data copying

ray.get(ref)
# Arrow C++ library (GIL RELEASED):
#   1. Read Arrow schema (metadata)
#   2. Wrap raw buffer as numpy array (pointer + shape, NO COPY)
#   3. torch.from_numpy() shares numpy buffer (NO COPY)
# Result: 12ms — just pointer arithmetic
```

### The comparison

| Aspect | Pickle (torch.save/load) | Arrow (ray.put/get) |
|:---|:---|:---|
| Language | Python bytecode interpreter | C++ library |
| GIL | Held during entire serialize/deserialize | Released during C++ execution |
| Data copy | Full copy on both serialize and deserialize | Near zero-copy (pointer reuse) |
| Metadata | Python object creation per tensor | Schema lookup (O(1)) |
| Our serialize time | 93ms | 106ms (+ 122ms MPS→CPU) |
| Our deserialize time | **1,370ms** (includes CPU→MPS) | **12ms** (then 1,424ms CPU→MPS separately) |

**Arrow eliminated deserialization cost — 1,370ms → 12ms (114x faster).** But the CPU→MPS copy (1,424ms) remained, hidden inside pickle's number but exposed as a separate step in Ray's pipeline.

---

## Plasma — Ray's Object Store

### Architecture

```
Each Ray node runs:
┌────────────────────────────────────┐
│  Raylet (C++ daemon)               │
│  ├── Scheduler (assigns tasks)     │
│  ├── Object Manager (cross-node)   │
│  └── Plasma Store (shared memory)  │
│       └── mmap'd region (~2 GB)    │
│           ├── ObjectID_abc [109MB] │
│           ├── ObjectID_def [4KB]   │
│           └── ...                  │
├────────────────────────────────────┤
│  Worker Process 1 (Python + GIL)   │ ← reads Plasma via mmap
│  Worker Process 2 (Python + GIL)   │ ← reads same mmap, NO COPY
│  Worker Process 3 (Python + GIL)   │ ← reads same mmap, NO COPY
│  ...                               │
└────────────────────────────────────┘
```

### Key properties

| Property | What it means |
|:---|:---|
| **Shared memory (mmap)** | All workers on the same machine read the same physical memory. Zero-copy between processes. |
| **Immutable objects** | Once written, objects cannot be modified. Enables safe concurrent reads without locks. |
| **C++ implementation** | Store management, serialization, and network transfer happen outside Python. No GIL contention. |
| **Reference counted** | Objects are garbage collected when no `ObjectRef` points to them. |
| **Spilling** | When Plasma fills up, objects spill to disk automatically. |

### Same-node vs cross-node

```
Same node (where Plasma shines):
  Worker 1: ray.put(data) → serialize to Plasma mmap
  Worker 2: ray.get(ref)  → pointer into SAME mmap → zero-copy!
  Worker 3: ray.get(ref)  → pointer into SAME mmap → zero-copy!
  Cost: serialize once, read free.

Cross-node (our experiment):
  M5 Plasma: ray.put(data) → serialize to local mmap
  M4 Object Manager: "I need ObjectID_abc"
  M5 Object Manager: reads from local mmap → TCP send → M4 receives
  M4 Plasma: writes to local mmap
  M4 Worker: ray.get(ref) → reads from local mmap
  Cost: serialize + full network transfer + store again. No zero-copy benefit.
```

**Plasma's zero-copy only helps on the same machine.** For cross-node, it's just a well-organized TCP pipe.

---

## Python GIL — What Ray Actually Solves

### The problem

Python's Global Interpreter Lock (GIL): only **one thread** can execute Python bytecode at a time per process. This means:

```python
# Threading (what most people try first):
import threading

def deserialize_for_user(data):
    return torch.load(data)  # Python pickle — holds GIL

# Thread 1: deserialize user A → GIL locked
# Thread 2: deserialize user B → BLOCKED, waiting for GIL
# Result: sequential execution, no parallelism
```

### Ray's solution: processes, not threads

```
Ray spawns separate Python PROCESSES (not threads).
Each process has its own Python interpreter and its own GIL.

Machine M4 Pro (12 cores):
  ray start → pre-spawns 12 worker processes

  Process 1 (pid=85467, GIL_1): DecodeActor — our experiment
  Process 2 (pid=85470, GIL_2): idle (available for another actor)
  Process 3 (pid=85473, GIL_3): idle
  ... 
  Process 12 (pid=85510, GIL_12): idle

  If 12 users send requests simultaneously:
    Process 1: deserialize + decode for User A  (GIL_1 — independent)
    Process 2: deserialize + decode for User B  (GIL_2 — independent)
    ...
    Process 12: deserialize + decode for User L (GIL_12 — independent)
    
  TRUE parallelism. 12 users served concurrently.
```

### The tradeoff Ray manages

| Approach | Memory sharing | Parallelism | IPC cost |
|:---|:---|:---|:---|
| **Threads** | Share everything (same process) | ❌ GIL kills it | Free (shared memory) |
| **Processes** | Share nothing (separate memory) | ✅ True parallel | Expensive (serialize + copy) |
| **Ray (processes + Plasma)** | Share via Plasma mmap | ✅ True parallel | ✅ Near-free (mmap pointer) |

Ray gives you the parallelism of processes with the data-sharing efficiency of threads. Plasma is the bridge.

### Did GIL matter for our experiment?

**No.** We had 1 user, 1 request, sequential execution: prefill → serialize → transfer → deserialize → decode. Nothing was concurrent. The GIL was never contended.

Where GIL bypass matters: **production serving with concurrent users.** 100 users hitting the same model → 12 Ray workers process 12 requests in true parallel → 12x throughput vs a single GIL-locked process.

---

## What Ray Provides — Honest Scorecard

### For our experiment (2 machines, 1 user)

| Capability | Value | Why |
|:---|:---|:---|
| Cluster management | ✅ Convenient | `ray start` vs manual TCP server/client |
| Actor scheduling | ✅ Convenient | Auto-placed DecodeActor on M4 Pro |
| Arrow serialization | ✅ 12ms vs 1,370ms deser | Real improvement, but CPU→MPS dominates |
| Plasma (same-node) | ❌ No benefit | We're cross-node |
| GIL bypass | ❌ No benefit | Sequential workload |
| Transport speed | ❌ Slower than TCP | Protocol overhead added ~900ms |
| **Net result** | **Convenience, not speed** | Raw TCP was 7% faster end-to-end |

### At datacenter scale (100+ machines, 1000+ users)

| Capability | Value | Why |
|:---|:---|:---|
| Cluster management | ✅✅ Essential | Can't SSH into 100 machines to start TCP servers |
| Task scheduling | ✅✅ Essential | Millions of tasks routed to optimal workers |
| Plasma (same-node) | ✅✅ Huge | 12 workers sharing model weights = zero-copy |
| GIL bypass | ✅✅ Critical | 12 concurrent requests per machine, true parallel |
| Fault tolerance | ✅✅ Essential | Node dies → Ray re-schedules tasks automatically |
| Auto-scaling | ✅✅ Essential | Add GPUs on demand |
| Ray Serve | ✅✅ Production | Model serving with batching, routing, canary deploys |
| Ray Train | ✅✅ Production | Distributed training across GPU clusters |

---

## The Fundamental Limits (Not Fixable by Software)

### What software CAN fix

| Problem | Solution | Who does it |
|:---|:---|:---|
| Slow Python pickle | Arrow zero-copy serialization | ✅ Ray/Arrow |
| GIL blocking concurrent users | Multi-process workers | ✅ Ray |
| Manual cluster management | Orchestration framework | ✅ Ray/Kubernetes |
| KV cache memory fragmentation | PagedAttention | ✅ vLLM |
| Inefficient batching | Continuous batching | ✅ vLLM |

### What software CANNOT fix

| Problem | Root cause | What would fix it |
|:---|:---|:---|
| MPS→CPU copy (45-306ms) | Apple's GPU memory model | PCIe BAR exposure or CXL |
| CPU→MPS copy (1,424ms) | Metal memory manager overhead | Apple kernel optimization |
| TCP/IP overhead on Thunderbolt | macOS exposes TB as network device | RDMA support (Linux has it) |
| 13% wire utilization (4.8 of 37.6 Gbps) | Python userspace + TCP congestion | Kernel bypass (DPDK) or RDMA |
| Serialization requirement | No shared memory across machines | CXL 3.0 shared memory or NVLink |

### The hardware comparison that explains everything

```
Our setup (macOS + Thunderbolt):
  GPU → CPU → serialize → TCP/IP → deserialize → CPU → GPU
  109 MB × 7 copies = ~2,000ms

NVIDIA H100 cluster (Linux + InfiniBand):
  GPU → NIC (GPUDirect RDMA, zero-copy) → NIC → GPU
  109 MB × 1 copy = ~3ms

Speedup: 667x — and it's ALL hardware/OS, not software
```

---

## Key Takeaways

1. **Arrow is genuinely better than pickle** — 12ms vs 1,370ms for deserialization. If you're transferring tensors in Python, use Arrow (or at minimum, save raw numpy buffers instead of pickling torch tensors).

2. **Ray's value is orchestration, not transport** — for a single transfer, raw TCP is faster. Ray's value is managing thousands of tasks across hundreds of nodes with fault tolerance and auto-scaling.

3. **The GIL tax is real but only at scale** — one user doesn't care. 100 concurrent users care a lot. Ray's multi-process architecture is the only way to serve concurrent users from Python.

4. **macOS is the bottleneck, not Python** — even if we wrote the transfer in C++ with zero-copy Arrow, the MPS↔CPU copies and TCP/IP stack overhead would still dominate. The fix is hardware (RDMA, PCIe BARs, CXL), not software.

5. **Disaggregated inference needs RDMA to work** — the entire premise of splitting prefill/decode across machines assumes cheap KV cache transfer. On macOS over TCP, the transfer overhead exceeds the compute savings. On InfiniBand with GPUDirect RDMA, it works because transfer is nearly free.

---

*Experiment conducted Feb 2026 on MacBook Pro M5 + MacBook Pro M4 Pro connected via Thunderbolt 5. Ray 2.51.2, PyTorch 2.x, Mistral-7B float16.*
