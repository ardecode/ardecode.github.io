---
layout: post
title: "620 Tokens per Second, No GPU. Tuning vLLM on CPUs"
date: 2026-11-04
categories: [LLM, Inference, Benchmarking]
description: "Ten vLLM runs serving a 3B model on 32 Granite Rapids vCPUs: what moved tail TTFT (KV cache size, prompt chunking, sequence limits, core 0) and what didn't."
---

**32 vCPUs, 128 concurrent users, one 3B model, and 620 output tokens/sec.** That's what I achieved running vLLM on an Intel Xeon 6 (Granite Rapids) CPU.

I wanted to understand how much tuning vLLM's arguments and environment variables could improve CPU inference performance compared to the default configuration.

The workload was about 2,048 input tokens and 256 output tokens, tested at 64 and 128 concurrent users.

**Did tuning help? Yes, but mostly with latency, not throughput.**

At 64 users, the default configuration delivered 581 tokens/sec, while the tuned configurations achieved 548–554 tokens/sec. However, P95 time to first token (TTFT) improved from 2.0 seconds to 1.6–1.8 seconds.

At 128 users, the tuned setup reached 620 tokens/sec with zero failed requests.

**Is that good enough? It depends on the workload.**

For summarization, yes. At 128 users, 95% of requests completed within roughly 31 seconds. Since users typically wait for the complete summary, TTFT matters less.

For interactive chat, 128 users is too much. P95 TTFT reached 4.7 seconds, with fewer than 5 output tokens/sec per user. For chat, I'd stick to 64 concurrent users.

**What did I learn from 10 benchmark runs?**

Most improvements came from avoiding bad configurations. Reducing the prompt processing batch size, allowing too many concurrent sequences, or assigning an OpenMP thread to core 0 increased P95 TTFT by 60–90%. Properly sizing the KV cache reduced it by around 20%, while one commonly recommended OpenMP setting barely made a difference.

**The biggest takeaway:** CPU inference tuning isn't always about maximizing throughput. Sometimes, the real improvement is reducing tail latency while giving up about 5% throughput.

## Setup

- Host: dual-socket Intel Xeon 6730P (Granite Rapids), 32 cores per socket, SMT on. Socket 0 is CPUs 0–31 with siblings 64–95. Socket 1 is CPUs 32–63 with siblings 96–127. AMX with BF16, AVX-512.
- Kubernetes (MicroK8s). The vLLM pod gets a hard quota: 32 vCPUs and 64 GiB RAM, requests equal to limits.
- vLLM v0.20.0, CPU backend, with Intel OpenMP (`libiomp5`) and oneDNN.
- Model: [Ministral-3-3B-Base-2512](https://huggingface.co/mistralai/Ministral-3-3B-Base-2512) in bfloat16, about 7.2 GB of weights.
- Workload: about 2,048 tokens in, 256 tokens out. Results below are at 64 and 128 concurrent streams.
- Load generator: [AIPerf](https://github.com/ai-dynamo/aiperf), with a warmup before each run.

## The ten runs

RUN-01 is the vanilla deployment. "KV" is `VLLM_CPU_KVCACHE_SPACE` in GiB, "seqs" is `--max-num-seqs`, "batched" is `--max-num-batched-tokens`.

| Run | Users | Config | Throughput | TTFT P50 | TTFT P95 | Req latency P95 |
|---|---|---|---|---|---|---|
| 01 | 64 | KV=4, seqs=256, batched=default | 581.03 tok/s | 514.6 ms | 2,003.2 ms | 15.96 s |
| 02 | 64 | KV=20, seqs=64, batched=2048 | 547.34 tok/s | 670.7 ms | **1,604.7 ms** | 17.35 s |
| 03 | 64 | KV=20, seqs=128, batched=1024 | 514.56 tok/s | 693.8 ms | 2,918.9 ms | 19.71 s |
| 04 | 64 | KV=20, seqs=256, batched=2048 | 531.18 tok/s | 615.9 ms | 3,054.3 ms | 17.37 s |
| 05 | 64 | KV=20, seqs=64, batched=default | 516.46 tok/s | 757.9 ms | 2,989.6 ms | 17.85 s |
| 06 | 64 | 02 + `KMP_AFFINITY`, `OMP_PROC_BIND` | hung in warmup | | | |
| 07 | 64 | 02 + `KMP_BLOCKTIME=0` | 553.83 tok/s | 776.9 ms | 1,816.2 ms | 16.77 s |
| 08 | 64 | 07 + threads bound to CPUs 0–30 | 552.85 tok/s | 734.2 ms | 2,967.2 ms | 16.77 s |
| 09 | 128 | KV=35, seqs=128, batched=2048, blocktime 0 | **620.04 tok/s** | 1,702.4 ms | 4,726.1 ms | 30.75 s |
| 10 | 64 | same config as 09 | 548.18 tok/s | 948.3 ms | 1,808.1 ms | 16.91 s |

Now let's go through what each change did.

## 1. Give the KV cache enough room

On the CPU backend, vLLM doesn't size the KV cache from free memory like it does on a GPU. You tell it with `VLLM_CPU_KVCACHE_SPACE`, in GiB. The baseline had 4.

Is 4 GiB enough for 64 users? Think about this. Bytes of KV per token:

```
2 (K and V) × 28 layers × 8 KV heads × 128 head dim × 2 bytes = 114,688 bytes
```

One request at 2048 + 256 = 2304 tokens is 2304 × 114,688 bytes, which is exactly 252 MiB. 4 GiB holds 16 of those. Leave about 10% for fragmentation and it's 14.

14 slots for 64 streams. So the scheduler keeps preempting running requests and recomputing their prefill later. P95 TTFT went past 2 seconds.

At 20 GiB (RUN-02), P95 TTFT dropped from 2,003 ms to 1,605 ms, **−19.9%**.

But be careful with that comparison. RUN-01 to RUN-02 also changed `max-num-seqs` (256 to 64) and `max-num-batched-tokens` (default to 2048). And RUN-01 actually had higher throughput and a better median TTFT. The tail is what got fixed.

## 2. Don't slice the prompt

`--max-num-batched-tokens` caps how many tokens go through one engine step. If a prompt is longer than that, chunked prefill splits it across steps.

On a GPU that's usually what you want. Long prompts stop blocking everyone else's decode. On this CPU it hurt.

The clean comparison is RUN-02 vs RUN-05. Same everything, except batched tokens went from 2048 to the default (512 in this build, so four slices per prompt):

- Throughput: 547 → 516 tok/s (−5.6%)
- P95 TTFT: 1,605 → 2,990 ms (+86%)

RUN-03 (1024, two slices) landed in the same place, 2,919 ms. It also changed seqs to 128, so I don't treat it as a clean data point.

Why? My read: every engine step on the CPU is a set of OpenMP parallel regions with barriers, and AMX GEMMs that are most efficient on big tiles. Four steps of 512 tokens pay that fixed cost four times and run smaller matrices each time. I didn't profile this, so treat it as a hypothesis.

## 3. Don't let in more sequences than you can serve

RUN-04 is RUN-02 with `--max-num-seqs` at 256 instead of 64. Only 64 users, so how much can it matter?

A lot. P95 TTFT 1,605 → 3,054 ms (+90%).

I don't have a good explanation for this one. With 64 users, a cap of 256 shouldn't let more than 64 in at once. But that's what the run showed, so I set `max-num-seqs` to my real concurrency and moved on.

## 4. KMP_BLOCKTIME=0 is not the hero

Intel OpenMP keeps worker threads spinning for a while after a parallel region ends (`KMP_BLOCKTIME`, 200 ms by default), so they're ready for the next one. In a pod with a hard 32-vCPU quota, that spinning eats into the same quota the HTTP server and the asyncio loop need. Setting it to 0 makes workers sleep right away.

It shows up in most CPU tuning advice. So did it help? Here's RUN-02 vs RUN-07, where that's the only change:

| | RUN-02 | RUN-07 (`KMP_BLOCKTIME=0`) |
|---|---|---|
| Throughput | 547.34 tok/s | 553.83 tok/s (+1.2%) |
| TTFT P50 | 670.7 ms | 776.9 ms |
| TTFT P95 | 1,604.7 ms | 1,816.2 ms (+13%) |

Throughput up 1%. Tail TTFT worse. I kept it in the final config because it makes sense in a CPU quota, but from one run each I can't say it did anything. That's noise until I repeat it.

## 5. Stay off core 0

RUN-06 tried the textbook pinning, `KMP_AFFINITY` and `OMP_PROC_BIND`. The server hung during the warmup GEMM and never came up. My guess is the affinity masks fought with the cgroup CPU quota, but I didn't dig in.

So for RUN-08 I used vLLM's own knob instead: `VLLM_CPU_OMP_THREADS_BIND="0-30"`, 31 threads on physical cores 0–30 of socket 0. Looks like the cleanest choice.

Throughput didn't move (553 vs 553 tok/s). P95 TTFT went from 1,816 ms to 2,967 ms. **+63%.**

What was vLLM doing on its own? With the default `auto`, it had put the threads on CPUs 64–94. Those are the SMT siblings of 0–30, still socket 0.

The difference is core 0. My read: on Linux, CPU 0 is where a lot of housekeeping lands: timer ticks, network softirqs (`ksoftirqd`), containerd I/O. Put one OpenMP worker there and every time it gets interrupted, the other 30 wait for it at the next barrier. One slow thread sets the pace for the whole GEMM.

I didn't check `/proc/interrupts` to confirm it, so this is the likely cause, not a measured one. But the +63% is measured.

Leave binding on `auto`, or if you pin, keep CPU 0 out of the set.

## 64 users or 128?

RUN-09 took the same box to 128 users with a 35 GiB KV cache and `max-num-seqs=128`. Best throughput of the series, 620 tok/s.

Then RUN-10 ran that exact config at 64 users. Same P95 TTFT as RUN-07 (1,808 vs 1,816 ms), 1% lower throughput. P50 was higher (948 vs 777 ms). So one config works for both loads, and you don't have to redeploy when traffic changes.

What do 128 users pay for that 12% extra throughput?

| | 64 users (RUN-07) | 128 users (RUN-09) |
|---|---|---|
| Throughput | 553.83 tok/s | 620.04 tok/s |
| TTFT P50 | 777 ms | 1,702 ms |
| TTFT P95 | 1,816 ms | 4,726 ms |
| Tokens/s per user (throughput ÷ users) | ~8.7 | ~4.8 |
| P95 request latency, 256 tokens out | 16.77 s | 30.75 s |
| Failed requests | 0 | 0 |

The per-user speed is derived, throughput divided by users, not a measured ITL.

My read: 64 for anything a person is watching. P95 TTFT under 2 s and 8+ tokens per second is fine for chat. 128 for batch jobs, RAG ingestion, summarization, where nobody is waiting on the first token and total throughput is what you pay for.

## Sizing it for another model

The same arithmetic works for any model and quota.

1. KV per token: `2 × layers × kv_heads × head_dim × dtype_bytes`, from the model's `config.json`.
2. KV cache: `ceil(users × (ISL + OSL) × kv_per_token × 1.10 / 1024³)` GiB.
3. Pod memory: `weights + ~6 GiB runtime + KV cache` must fit in the limit.
4. Batched tokens: the next power of two at or above ISL, so a prompt never gets sliced.
5. `max-num-seqs`: your peak concurrency, no more.
6. `OMP_NUM_THREADS`: the vCPU quota.

For this model at 2048/256:

- 64 users: 64 × 252 MiB × 1.1 = 17.3 GiB, round up to 18. I ran 20.
- 128 users: 128 × 252 MiB × 1.1 = 34.7 GiB, so 35.
- Memory at 128 users: 7.2 + 6 + 35 = 48.2 GiB. 15.8 GiB of headroom under 64.

The 6 GiB runtime number (PyTorch allocator, oneDNN scratchpads, AMX buffers) is an estimate, not a formula.

## The config

The vLLM part of the final pod:

```yaml
args:
  - --dtype=bfloat16
  - --max-model-len=4096
  - --max-num-seqs=128
  - --max-num-batched-tokens=2048
  - --disable-sliding-window
  - --enable-prefix-caching
env:
  - name: OMP_NUM_THREADS
    value: "32"
  - name: VLLM_CPU_KVCACHE_SPACE
    value: "35"
  - name: KMP_BLOCKTIME
    value: "0"
  - name: VLLM_ENGINE_ITERATION_TIMEOUT_S
    value: "600"
resources:
  requests: { cpu: "32", memory: 64Gi }
  limits:   { cpu: "32", memory: 64Gi }
```

Plus a `nodeSelector` on `feature.node.kubernetes.io/cpu-cpuid.AMXBF16: "true"` so it only lands on AMX nodes.

## What I'd do next time

1. KV cache first: do the math before the first run. 4 GiB for 64 long-prompt users was never going to work.
2. Batched tokens at least ISL: on CPU, slicing the prompt cost 86% on tail TTFT.
3. `max-num-seqs` equal to real concurrency: 256 on 32 cores doubled P95 TTFT.
4. No threads on CPU 0: leave binding to vLLM, or exclude 0 yourself.
5. Repeat runs: some of these deltas (`KMP_BLOCKTIME`) are inside what one run can tell you.

A 3B model on 32 CPU cores, 64 users, under 2 seconds at P95. Just don't put a thread on core 0.
