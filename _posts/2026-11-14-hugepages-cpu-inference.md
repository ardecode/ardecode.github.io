---
layout: post
title: "10x Fewer TLB Misses, 2.5% More Tokens. Hugepages on CPU Inference"
date: 2026-11-14
categories: [LLM, Inference, Benchmarking]
description: "Nine vLLM runs on a Granite Rapids VM with transparent hugepages off, madvise and always: TLB misses dropped 8 to 10x, throughput moved 2.5%, TTFT didn't move."
---

**10x fewer TLB misses. 2.5% more tokens per second.** That's what turning on hugepages did for vLLM on a Granite Rapids CPU.

After the [last post]({% post_url 2026-11-04-vllm-cpu-inference-granite-rapids %}) I kept wondering if hugepages would help CPU inference. Decode on a CPU is memory-bound, and the KV cache is a 35 GiB buffer. So I tested it.

**Did it help? Yes, a little.** Throughput up 2.4 to 2.6%, inter-token latency (ITL) down by the same.

**Did it help TTFT? No.** Flat, within 1%.

**Do you need `always`? No.** The `madvise` setting, which was already the default on this Ubuntu image, gave the same result.

## Why hugepages might matter

Every memory access goes through a virtual-to-physical address translation. The CPU caches recent translations in the TLB. With normal 4 KiB pages, the TLB covers a few MB of memory. With 2 MiB hugepages it covers a few GB.

Now think about one decode step for 32 users at about 2,200 tokens of context each:

```
KV per token:  2 × 28 layers × 8 KV heads × 128 head dim × 2 bytes = 114,688 bytes
KV per step:   32 × ~2,200 × 114,688 bytes ≈ 8 GB
Weights:       7.2 GB
```

About 15 GB read per step, spread over tens of GB of memory. With 4 KiB pages that's millions of pages. A TLB miss means a page walk, and a page walk means more memory reads before the real read can start.

That's the theory. Here's what happened.

## Setup

I don't have a Xeon to test on, so I rented one.

- AWS `c8i.16xlarge` in us-east-2: Intel Xeon 6975P-C (Granite Rapids), 1 socket, 32 cores / 64 vCPUs, 123 GiB RAM. Ubuntu 24.04, kernel 7.0.
- vLLM v0.20.0 CPU image in Docker, pinned to 32 vCPUs (cores 1–16 and their SMT siblings), core 0 left out.
- vLLM's `auto` binding put 15 OpenMP threads on those cores, one per physical core, and reserved one. The last post's setup had 31.
- Model: [Ministral-3-3B-Base-2512](https://huggingface.co/mistralai/Ministral-3-3B-Base-2512), bfloat16, 7.2 GB.
- Same vLLM config as RUN-10 in the last post: `VLLM_CPU_KVCACHE_SPACE=35`, `--max-num-seqs=128`, `--max-num-batched-tokens=2048`, `KMP_BLOCKTIME=0`, prefix caching on. Plus `THP_MEM_ALLOC_ENABLE=1`, which tells PyTorch's CPU allocator to ask for hugepages on large allocations.- Load: [AIPerf](https://github.com/ai-dynamo/aiperf) 0.13.0 pinned to the other 15 cores. 32 concurrent users, exactly 2,048 tokens in and 256 out, 32 warmup requests, then 320 measured.

So do vLLM and AIPerf share cores? No. Here's the split. On this VM, CPU N and CPU N+32 are SMT siblings on the same physical core.

| Who | CPUs | Physical cores |
|---|---|---|
| vLLM container (`--cpuset-cpus`) | 1–16, 33–48 | 1–16 |
| vLLM OpenMP threads (`auto` binding) | 33–47 | 1–15 |
| Reserved by vLLM | 48 | 16 |
| AIPerf (`taskset -c`) | 17–31, 49–63 | 17–31 |
| OS only | 0, 32 | 0 |

They still share the socket's L3 cache and memory bandwidth. AIPerf's traffic is light, and it's the same in every run, so it can't favour one THP setting.

Every run got a fresh vLLM container, with the page cache dropped and memory compacted first, so the hugepage setting applied from the first allocation.

The only thing that changed between runs was the kernel's transparent hugepage (THP) setting:

| Setting | `enabled` | `defrag` | What it means |
|---|---|---|---|
| never | never | never | No hugepages. Baseline. |
| madvise | madvise | defer+madvise | Hugepages only for memory that asks for them |
| always | always | defer+madvise | Hugepages for any large enough anonymous memory |

Three repeats each, nine runs in total. The order rotated every repeat, so slow drift on the machine couldn't favour one setting.

## Results

Mean of three runs, with the min–max range for throughput:

| THP | Throughput | ITL P50 | TTFT P50 | TTFT P95 | Req latency P95 |
|---|---|---|---|---|---|
| never | 128.4 tok/s (128.1–128.6) | 236.7 ms | 2.98 s | 16.08 s | 77.2 s |
| madvise | 131.5 tok/s (131.4–131.7) | 231.0 ms | 2.97 s | 16.02 s | 75.5 s |
| always | 131.8 tok/s (131.4–132.4) | 230.6 ms | 2.98 s | 15.97 s | 75.3 s |

And what was going on underneath:

| THP | Memory on hugepages | dTLB misses per 1,000 instructions |
|---|---|---|
| never | 0 of 47 GiB | 0.62 |
| madvise | 43.3 of 47 GiB (92%) | 0.06 |
| always | 45.9 of 47 GiB (97%) | 0.08 |

Memory on hugepages is `AnonHugePages` from `/proc/<pid>/smaps_rollup`, summed over the container's processes. TLB misses are from `perf stat` over 60 seconds of steady load.

## What the numbers say

1. Hugepages were actually used: 92% of vLLM's memory with `madvise`, 97% with `always`. If you turn THP on and don't check `AnonHugePages`, you don't know this.
2. TLB misses dropped 8 to 10x: from 0.62 per 1,000 instructions to 0.06–0.08.
3. Throughput went up 2.4% (`madvise`) and 2.6% (`always`): the three `never` runs sit at 128.1–128.6 tok/s, and all six hugepage runs sit at 131.4–132.4. No overlap.
4. ITL went down by the same amount: 236.7 ms to about 231 ms. Request latency P95 dropped 2.2 to 2.6%, which is mostly the ITL change carried over 256 output tokens.
5. TTFT didn't move: P50 and P95 are within 1% across all three settings.
6. `madvise` and `always` are the same within noise.

## Why so small?

10x fewer TLB misses, only 2.5% faster. Why?

Because TLB misses were already rare. 0.62 per 1,000 instructions is low. Even if every miss cost a full page walk, that's not much of the total time. Cutting it by 10x can only give back what it was costing, and it wasn't costing much.

And why does only decode benefit? TTFT here is mostly queueing plus prefill. Prefill is big matrix multiplies on AMX, which is compute-bound. Decode is the memory-bound part, and that's where the 2.5% shows up.

## Does every workload get 2.5%?

No. It depends on where your time goes.

Think about this. By token count, this workload is prefill-heavy: 2,048 in, 256 out. But by time it's decode-heavy. A request waited about 16 s for its first token at P95, then spent about 255 × 231 ms ≈ 59 s generating. Most of the time was decode, so most of the 2.5% reached the user. Request latency P95 dropped 2.2 to 2.6%.

1. Time mostly in decode (chat, long answers, long summaries): expect close to the full 2.5% end to end.
2. Time mostly in prefill (long prompts with short answers, like classification, extraction or scoring): expect close to nothing. TTFT didn't move in any run.
3. Longer contexts: more KV read per decode step, spread over more memory. TLB pressure could go up, and the gain with it. My read, not tested.

Hugepages help decode, a little. Prefill, not at all.
