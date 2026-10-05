---
layout: post
title: "Long Prompts Broke My SLO. INT4 Made It Worse."
date: 2026-10-05
categories: [LLM, Inference, Benchmarking]
description: "Benchmarking Laguna-XS.2 on 2x H100: long prompts, not long outputs, break a TTFT SLO, and INT4 makes it worse because Hopper has no native 4-bit compute."
---

Two workloads, both at 10.5 ms per output token. One generates twice as many tokens as the other. Only one of them meets my SLO, and it's the one with the longer output.

You'd expect the decode-heavy workload (500 tokens in, 2,000 out) to be the hard one. Its requests live much longer. But on Laguna-XS.2 on two H100s, the long outputs weren't the problem. The long prompt was.

And INT4 quantization made it worse. H100 has no native 4-bit compute, so INT4 speeds up decode and does nothing for prefill. On this SLO, BF16 beat it.

## Setup

- Model: poolside/Laguna-XS.2, a 33B MoE with 3B active (256 + 1 experts, 40 layers: 30 sliding-window attention, 10 global). INT4 and BF16 versions, FP8 KV cache.
- Hardware: 2× H100 NVL, no NVLink between them.
- Software: vLLM 0.25.1 and AIPerf 0.6.0. Chunked prefill, prefix caching and torch.compile on for every run.
- Topologies: TP=2, PP=2, DP=2 (two independent vLLM processes, round-robin) and TP=2 with expert parallel.
- Load: fixed concurrency at 1, 4, 8, 16, 32 and 64. temperature 0.7, top_k 20, seed 42.
- SLO: TTFT p90 ≤ 500 ms and TPOT p90 ≤ 40 ms/token.

The full sweep went from 200/200 chat all the way up to 64,000-token prompts, in both precisions. DP=2 won most of it. My read is that without NVLink, anything that has to talk between the GPUs (TP, EP) pays for it over PCIe, and DP never talks. But this post is about one comparison from that sweep.

## The comparison

INT4, DP=2, concurrency 64:

| Workload | ISL / OSL | TTFT p90 | TPOT p90 | Meets SLO? |
|---|---|---|---|---|
| Balanced | 1000 / 1000 | 547 ms | 10.5 ms | No |
| Decode-heavy | 500 / 2000 | 326 ms | 10.5 ms | Yes |

Same TPOT. TTFT is the whole difference.

Decode-heavy holds the SLO all the way to concurrency 64 and does 6.11k tokens/s. Balanced falls off after concurrency 32, where it peaks at 3.70k tokens/s of goodput.

## Why doesn't a longer output raise TPOT?

A simple model:

`TPOT ≈ decode step time (depends on active batch size) + prefill interference`

A decode step produces one token for every running request. At concurrency 64, both workloads have about the same number of requests decoding at once. A 2,000-token output means a request stays in the batch longer. It doesn't make each step more expensive.

There's a second reason they match so closely. Each step also has to read every request's KV cache, so context length matters too.

Think about this. Balanced starts at 1,000 tokens of context and ends at 2,000, so it averages about 1,500 while decoding. Decode-heavy starts at 500 and ends at 2,500. Average: about 1,500. Same.

Short chat backs this up. 200 in, 200 out, average context around 300, and TPOT drops to 9.7 ms. Context does matter a little. It just happens to be the same for these two.

So "OSL doesn't matter for TPOT" is too strong. OSL doesn't directly set TPOT. The active batch size does, with a smaller effect from context length.

One qualification: this is a fixed-concurrency test. AIPerf keeps 64 requests in flight no matter how long each one runs. Under a fixed arrival rate, 2,000-token outputs use a lot more GPU time per request. That means less capacity, deeper queues and eventually higher latency. That cost doesn't show up here.

## Why does a longer prompt raise TTFT?

`TTFT ≈ queue wait + prefill + overhead`

At concurrency 64, a new request doesn't land on an idle GPU. It waits for the scheduler, and then the whole prompt has to go through prefill before the first token comes back. Prefill work grows with prompt length.

Prompt goes from 500 to 1,000 tokens. TTFT goes from 326 ms to 547 ms. That's +68%, not 2×, because part of TTFT doesn't grow with the prompt. But it's enough to cross 500 ms.

So "total tokens" is a misleading way to describe a workload. 1000/1000 and 500/2000 are in the same ballpark. But a prompt token sits on the critical path to the first token. An output token doesn't.

## INT4 doesn't help where it counts

Now the same comparison in BF16:

| Workload | ISL / OSL | TTFT p90 | TPOT p90 | Meets SLO? |
|---|---|---|---|---|
| Balanced | 1000 / 1000 | 290 ms | 15.7 ms | Yes |
| Decode-heavy | 500 / 2000 | 245 ms | 15.9 ms | Yes |

BF16 decodes slower (15.7 vs 10.5 ms/token) but gets to the first token much faster (290 vs 547 ms). Both workloads meet the SLO at concurrency 64, and Balanced does 4.14k tokens/s of goodput against INT4's 3.70k.

That isn't a one-off. Here's the whole sweep at concurrency 64, best topology for each metric:

| Workload | ISL / OSL | TTFT p90 (INT4 / BF16) | Throughput (INT4 / BF16) |
|---|---|---|---|
| Short chat | 200 / 200 | 183 / 146 ms | 6.49k / 4.62k tok/s |
| Balanced | 1000 / 1000 | 547 / 290 ms | 6.04k / 4.14k tok/s |
| Decode-heavy | 500 / 2000 | 326 / 245 ms | 6.11k / 4.09k tok/s |
| Prefill-heavy | 6000 / 100 | 1.08 / 0.77 s | 2.24k / 2.08k tok/s |
| Mixed long-context | 5000 / 500 | 809 / 733 ms | 4.53k / 3.37k tok/s |
| Extreme prefill | 16000 / 100 | 2.46 / 1.16 s | 1.03k / 841 tok/s |
| Ultra prefill | 64000 / 100 | 59.8 / 56.4 s | 69.6 / 75.6 tok/s |

BF16 has the lower TTFT on every single shape. INT4 wins throughput by 40–50% where there's plenty of decode. On prefill-heavy (6000/100) that drops to 8%. At 64,000-token prompts BF16 wins outright.

Put the SLO on top and it gets clearer. INT4 wins goodput on short chat and decode-heavy. BF16 wins it on Balanced, mixed long-context (967 vs 619 tokens/s) and extreme prefill. Prefill-heavy is basically a tie (426 vs 414). Once prompts get past about 1,000 tokens, the quantized model is no better, and usually worse.

Why? My read: H100 has no 4-bit tensor cores. It does FP8, INT8, FP16 and BF16 natively, but not INT4. So INT4 weights have to be unpacked to a higher precision before the matmuls run.

1. Decode: memory-bound. Each step has to read the weights (for an MoE, every expert the batch touches) just to produce one token per request. Reading far fewer bytes is a big win, and that's where the 10.5 vs 15.7 ms/token comes from.
2. Prefill: compute-bound. The matmuls are big enough that memory isn't the limit, and they still run at 8- or 16-bit speed. The unpacking is just extra work on the critical path, so TTFT gets worse, not better.

And look at where the headroom was. The TPOT budget was 40 ms and the actual numbers were 10–16 ms. TTFT was the constraint that mattered. INT4 sped up the metric that had room to spare and slowed down the one that didn't.

So on H100, 4-bit quantization is a decode optimization. If your SLO is bound by TTFT, it can cost you.

## Why I want to run this on Blackwell

This is exactly the gap Blackwell is supposed to close. Its tensor cores support NVFP4 natively, for both weights and activations. So a 4-bit model doesn't have to unpack its way through prefill. The matmuls themselves can run in 4-bit.

If that holds up, quantization stops being decode-only. You'd keep the 4-bit decode speed and get faster prefill too, instead of paying for it in TTFT.

The question I care about isn't "how much more throughput?". It's "at what prompt length and concurrency does TTFT cross 500 ms now?" Maybe Balanced stays under it at concurrency 64. Maybe the point where prefill-heavy traffic stops being practical moves a long way out.

That needs measuring, not guessing from H100 numbers. Next experiment.

## What I took away

1. Under fixed concurrency, a long output doesn't make each token slower. TPOT follows the active batch size.
2. A long prompt goes straight onto TTFT. That's what broke the SLO.
3. On H100, INT4 is a decode optimization. No native 4-bit compute means no prefill speedup, and TTFT gets worse.
4. Check which constraint is actually binding before picking a precision.
5. Describe workloads by ISL and OSL separately, never by total tokens.

Long outputs share the decode cost. Long prompts pay theirs up front. And on H100, 4-bit doesn't help with that bill.
