---
layout: post
title: "TTFT Doubled. The GPU Didn't Notice."
date: 2026-10-05
categories: [LLM, Inference, Benchmarking]
---

84 ms to 167 ms. That's what happened to time to first token (TTFT) when I changed one setting. It wasn't a GPU setting. It was the CPU frequency governor.

Same vLLM endpoint, same model, same [AIPerf](https://github.com/ai-dynamo/aiperf) command. The only change was switching the governor from `performance` to `powersave`. Decode speed didn't move. vLLM's own prefill time didn't get worse. Client TTFT almost doubled anyway.

Here's the catch. AIPerf's TTFT and its `effective_*` and `active_*` metrics are measured from the client. They include all the CPU work between "send the request" and "first token arrives": the AIPerf worker, HTTP, vLLM's API server, tokenization. That isn't GPU prefill. Read them as GPU prefill and a slow CPU will mislead you.

## Setup

- Host: dual-socket Intel Xeon Gold 6448Y, 64 cores / 128 threads. Governor changed on all 128.
- GPU: one NVIDIA L40S running vLLM. OpenAI chat endpoint, streaming, `max_model_len` 4096.
- Client: AIPerf with 32 workers (`--concurrency 32`), not pinned to individual cores. Same host as vLLM but on the other socket, so it isn't fighting vLLM for cores. It hits vLLM at `127.0.0.1`.
- The only change: `cpupower frequency-set -g performance` vs `-g powersave`. The box normally runs `schedutil`.

What the CPUs actually ran at:

| Governor | Average | Range |
|---|---|---|
| performance | 3.7–4.0 GHz | max 4.1 GHz |
| powersave | ~800 MHz | min ~756 MHz |

About 5× slower. Keep that number in mind.

The command, identical for both runs:

```bash
aiperf profile \
  --model <model_name> \
  --tokenizer <model_tokenizer> \
  --endpoint-type chat \
  --url http://127.0.0.1:8000 \
  --streaming \
  --concurrency 32 \
  --request-count 128 \
  --warmup-request-count 32 \
  --synthetic-input-tokens-mean 256 \
  --synthetic-input-tokens-stddev 0 \
  --output-tokens-mean 128 \
  --output-tokens-stddev 0 \
  --extra-inputs ignore_eos:true \
  --random-seed 42 \
  --ui none
```

256 tokens in, exactly 128 out (`ignore_eos` makes every request generate all 128), 128 requests after 32 warmup.

## What AIPerf saw

| Client metric | performance | powersave | Change |
|---|---|---|---|
| TTFT avg (p50) | 83.87 ms (78.6) | 166.50 ms (153.2) | +99% |
| `http_req_waiting` (TTFB) | 83.63 ms | 165.80 ms | +98% |
| `http_req_sending` | 0.17 ms | 0.49 ms | 3.0× |
| `http_req_blocked` | 0 | 0 | |
| `credit_to_start_latency` | 0.80 ms | 2.05 ms | 2.6× |
| Request latency | 936.9 ms | 1015.7 ms | +8.4% |
| ITL | 6.84 ms | 6.80 ms | flat |
| Decode duration | 853.0 ms | 849.2 ms | flat |
| Output tokens/s | 4273 | 3927 | −8.1% |
| Requests/s | 33.96 | 31.17 | −8.2% |
| Benchmark duration | 3.77 s | 4.10 s | +8.8% |
| `effective_concurrency` avg (max) | 31.82 (32) | 31.66 (32) | |
| `effective_prefill_concurrency` avg | 2.85 | 5.19 | up |
| `effective_decode_concurrency` avg | 28.97 | 26.47 | |

ITL and decode are flat, so the GPU was generating tokens at the same speed. TTFT doubled and tracked `http_req_waiting` (time to first byte) almost 1:1.

The ~8% throughput drop is the same story. Request latency went up by about 79 ms, and TTFT went up by about 83 ms. All of the extra time is before the first token.

## What vLLM saw

| Server metric (vLLM `/metrics`) | performance | powersave | Change |
|---|---|---|---|
| `request_prefill_time` avg | 50.4 ms | 42.6 ms | −15% |
| `request_queue_time` avg | 7.82 ms | 0.07 ms | down |
| `request_decode_time` avg | 853.4 ms | 842.3 ms | flat |
| `num_requests_running` p50 / max | 32 / 32 | 32 / 32 | |
| `num_requests_waiting` max | 1 | 0 | |

Prefill time went *down*. Queue time went to basically zero. 32 requests running the whole time.

From the engine's side, powersave was, if anything, a slightly easier run. My read: the slower client spread the requests out (more on that below), so fewer of them landed in the same prefill batch and nothing had to wait.

And no, `num_requests_waiting` at 0 doesn't mean we missed the target concurrency. 32 were running. Nothing was queued.

## Where did the time go?

Think about this. Take client TTFT and subtract what vLLM says it spent on prefill and queueing:

- performance: 83.87 − 50.4 − 7.82 ≈ **26 ms**
- powersave: 166.50 − 42.6 − 0.07 ≈ **124 ms**

That leftover is the CPU work outside the engine's prefill: AIPerf building and sending the request, HTTP, vLLM's API server, tokenization. It went up about 4.8×. The clock went down about 5×. The leftover scaled with the CPU clock.

## The prefill concurrency trap

`effective_prefill_concurrency` went from 2.85 to 5.19. That looks like more requests sitting in prefill on the GPU at the same time.

It isn't. AIPerf treats the window from request start to first token, measured on the client's clock, as "prefill". TTFT gets longer, the window gets longer, more requests overlap in it. Meanwhile actual GPU prefill time went down.

The AIPerf maintainers confirmed this in [issue #1424](https://github.com/ai-dynamo/aiperf/issues/1424): `effective_*` and `active_*` are all client-based. Only the `server_metrics` exports use server timestamps. They also pointed out that if the client doesn't have enough CPU, it can fall behind the target concurrency or request rate and inflate the client metrics.

## Did the client still hit 32?

Yes. But it got there more slowly.

The concurrency burst still issues credits one at a time. Each request yields to the event loop and opens a new `aiohttp.ClientSession`. So bring-up is staggered, and a slower CPU staggers it more.

| First 32 profiling requests | performance | powersave |
|---|---|---|
| Time to issue / start all 32 | 1.16 / 1.19 ms | 4.12 / 4.10 ms |
| HTTP requests started in the first 1 ms | 26 / 32 | 8 / 32 |
| Unique workers | 32 | 32 |
| Time spent at 32 in flight | 98.2% | 97.2% |

Both runs reached 32 in flight and stayed there for 97–98% of the run. The difference is in the first few milliseconds: 26 of 32 requests started within 1 ms on `performance`, only 8 on `powersave`.

## What I do now

1. Run the AIPerf host on `performance` for any GPU or engine read. `schedutil` is probably fine, but I haven't measured it. Never `powersave`.
2. For engine prefill and queue time, use `server_metrics`, not client TTFT or `effective_prefill_concurrency`.
3. When TTFT moves, check ITL and server-side prefill before blaming the GPU. If those are flat, look at the CPU.

Same GPU, same work, twice the TTFT. Check the governor.
