---
layout: post
title: "My KMP_BLOCKTIME=0 Never Reached the Worker. Four Bugs in vLLM's CPU Backend"
date: 2026-11-14
categories: [LLM, Inference, vLLM]
description: "Four bugs in vLLM's CPU backend: your OpenMP settings replaced in the worker, page cache counted as used memory, workers over-reserving a container's memory limit, and a NUMA fallback that's ignored. Two fixes are up as PRs."
---

**`KMP_BLOCKTIME=0` in my config. `KMP_BLOCKTIME=1` in the worker.** The setting I tuned in my last two posts never reached the process that runs the model.

After the [Granite Rapids post]({% post_url 2026-11-04-vllm-cpu-inference-granite-rapids %}) and the [hugepages post]({% post_url 2026-11-14-hugepages-cpu-inference %}), I read through vLLM's CPU backend to see what my settings actually did. I found four bugs:

1. Your `KMP_BLOCKTIME` and `KMP_TPAUSE` are replaced in the worker.
2. In a container, page cache is counted as used memory, so vLLM can refuse to start.
3. With more than one worker, each one reserves memory as if it had the container to itself.
4. A NUMA fallback is picked at startup and then ignored.

I reproduced 1 and 2 on small AWS instances and opened PRs for them. 3 and 4 have unit tests and a PR that's almost ready.

## 1. Your KMP_* settings don't reach the worker

vLLM runs the model in a separate worker process. Before starting it, vLLM writes the OpenMP settings the worker needs to pin its threads: `KMP_AFFINITY` (which CPUs) and `OMP_NUM_THREADS` (how many threads). That part is fine.

But it also wrote `KMP_BLOCKTIME` and `KMP_TPAUSE`, every time. Whatever you set was thrown away.

When does this happen? When thread binding is on, which is the default, and Intel OpenMP is loaded, which the CPU Docker image does. With `VLLM_CPU_OMP_THREADS_BIND=nobind`, vLLM leaves your settings alone.

How do you check? Not with `/proc/<pid>/environ`. vLLM renames the worker to `VLLM::Worker`, and after that most of its environment doesn't show there. Set `KMP_SETTINGS=1` instead. Intel OpenMP then prints the settings each process actually uses.

I started the nightly CPU image with `KMP_BLOCKTIME=0 KMP_TPAUSE=1 KMP_SETTINGS=1` on a c7i.2xlarge:

| Process | As shipped | With the fix |
|---|---|---|
| API server | `KMP_BLOCKTIME=0ms`, `KMP_TPAUSE=1` | same |
| EngineCore | `KMP_BLOCKTIME=0ms`, `KMP_TPAUSE=1` | same |
| Worker | `KMP_BLOCKTIME=5ms`, `KMP_TPAUSE=0` | `KMP_BLOCKTIME=0ms`, `KMP_TPAUSE=1` |

The two processes that don't run the model get your values. The one that does gets vLLM's. v0.33.0 does the same. v0.20.0, the version my last two posts used, writes `1` instead of `5` (I read that in the source, I didn't measure it).

So what about those posts?

1. Granite Rapids: section 4 compared `KMP_BLOCKTIME=0` with the default. Both actually ran 1 ms. I called the difference noise, and it was, just not for the reason I gave. `OMP_NUM_THREADS` didn't apply there either.
2. Hugepages: every run had the same 1 ms, so the hugepage result holds. The numbers are for 1 ms, not 0.

Both posts now have an update note.

The fix: keep setting `KMP_AFFINITY` and `OMP_NUM_THREADS`, and set `KMP_BLOCKTIME` and `KMP_TPAUSE` only if you haven't. [vllm-project/vllm#67726](https://github.com/vllm-project/vllm/pull/67726).

Does `KMP_BLOCKTIME=0` help CPU inference, now that it can actually be set? I don't know yet. Next test.

## 2. Page cache counted as used memory

When you read a file, Linux keeps a copy in memory, the page cache. Inside a container that copy counts against the container's memory limit. But it isn't really used. The kernel drops it as soon as something else needs the space.

vLLM didn't know that. In a container, it worked out free memory as limit minus everything charged to the container, page cache included. And right after loading a model, there's a lot of page cache: the weight files it just read.

Think about this. Qwen2.5-7B in a container with a 24 GiB limit, asking for a 4 GiB KV cache, on a c7i.4xlarge. Right after loading:

```
charged to the container   23.4 GiB
  real memory              18.7 GiB   (mostly the weights)
  page cache                4.6 GiB   (the weight files, read from disk)
```

vLLM saw 24.0 - 23.4 = 0.56 GiB free and refused to start:

```
ValueError: Available memory on node 0 (0.56/24.0 GiB) on kv cache allocation is less than requested memory for kv (4.0 GiB).
```

Count the page cache as free and it's about 5.3 GiB. Enough for 4.

With the fix it started, with room for 74,880 tokens of KV cache. I sent 16 requests of about 3.5k tokens each, about 59k tokens of KV. All 16 came back. No out-of-memory kill. At the end the page cache was down to 0.8 GiB. The kernel dropped it as the KV cache grew.

Why count all of the page cache? Linux keeps it on two lists, active and inactive. `docker stats` only counts the inactive list as free. Right after loading, most of it is on the active list (3.4 of 4.6 GiB here), so counting only inactive gives about 1.8 GiB. Still not enough. vLLM already counts both lists when there's no container limit. The container path just didn't.

[vllm-project/vllm#67727](https://github.com/vllm-project/vllm/pull/67727).

## 3. Workers over-reserve the container's memory

On a machine with two sockets, you can run one vLLM worker per socket (tensor parallel 2). Each worker reserves 90% of its socket's memory (`gpu_memory_utilization`, 0.9 by default), capped at the container's limit.

The cap is per worker. The limit is for the whole container.

Two sockets with 96 GiB each, a container limited to 64 GiB:

```
worker 0:  0.9 × 64 GiB = 57.6 GiB
worker 1:  0.9 × 64 GiB = 57.6 GiB
together:                 115.2 GiB, in a 64 GiB container
```

Each worker sizes its KV cache to fill its reservation. So the server starts fine, and then gets killed for running out of memory once the caches fill up under load.

The fix splits the limit between the workers first. Here that's 32 GiB each, so 0.9 × 32 = 28.8 GiB per worker. With one worker, nothing changes.

I only unit-tested this one. I don't have a two-socket machine. On v0.33.0 the test fails with exactly the numbers above.

## 4. A NUMA fallback that's picked and then ignored

A worker looks at memory twice:

1. At startup: which memory node am I on, and how much can I reserve?
2. Before sizing the KV cache: how much is still free on that node?

Both should look at the same node. They didn't.

On IBM s390x, the "NUMA node" vLLM reads for a CPU is really a hardware id, not a memory node. Say the worker's CPUs report 7, but the only node with memory is 0.

1. Startup sees node 7 has no memory, logs "Falling back to memory node 0", and reserves 90% of node 0. Fine.
2. The KV cache step forgets that and asks about node 7 again. Node 7 doesn't exist, so vLLM quietly uses the whole machine's numbers instead.

So the reservation comes from node 0, the free-memory check comes from the whole machine, and any error message names node 7. With one node, the numbers come out about the same. With more than one, the check sees more free memory than the worker's node has.

The unit test records which node each step asks about. On v0.33.0 it gets `[0, 7]` instead of `[0, 0]`. The fix remembers the node chosen at startup and uses it both times. About ten lines.

Who hits this? Mostly s390x. My read: it hurts debugging more than sizing, since the errors point at a node with no memory.

Bugs 3 and 4 are in the same file, so they're one PR. The tests need an x86 machine with AVX-512, which my Mac isn't. I'll open it once they've run there.

## Where it stands

| Bug | Reproduced | PR |
|---|---|---|
| 1. KMP_* replaced in the worker | Yes, v0.33.0 and nightly | [#67726](https://github.com/vllm-project/vllm/pull/67726), open |
| 2. Page cache counted as used | Yes, v0.33.0 | [#67727](https://github.com/vllm-project/vllm/pull/67727), open |
| 3. Workers over-reserve memory | Unit test | Not opened yet |
| 4. NUMA fallback ignored | Unit test | Same PR as 3 |

All the AWS runs together cost under a dollar.

Check the worker, not the launcher.
