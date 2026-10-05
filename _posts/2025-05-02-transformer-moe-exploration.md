---
layout: post
title: "From Self-Attention to Mixture of Experts, One Question at a Time"
date: 2025-05-02
categories: [transformers, MoE, deep-learning]
---

I wanted to understand Mixture of Experts. Turned out I didn't fully understand the layer it replaces. So I went back to the start and asked the basic questions in order.

All the numbers below assume a hidden size of 768.

## What does self-attention give you?

Self-attention decides which tokens should pay attention to which. Each token comes out as a 768-dim vector that now carries some context from the others. 10 tokens in, a 10 × 768 matrix out.

That matrix goes to the feed-forward network (FFN).

## Do all 10 tokens go into one FFN together?

This is where I got stuck. No.

Each of the 10 vectors goes through the FFN separately. Same FFN, same weights, applied to every token. Not 10 different networks, and the tokens can't see each other at this step. In practice it's one matrix multiply over the whole 10 × 768 block, so it's still fast.

When papers say the FFN is applied "independently", this is all they mean. Same network, no talking between tokens.

## What does the FFN actually do?

Expand, apply a nonlinearity, compress back:

`FFN(x) = max(0, xW₁ + b₁)W₂ + b₂`

768 → 3,072 → 768. The inner size is called `d_ff` and is usually 4× the hidden size. (That's ReLU, from the original Transformer paper. GPT models use GELU. Same idea.)

The FFN is also where most of the parameters are. Per layer it has about twice the weights of attention.

## Where does MoE come in?

MoE swaps that one FFN for a set of FFNs, called experts, plus a small router. For each token the router picks the top K experts (often 2) and only those run.

So you get a lot more parameters without a lot more compute per token. 16 experts with top-2 routing is 16× the FFN parameters, but each token only runs 2 of them.

## So what's the problem?

With only a handful of experts, each one ends up covering a lot of unrelated stuff. One expert might be handling code, French and arithmetic because those tokens have nowhere else to go. That's the opposite of specialization, which was the whole point.

## Fine-grained experts

The paper I was reading, DeepSeekMoE, fixes this by cutting each expert into `m` smaller ones.

1. Each small expert's hidden size drops to `d_ff / m`.
2. You now have `m` times as many experts.
3. The router picks `mK` of them instead of `K`.

Total parameters stay the same. Compute per token stays the same. What changes is how many ways the experts can be combined.

Think about this. 16 experts with top-2 routing gives 120 possible combinations. Split each one into 4 and you have 64 experts with top-8 routing. That's 4,426,165,368 combinations. Same compute, from 120 to 4.4 billion.

Smaller experts can stick to narrower jobs, and the router has far more ways to mix them.

DeepSeekMoE also keeps a few "shared" experts that every token goes through, so the routed ones don't all have to relearn common knowledge. That's for another post.

## What I took away

- Attention moves information between tokens. The FFN works on each token alone.
- MoE turns that one FFN into many and runs only a few per token.
- Fine-grained MoE makes the experts smaller and more numerous. Same compute, far more combinations.

Starting from "does the FFN see all 10 tokens?" was the right call. Most of what confused me about MoE was really confusion about the FFN.
