---
layout: post
title: "From Self-Attention to Mixture of Experts: A First-Principles Exploration"
date: 2025-05-02
categories: [transformers, MoE, deep-learning]
---

This blog captures my journey as I tried to understand how a Transformer block works — from the self-attention mechanism to the feed-forward network (FNN), and eventually into the world of **Mixture of Experts (MoE)**. Rather than jumping into equations or research jargon, I started with simple questions that naturally arose as I tried to break things down from first principles.

---

## How Does Self-Attention Work?

I began by trying to make sense of self-attention. The key insight is:

> **Self-attention learns which tokens in a sequence should attend to each other.**

Each token outputs a 768-dimensional vector (assuming hidden size = 768). For 10 tokens, the self-attention layer gives us a **10×768 matrix**. This output then goes to the next layer in the Transformer block.

---

## What Happens After Self-Attention?

The next layer is the **Feed-Forward Network (FNN)**, which is applied **independently to each token**.

This initially confused me:

> “Do we feed all 10 tokens into a single FNN?”

No. It turns out that **each of the 10 token embeddings is passed independently through the same FNN**, i.e., the same parameters are applied to each token’s vector in parallel (not 10 different FNNs). This is efficiently implemented in batch using matrix operations.

---

## What Does "Independently" Mean?

"Independently" just means that **the FNN processes each token vector without considering the others**. It does not mean separate FNNs—just the same FNN applied to each token.

---

## So What Does the FNN Actually Do?

A standard Transformer FNN looks like this:

`FNN(x) = max(0, xW₁ + b₁)W₂ + b₂`

- Input: A 768-dim token vector
- Hidden layer: Often 3072-dim (called `d_ff`)
- Output: Back to 768-dim

So it expands, applies nonlinearity, and compresses back.

---

## Where Do Mixture of Experts (MoE) Come In?

While standard Transformers use the same FNN for all tokens, **MoE replaces this shared FNN with a pool of multiple FNNs (called “experts”)**, and each token is routed to a subset of these experts.

However, this raises problems:

- If you have **limited experts**, they may be forced to learn a **hybrid of unrelated knowledge**.
- This **reduces specialization**—the very thing MoE was supposed to improve.

---

## 💡 Fine-Grained Expert Segmentation

To solve this, the paper I studied proposed **splitting each expert into `m` smaller "micro-experts"**, reducing the hidden size of each by `1/m`.

Then, instead of activating 2 full experts (top-2 routing), we activate **m×K micro-experts** to maintain the same total compute.

### Why is this better?

1. **Each micro-expert specializes in a smaller space.**
2. **More combinations**: If you split 16 experts into 4 each → 64 micro-experts. Top-8 routing yields millions of combinations, improving flexibility.

---

## How Does the Math Work?

If each expert’s hidden size is `d_ff`, and we split it into `m` micro-experts, then:

- Each micro-expert gets a hidden size of `d_ff / m`
- Total parameter count stays the same
- Routing selects `mK` micro-experts instead of `K` full experts

This gives you a fine-grained, modular way to distribute knowledge across specialized units.

---

## Summary

Here's what I learned:

- **Self-attention** routes information across tokens.
- **FNNs** act token-wise, without inter-token communication.
- **MoE** allows model capacity to scale without increasing compute per token.
- **Fine-grained MoE** introduces micro-experts to balance specialization and compute.
- More expert combinations = better coverage and flexibility.

---

Thanks for reading!