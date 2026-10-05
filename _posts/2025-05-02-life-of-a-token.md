---
layout: post
title: "The Life of a Token in GPT"
date: 2025-05-02 12:00:00 -0000
categories: [GPT, Transformers, NLP]
---

Type "Hello world" into GPT-2 and it gives you back one token. Just one. Everything else you see is that same step running in a loop.

I kept reading about attention heads and MoE layers without having the full path in my head, so I wrote it down end to end. GPT-2 small is the example throughout because the numbers are small enough to hold on to: a 50,257-token vocabulary, 768-dimensional embeddings, 12 layers and a 1,024-token context.

## 1. Text to tokens

The model never sees characters. A tokenizer (Byte Pair Encoding, for GPT-2) chops the text into subword pieces. "Hello world" becomes two tokens: `Hello` and ` world`. The space belongs to the second token.

Common words get one token. Rare words get split into pieces.

## 2. Tokens to IDs

Each token is looked up in the vocabulary and swapped for an integer. `Hello` is 15496, ` world` is 995. That's all a token ID is. A row number.

## 3. IDs to embeddings

That row number indexes into an embedding matrix of 50,257 × 768. Row 15496 is the 768-number vector for `Hello`.

Think about this. That one matrix is 38.6 million parameters, close to a third of GPT-2 small.

## 4. Add the position

Attention by itself doesn't know word order. "Dog bites man" and "man bites dog" have the same tokens. So GPT-2 learns a second vector for every position (a 1,024 × 768 matrix) and adds it to the token embedding. That sum is what goes into the first layer.

## 5. Inside a layer

Each of the 12 layers does two things.

1. Attention: Every token looks at the tokens before it (never after, that's the causal mask) and pulls in what's relevant. This is where ` world` picks up that it came after `Hello`. There are 12 heads running in parallel, each free to look for something different.
2. Feed-forward network (FFN): Each token's vector goes through the same small MLP, on its own. 768 → 3,072 → 768. Attention moves information between tokens. The FFN works on what each token has collected.

Both are wrapped with a residual connection (add the input back to the output) and layer normalization. GPT-2 puts the layer norm before each block instead of after, which makes deep stacks easier to train.

A layer gives back the same shape it took in, one 768-dim vector per token. That's why you can stack 12 of them. Or 96, like GPT-3.

## 6. Vectors to logits

After the last layer and one final layer norm, each vector is multiplied by the transpose of the embedding matrix from step 3. GPT-2 reuses that matrix instead of learning a new one (weight tying). Out come 50,257 numbers, one score per vocabulary entry. These are the logits.

For generation, only the logits at the last position matter. That's the model's guess for what comes next.

## 7. Logits to a token

Softmax turns the logits into probabilities. Then you pick one.

Always picking the highest (greedy decoding) is the simplest option, but it gets repetitive. Usually the next token is sampled instead, with temperature, top-k or top-p deciding how adventurous it gets.

## 8. Do it again

Append the new token and run the whole thing again. Stop at an end-of-text token or a length limit.

This is the part I actually care about. Every generated token is a full pass through all 12 layers. Done naively, you'd recompute attention over the whole prefix every single time. The KV cache stores the keys and values from earlier tokens so each step only does the work for the new one. That's also why inference splits into prefill (the whole prompt in one go, compute-bound) and decode (one token at a time, memory-bound). Different post.

Text in, one token out, repeat.
