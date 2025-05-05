---
layout: post
title: "The Life of a Token: From Text to Prediction in GPT"
date: 2025-05-02 12:00:00 -0000
categories: [GPT, Transformers, NLP]
---

The GPT (Generative Pretrained Transformer) model has revolutionized natural language processing (NLP) with its ability to generate human-like text. But how does it actually work? In this blog, we’ll take you through the journey of a token — from raw text to meaningful predictions.

## 1. Text → Tokenization

The journey begins with **tokenization**, where raw input text is broken down into smaller units, often subwords or words. This process allows the model to work with manageable chunks of text. Tokenizers like **Byte Pair Encoding (BPE)** split the text into tokens, which serve as the building blocks for further processing.

## 2. Tokens → Token IDs

Next, each token is mapped to a unique integer ID based on its position in the model's vocabulary. This step ensures that the model can work with a standardized representation of words and phrases.

## 3. Token IDs → Token Embeddings

The token IDs are then transformed into **token embeddings**. These are dense vector representations that capture semantic information about each token. These embeddings allow the model to understand the meaning of each word in the context of the input.

## 4. Token Embeddings + Positional Embeddings = Input Embeddings

Since the GPT model, like most Transformers, does not inherently understand the order of tokens, **positional embeddings** are added to the token embeddings. This step encodes the position of each token in the sequence, ensuring the model understands the relative order of words in the input text.

## 5. Input Embedding → Contextualized Embeddings (via Self-Attention)

The input embeddings are passed through multiple layers of the Transformer, where the **self-attention mechanism** (often multi-head attention) comes into play. This mechanism allows each token to gather context from the other tokens in the sequence. By doing this, the model builds **contextualized embeddings**, where each token’s meaning is influenced by the surrounding words.

## 6. Contextualized Embeddings → Feed-Forward Neural Network (FNN) + Layer Normalization

After self-attention, the embeddings go through **feed-forward neural networks (FNN)**, which refine the embeddings further. These networks are followed by **layer normalization**, a technique that ensures the model remains stable and learns efficiently.

## 7. FNN Output → Residual Connection

To maintain a flow of information across layers, a **residual connection** is added. This means that the output of the FNN is combined with the input of the layer, allowing the model to preserve information from earlier stages.

## 8. Stacking of Layers

The process of self-attention, feed-forward networks, layer normalization, and residual connections is repeated across multiple layers. With each layer, the model builds increasingly complex representations of the input text.

## 9. Final Contextualized Embedding

After passing through all the layers, the final output is a set of **context-aware embeddings** that incorporate information from the entire sequence. These embeddings now contain a rich understanding of the text.

## 10. Contextualized Embedding → Projection to Vocabulary Size

Next, these embeddings are projected into a space that matches the model’s vocabulary size. This projection step is achieved through a linear transformation, and it produces a set of scores or **logits** that represent the likelihood of each word in the vocabulary being the next token.

## 11. Logits → Softmax

The logits are passed through the **softmax function**, which transforms them into probabilities. These probabilities represent the model’s confidence in each possible next token.

## 12. Prediction (Next Token)

Finally, the token with the highest probability is selected as the **next token** in the sequence. This process is repeated iteratively until the model generates the desired length of text or reaches a stopping point (such as an end-of-sequence token).

---
Through this series of transformations, GPT models are able to take raw text, process it in layers, and generate coherent, contextually relevant outputs. Understanding these layers helps us appreciate the complexity and power behind GPT.