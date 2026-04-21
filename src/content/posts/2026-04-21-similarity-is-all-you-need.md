---
title: "Similarity is (almost) all you need"
description: "From spectral clustering to Gaussian processes to transformer attention — the same primitive, a similarity matrix between points, keeps showing up as the load-bearing piece of very different models."
date: 2026-04-21
tags: ["machine-learning", "unified-views"]
category: "data_science"
---

> **Note (2026):** This started as a 2017 scratchpad called "similarity
> matrix incarnations" — a list of algorithms that, once you squinted,
> were all doing the same thing with a table of pairwise similarities.
> Nine years later the list has a punchline I didn't have at the time:
> the defining architecture of modern deep learning, the transformer, is
> built on a similarity matrix. So here it is, rewritten.

Take a set of points — rows of a dataset, tokens in a sentence, pixels
in an image, users in a product. Build a matrix \(S\) where \(S_{ij}\)
is some measure of how similar point \(i\) is to point \(j\). That's
it. That's the primitive.

An astonishing number of machine-learning methods, once you pull the
mathematics apart, are variations on what to *do* with that matrix.
This post walks through a few of them in chronological order of when
they were the hot new thing, and ends where the story ends up in 2026.

## Step 1: Pick a notion of "similar"

Similarity is whatever you decide it is — it's the modelling prior
dressed up in algebra. Common choices:

- **Euclidean distance** or its squared form — good for geometry in
  \(\mathbb{R}^n\).
- **Cosine similarity** — ignores magnitude, good for directions (text
  embeddings, word vectors).
- **RBF / Gaussian kernel** \(\exp(-\|x_i - x_j\|^2 / 2\sigma^2)\) —
  smooth, local, differentiable; the workhorse of kernel methods.
- **Dot product** — cheap, ubiquitous, and, as we'll see, the one the
  GPU loves.
- **KL divergence / Wasserstein** — similarity between *distributions*
  rather than between points. Asymmetric in the first case; a proper
  distance in the second.

If your measure doesn't land in \([0, 1]\), squash it with a negative
exponential or a softmax and move on. The choice of measure *is* the
inductive bias of the method; everything downstream is linear algebra.

## Spectral clustering — similarity as a graph

Build \(S\). Treat it as the weighted adjacency matrix of a graph,
form the graph Laplacian \(L = D - S\), take its smallest eigenvectors,
and cluster the resulting embedding with k-means.

The point is that once you've committed to "similarity", clustering
reduces to *partitioning a graph*. All the algorithmic machinery of
spectral graph theory becomes available to you for free. Affinity
propagation and normalised cuts are variations on the same theme.

## Gaussian processes — similarity as a covariance

Now treat \(S\) as a *covariance matrix*. If similar points have high
covariance, and you assume everything is jointly Gaussian, you get a
Gaussian process. Prediction at a new point is a conditional Gaussian,
which has a closed form: invert a block, multiply. Regression becomes
linear algebra on \(S\).

This is the cleanest example of "the similarity function is the model".
Pick an RBF kernel and you've chosen smoothness as your prior. Pick a
linear kernel and you've chosen linear regression. Pick a periodic
kernel and you've chosen a model that believes in seasons. The
algorithm doesn't change; the kernel does.

## Kernel methods and SVMs — similarity as a shortcut

The kernel trick makes the same move explicit: instead of mapping
points into some high-dimensional feature space and taking dot
products there, just compute a similarity function directly and
pretend you did. Any learning algorithm that can be written in terms
of inner products — SVMs, PCA, ridge regression — can be "kernelised"
this way. The method stays the same. Only the similarity changes.

## Distributional similarity — GANs, VAEs, and friends

Zoom out one level. Instead of similarity *between points*, measure
similarity *between distributions*. Now the similarity function is a
divergence:

- **KL divergence** — the engine of variational inference; the ELBO is,
  underneath, a KL between an approximate posterior and the true one.
- **Jensen–Shannon** — the original GAN objective.
- **Wasserstein** — the WGAN objective; more stable because it
  degrades gracefully when distributions don't overlap.

The pattern holds: the choice of divergence is the modelling prior,
and almost everything else is optimisation.

## The 2026 punchline: attention is a similarity matrix

Here's the part I couldn't see in 2017. The transformer — the
architecture behind every LLM you've heard of — is, structurally, a
similarity-matrix model.

For each pair of tokens \((i, j)\), attention computes

$$
A_{ij} = \mathrm{softmax}_j\!\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right)
$$

That is: a dot-product similarity between a "query" vector for token
\(i\) and a "key" vector for token \(j\), row-normalised by softmax
into a probability distribution over \(j\). The output for token
\(i\) is then a weighted average of the value vectors \(v_j\), with
the attention weights as mixing coefficients.

Strip away the engineering — multi-head, positional encodings,
residuals, layer norm — and the load-bearing operation inside a
transformer layer is: **build a learned similarity matrix between all
pairs of tokens and use it to re-mix their representations.**

The similarity function is dot product. The "kernel" is learned
end-to-end rather than designed by hand. The graph is fully connected
instead of sparse. Attention weights are normalised per row rather
than symmetric. These are genuine differences. But the primitive at
the core — *pairwise similarity drives the computation* — is the same
primitive that drives spectral clustering, Gaussian processes, kernel
SVMs, and most of classical unsupervised learning.

## Why this framing is useful

Once you see the pattern, new architectures stop feeling like magic:

- **Graph neural networks** — similarity comes from an explicit graph
  instead of being learned; message passing re-mixes neighbour
  representations. Same primitive, different similarity source.
- **Contrastive learning** (SimCLR, CLIP, etc.) — the loss *is* a
  similarity function, and training is an explicit instruction to
  pull similar things together and push dissimilar things apart.
- **Retrieval-augmented models** — a vector store is a similarity
  index; retrieval is a sparse, top-\(k\) approximation of an
  attention layer over a much larger corpus.
- **Diffusion models** — less directly, but the denoising score is a
  learned gradient field that smooths toward regions of high data
  similarity.

This is not a claim that everything is the same. Loss functions,
inductive biases, training regimes, and scaling properties differ in
ways that matter enormously in practice. But when you're trying to
*learn* a new method, the fastest question to ask is:

> Where's the similarity matrix, and who decides what it contains?

If you can answer that, you usually understand the method.
