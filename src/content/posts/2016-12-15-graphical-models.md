---
title: "Most probabilistic models are one model in costumes"
description: "PCA, factor analysis, logistic regression, Gaussian mixtures, HMMs, and Kalman filters are the same probabilistic graphical model with different independence assumptions. Seeing this gives you one inference recipe that handles all of them."
date: 2016-12-15
tags: ["machine-learning", "unified-views", "bayesian"]
category: "data_science"
cover: "/images/galton-board.jpg"
coverAlt: "A Galton board: balls bouncing through a grid of pegs accumulate into a bell curve at the bottom. The physical ur-example of a graphical model: local stochastic rules produce a global distribution."
---

Most of what we call "machine learning" is a small vocabulary of probabilistic
models written out in a common language. PCA, factor analysis, logistic
regression, Gaussian mixtures, hidden Markov models, and Kalman filters are
not separate inventions. They are the same object (a joint distribution over
observed and hidden variables) with different independence assumptions.

Once you learn to read the language, two things become easier. You stop
memorizing models and start composing them, and you inherit one inference
recipe, message passing, that works across all of them.

## The question every model answers

A probabilistic model is a joint distribution $p(x, z)$ over observed data
$x$ and hidden variables $z$. Every modeling task reduces to one question:
given the data, what do we believe about the hidden parts? Formally, compute
the posterior $p(z \mid x)$ by Bayes' rule:

$$
p(z \mid x) \;=\; \frac{p(x \mid z)\, p(z)}{p(x)}.
$$

That is the whole program. Choose how $z$ generates $x$ (the likelihood),
choose what you believe about $z$ before seeing data (the prior), and the
math prescribes what to believe after seeing data (the posterior).

What varies across models is not this question. It is the structure of the
joint: which variables depend on which, and how. That structure is a graph.

## The graph factors the joint

Writing the full joint $p(x_1, \dots, x_n)$ over $n$ variables is a table of
size exponential in $n$. What rescues us is conditional independence. In most
sensible models, each variable depends directly on only a few others.
Graphical models make that structure visible: nodes are variables, edges
encode dependence, and the graph dictates how the joint factors into small
local pieces.

Three patterns of three nodes are enough to reason about any graph:

1. **Chain** ($x \to y \to z$). Given $y$, $x$ and $z$ are independent. The
   grandparent stops mattering once you know the parent.
2. **Fork** ($x \leftarrow y \to z$). Given $y$, $x$ and $z$ are independent.
   Siblings are conditionally independent given their common cause.
3. **Collider** ($x \to y \leftarrow z$). Given $y$, $x$ and $z$ may become
   *dependent*, even if they were marginally independent. This is the
   "explaining away" pattern.

These three cases generalize into an algorithm (Bayes-ball) that decides
independence in any graph by a reachability test on shaded observed nodes.
Hammersley and Clifford's theorem then says something stronger: the set of
joints satisfying a given set of independences is exactly the set that
factors according to the graph. Checking independences is the same thing as
factoring the joint.

## One algorithm: message passing

Once you have a factored joint, computing $p(z \mid x)$ means marginalizing
over everything except $z$ and $x$. A naive nested sum is exponential. The
same factoring that gave us the graph lets us reorder the sums so that terms
that do not depend on the current summation variable come outside of it.
This is the elimination algorithm, and it is just dynamic programming.

Doing the same work *locally* by letting each node send a summary to its
neighbours gives message passing. Two rules suffice:

1. At a variable node, multiply the incoming messages and sum over the
   variable's value.
2. At a factor node, multiply by the local factor and marginalize.

For trees, two passes (leaves to root, then root to leaves) produce every
marginal exactly. The forward-backward algorithm for HMMs, the Kalman
smoother, and belief propagation on loopy graphs are all this same recipe
on different graphs. Viterbi decoding is the same algorithm with `max` in
place of `sum`.

Seeing message passing once makes several other algorithms fall into place.
EM is message passing interleaved with a parameter update. Backpropagation
is message passing on a computational graph (there is an
[earlier post](/blog/2017-01-09-incarnations-graphs-dynamic-programming/)
that unpacks this). Variational inference is message passing where you have
replaced intractable messages with tractable approximations.

## What this buys you

Graphical models stopped being the headline in the mid-2010s because the
headline moved to neural networks. The framing did not go away though. It
moved inside the networks.

- A **VAE** is a Bayes net $z \to x$ with a Gaussian prior on $z$, a neural
  likelihood $p_\theta(x \mid z)$, and a variational approximation to
  $p(z \mid x)$. Training maximizes a lower bound (the ELBO) that is a
  message-passing expression in disguise.
- A **diffusion model** is a chain $x_0 \to x_1 \to \cdots \to x_T$ of
  Gaussians, reverse-engineered by learning the backward messages.
- A **graph neural network** is message passing over a graph, where the
  messages are learned rather than derived.
- **Score-based**, **flow-based**, and **energy-based** models all sit
  inside the same frame.

The framing is useful not because it makes these methods "really" PGMs in
some strong sense, but because it tells you what choice you are making at
each design step. What is the graph? What are the independence assumptions?
Which marginals do you want? How are you approximating the intractable
ones? Those are the decisions that matter. The rest is engineering.

## A short reading list

- Pearl, *Probabilistic Reasoning in Intelligent Systems* (1988), the
  founding text and still the best place to learn the intuitions.
- Bishop, [*Model-based machine learning*](https://royalsocietypublishing.org/doi/10.1098/rsta.2012.0222)
  (Phil. Trans. R. Soc. A, 2013), a short and readable summary of the
  "model then inference" paradigm.
- Ghahramani, [*Probabilistic machine learning and artificial intelligence*](https://www.nature.com/articles/nature14541)
  (Nature, 2015), a broader map that includes non-parametrics and
  probabilistic programming.

---

*2026 note: this post was drafted in late 2016, when graphical models were
the standard vocabulary for probabilistic modeling. The vocabulary is still
the right one. What changed is that the likelihood $p(x \mid z)$ is now
routinely a deep network, and the posterior $p(z \mid x)$ is now routinely
approximated by another one. The scaffolding stayed. The bricks got bigger.*

*Cover image: Galton board (Matemateca IME/USP), photo by [Rodrigo Argenton](https://commons.wikimedia.org/wiki/File:Galton_box.jpg), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*
