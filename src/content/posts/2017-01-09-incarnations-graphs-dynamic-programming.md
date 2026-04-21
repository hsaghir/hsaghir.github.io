---
title: "The many incarnations of computational graphs, linearization, and dynamic programming"
description: "Backpropagation, belief propagation, the Viterbi algorithm, and matrix-chain multiplication all solve the same problem: summing over exponentially many paths in a graph by reusing work."
date: 2017-01-09
tags: [machine-learning, intuitions, autodiff]
category: data_science
---

Backpropagation is usually taught as "the chain rule, applied carefully".
That is true, but it undersells it. Backpropagation is one instance of a
very general trick: whenever you need to combine values along every path
through a directed graph, you can factor the work so each edge is touched
exactly once. The same trick shows up under at least four different names
in machine learning, and noticing the pattern is one of the best returns on
effort I know of.

## The problem

You have a directed acyclic graph. The edges carry local information —
local derivatives, local probabilities, local costs. You want a global
quantity that is defined as a sum (or product) over all paths from some
source to some sink.

Done naively this is exponential. A graph with three layers and three
nodes per layer already has $3 \times 3 = 9$ paths between the outer
nodes; deep networks have astronomically many. But notice what happens if
you push the sum through the factorisation:

$$
\sum_{\text{paths } p} \prod_{e \in p} w_e
= \sum_{y} \left( \sum_{\text{paths to } y} \cdots \right) \cdot w_{y \to z}
$$

Once you have computed the bracketed quantity at node $y$, you can reuse it
for every path that passes through $y$. The whole computation collapses
from exponential in the depth to linear in the number of edges. This is
exactly **dynamic programming**: solve each subproblem once, cache the
answer.

Once you see it, you start seeing it everywhere.

## Four incarnations of the same algorithm

**Backpropagation.** The graph is a neural network. The local information
on each edge is the Jacobian of one operation with respect to its input.
The "sum over paths" is the total derivative of the loss with respect to a
parameter. Backprop walks the graph once from output to input, at each
node multiplying the incoming gradient by the local Jacobian. Every edge
is touched once.

**Belief propagation (sum-product).** The graph is a probabilistic
graphical model. The local information is a conditional probability or a
potential function. The "sum over paths" is the marginal probability of a
query variable, which by the chain rule of probability is a sum over the
product of local conditionals along every consistent assignment. Message
passing sends one message per edge.

**The Viterbi algorithm (max-product).** Same graph as belief propagation,
but we want the most likely path, not the marginal. Replace the sum with a
max and you get Viterbi. The factorisation is identical. One pass,
forward; one pass, back to recover the argmax.

**Matrix-chain multiplication.** Forget probabilities for a moment. If you
compose functions $f(g(h(x)))$ whose derivatives are Jacobians $F, G, H$,
the total derivative is the matrix product $F \cdot G \cdot H$. Matrix
multiplication is associative but not commutative in cost — the order in
which you multiply changes the flop count. If the input is a single
scalar and the output is a vector ($H$ is tall and thin), you want
$F \cdot (G \cdot H)$, which is **forward-mode AD**. If the input is a
vector and the output is a scalar ($F$ is short and wide), you want
$(F \cdot G) \cdot H$, which is **reverse-mode AD**, a.k.a.
backpropagation. For intermediate shapes, neither is optimal and you get a
classical optimisation problem — the same matrix-chain problem you meet in
an undergraduate algorithms class.

That last one is the punchline. **Backprop is what matrix-chain
multiplication tells you to do when you have many inputs and one output.**

## Why this matters

The taxonomy matters for two reasons.

First, it tells you when each variant is the right tool. Training a neural
net with a million parameters and a scalar loss is the canonical
"many-to-one" case, so reverse-mode wins by a factor of a million.
Sensitivity analysis of a one-dimensional input that feeds into a giant
downstream system is the mirror image, and forward-mode wins. A Jacobian
for a vector-valued intrinsic (a robot's end-effector pose) sits somewhere
in the middle, and the question of which order to multiply in is a real
engineering decision.

Second, it tells you how to invent algorithms. "Sum-product belief
propagation but with max in place of sum" is how Viterbi was discovered.
"Backprop but with the Hessian instead of the Jacobian" is Hessian-free
optimisation. "Backprop but the local operations are stochastic samples"
is the reparameterisation trick. The skeleton is the same; the local
information changes.

## A tiny worked example

Take the expression $e = (a + b)(b + 1)$. Introduce $c = a + b$ and
$d = b + 1$ so every operation has a name. The computational graph is

$$
a, b \longrightarrow c = a+b, \quad b \longrightarrow d = b+1, \quad c, d \longrightarrow e = c \cdot d
$$

We want $\partial e / \partial b$. Notice that $b$ affects $e$ through
**two** paths — once via $c$, once via $d$. Naively:

$$
\frac{\partial e}{\partial b}
= \underbrace{\frac{\partial e}{\partial c} \frac{\partial c}{\partial b}}_{\text{path via } c}
+ \underbrace{\frac{\partial e}{\partial d} \frac{\partial d}{\partial b}}_{\text{path via } d}
$$

Both terms reuse $\partial e / \partial c$ and $\partial e / \partial d$.
Compute those once, at the top, and push them back. That is literally the
backward pass. Now replace "partial derivative" with "conditional
probability" and you have sum-product belief propagation on the same
graph.

## The chain rule is a design pattern

It is worth taking stock of how many different objects obey a "chain rule":

$$
\begin{aligned}
\frac{dy}{dx} &= \frac{dy}{du} \cdot \frac{du}{dx} && \text{(calculus)} \\
P(X_1, \ldots, X_n) &= \prod_i P(X_i \mid X_{i+1}, \ldots, X_n) && \text{(probability)} \\
H(X, Y) &= H(X) + H(Y \mid X) && \text{(entropy)} \\
K(X, Y) &= K(X) + K(Y \mid X) + O(\log K(X, Y)) && \text{(Kolmogorov complexity)}
\end{aligned}
$$

Every one of these factorises "a global quantity about a composite object"
into "a local quantity on each piece". Every one of them can be computed
on a graph by message passing. Whenever you see a chain rule, the graph
and the dynamic-programming algorithm are hiding just behind it.

## The takeaway

There is one algorithm. It has a forward pass that propagates local
information from source to sink, and a backward pass that propagates the
accumulated global information back. The "local information" is whatever
object has a chain rule — derivatives, probabilities, costs, entropies.
The "accumulation" is whatever operation is distributive over the chain
rule — sum, max, multiplication.

Backpropagation, belief propagation, the Viterbi algorithm, and the
Bellman–Ford dynamic program are not four things to memorise. They are
one thing, parameterised four ways.

---

*This is an old note from 2017, lightly polished. I am publishing it now
because the pattern has only gotten more useful — transformers, diffusion
models, and modern probabilistic programming frameworks all live or die by
it.*
