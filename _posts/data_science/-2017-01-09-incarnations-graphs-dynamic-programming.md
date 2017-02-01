---
layout: article
title: The many incarnations of computational graphs, linearization, and dynamic programming
comments: true
image:
  teaser: jupyter-main-logo.svg
---

Computational graphs are a nice way to think about mathematical expressions. For example, consider the expression e=(a+b)∗(b+1). There are three operations: two additions and one multiplication. To help us talk about this, let’s introduce two intermediary variables, c and dd so that every function’s output has a variable. We now have:

c=a+b

d=b+1

e=c∗d

To create a computational graph, we make each of these operations, along with the input variables, into nodes. When one node’s value is the input to another node, an arrow goes from one to another.

These sorts of graphs come up all the time in computer science. We can evaluate the expression by setting the input variables to certain values and computing nodes up through the graph. If there is some sort of chain rule involved in a computations of interest, then it can be solved for all nodes on the graph using some variation of a forward and backward pass. If one wants to understand computation of interest (i.e. derivative, marginal probability, etc) in a computational graph, the key is to understand it on the edges. 

Depending on the way a graph looks like, doing a forward/backward computation of interest may be more appropriate since it’s very easy to get a combinatorial explosion in the number of possible paths in the graph the other way around. For example, suppose three nodes X,Y,Z, there are three paths from X to Y, and a further three paths from Y to Z. If we want to get the derivative of z wrt x by summing over all paths, we need to sum over 3∗3=9 paths which grows exponentially. Instead of just naively summing over the paths, it would be much better to factor them. 

This is where “forward-mode differentiation” and “reverse-mode differentiation” come in. They’re algorithms for efficiently computing the sum by factoring the paths. Instead of summing over all of the paths explicitly, they compute the same sum more efficiently by merging paths back together at every node. In fact, both algorithms touch each edge exactly once! Forward-mode differentiation starts at an input to the graph and moves towards the end. At every node, it sums all the paths feeding in. Each of those paths represents one way in which the input affects that node. By adding them up, we get the total way in which the node is affected by the input, it’s derivative. Though you probably didn’t think of it in terms of graphs, forward-mode differentiation is very similar to what you implicitly learned to do if you took an introduction to calculus class. Reverse-mode differentiation, on the other hand, starts at an output of the graph and moves towards the beginning. At each node, it merges all paths which originated at that node. 

Forward-mode differentiation gave us the derivative of our output with respect to a single input, but reverse-mode differentiation gives us all of them. For this graph, that’s only a factor of two speed up, but imagine a function with a million inputs and one output. Forward-mode differentiation would require us to go through the graph a million times to get the derivatives. Reverse-mode differentiation can get them all in one fell swoop! A speed up of a factor of a million is pretty nice!

First, you could derive explicit expressions for all the individual partial derivatives in Equation. That's easy to do with a bit of calculus. Having done that, you could then try to figure out how to write all the sums over indices as matrix multiplications. This turns out to be tedious, and requires some persistence, but not extraordinary insight. After doing all this, and then simplifying as much as possible, what you discover is that you end up with exactly the backpropagation algorithm! And so you can think of the backpropagation algorithm as providing a way of computing the sum over the rate factor for all these paths. Or, to put it slightly differently, the backpropagation algorithm is a clever way of keeping track of small perturbations to the weights (and biases) as they propagate through the network, reach the output, and then affect the cost.


Are there any cases where forward-mode differentiation makes more sense? Yes, there are! Where the reverse-mode gives the derivatives of one output with respect to all inputs, the forward-mode gives us the derivatives of all outputs with respect to one input. If one has a function with lots of outputs, forward-mode differentiation can be much, much, much faster.

There's another cool algebraic view: for f(g(h(...))) the derivative is F*G*H where * is matmul and F,G,H are Jacobian matrices. If you have many inputs and one output, f is R^n->R^1, then your last matrix is skinny and tall, then Matrix Chain Multiplication solution tells you to do (F G)H, which is reverse mode AD. But if you have many outputs and one input, your H is wide and short, so most efficient is to do F(G H) which is forward mode AD. But also there are cases where neither forward nor reverse mode AD are the most efficient, and those are the "other" solutions of the MCM problem




The chain rule is a rule for some calculation on a compositions of functions.Chain rule may refer to:


Chain rule in calculus (for calculating differentiations):
\frac {\mathrm dy}{\mathrm dx} = \frac {\mathrm dy} {\mathrm du} \cdot\frac {\mathrm du}{\mathrm dx}.

Cyclic chain rule, or triple product rule (aslo known as cyclic relation, cyclical rule or Euler's chain rule, is a formula which relates partial derivatives of three interdependent variables. ):
\left({\frac  {\partial x}{\partial y}}\right)_{z}\left({\frac  {\partial y}{\partial z}}\right)_{x}\left({\frac  {\partial z}{\partial x}}\right)_{y}=-1.

Chain rule- probability (calculation of any member of the joint distribution of a set of random variables using only conditional probabilities):
\mathrm  P(X_1=x_1, \ldots, X_n=x_n) = \prod_{i=1}^n  \mathrm P(X_i=x_i \mid X_{i+1}=x_{i+1}, \ldots, X_n=x_n )

Chain rule for information entropy (similar form to Chain rule in probability theory, except that addition instead of multiplication is used.):
H(X,Y) = H(X) + H(Y|X)

Chain rule for Kolmogorov complexity (an analogue of the chain rule for information entropy):
K(X,Y) = K(X) + K(Y|X) + O(\log(K(X,Y)))

Instances:
- a graph of neurons (Neural net): Backpropagation calculates the gradient for every node using a forward pass and using the chain rule in the backward pass. 
- a graph of probability dist (PGM): Message passing calculates marginal probability for every node using a forward pass of probabilities and using the chain rule in the backward pass. 

- A* !?