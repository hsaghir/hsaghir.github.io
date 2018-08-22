---
layout: article
title: ML agents in interaction
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---


# Unifying all ML as interaction among agents
Most problems in machine learning have been formulated as a single model with an optimization problem over a single objective. However, all ML problems can be formulated as interactions between multiple agents. In the simplest case, a single model learning from data, the interaction is between a learning agent (model) and a non-learning agent (data sampler). 

## Multiple Learning agents
A number of problems consist of a hybrid of several learning agents (models), each of which passes information to other models but tries to minimize its own private loss function. This upsets many of the assumptions behind most learning algorithms, SGD optimization usually results in pathological behavior such as oscillations or collapse onto degenerate solutions. it has been hypothesized that the combination of many different local losses underlies the functioning of the brain as well. 

- [The Mechanics of n-Player Differentiable Games](https://arxiv.org/pdf/1802.05642.pdf) This paper argues that in interactive ML, the loss function consists of competing terms that constitute games. It analyzes the possible games into three categories based on the Hessaing of multiple terms of the loss function w.r.t. their respective variables. If we re-write the Hessian in terms of the addition of a symmetric ((H+H')/ 2) and an anti-symmetric function ((H-H')/2), the games are categorized into three classes. 
	+ first class is potential games where the anti-symmetric term of the Hessian is zero. The constituting terms of the loss in this case, form gradients in the same direction for example, a single objective classification problem. In such scenarios SGD works well since the direction of the first order gradient of the loss constitutes a gradient field and we can follow it to get to a local minimum. 
	+ second class of games where the symmetric term is zero, are what the paper calls Hamiltonian games. Hamiltonian games are similar to energy conserving physical systems that constitute a limit cycle in the gradient field. The direction of the first order gradient is tangent to this limit cycle, therefore, we can't really reduce the loss. for this class of games, the paper suggests Synthetic Gradient Averaging (SGA), that is a transformation on the gradient to map it to the direction perpendicular to the limit cycle. This gradient has similarities to second order and natural gradient methods that map the gradient from a euclidean space to a hamiltonian space. The paper suggests to move in the direction of $$\epsilon + \lambda A^T \epsilon$$, where $$\epsilon$$ is SGD gradient, $$A^T$$ is the anti-symmetric part of the Hessian matrix. 
	+ the third class of games are general games, where we don't have only a potential game or a Hamiltonian game but a mixture of both. In these situations, my physical intuition is that physical system containing all interacting models will be a non-energy conserving system dissipating or adding energy. Therefore, we won't have limit-cycle-like balances in the total energy gradient field landscape and one of the interacting systems may dominate the others. GANs are usually systems of this kind and we usually see one of the involved systems dominate the other, therefore, achieving a balanced minimum is usually hard.  



- Following are example where optimization has been notoriously hard, 
    - GANs: formulate the unsupervised learning problem as a game between two opponents - a generator G which samples from a distribution, and a discriminator D which classifies the samples as real or false. 
        + To make sure the generator has gradients from which to learn even when the discriminator’s classification accuracy is high, the generator’s loss function is usually formulated as maximizing the probability of classifying a sample as true rather than minimizing its probability of being classified false.
    - Actor Critic RL methods (a single Generator takes an action)
        + While most RL algorithms either focus on learning a value function, or a policy directly, AC methods learn both simultaneously - the actor being the policy and the critic being the value function. 
    - Multiple dialog agents in a conversation (Multiple models take turns in generating an utterance)
    - Even a sequence model with a teacher forcing the right output?