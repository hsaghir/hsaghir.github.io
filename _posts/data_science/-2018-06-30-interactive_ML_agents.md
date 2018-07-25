---
layout: article
title: ML agents in interaction
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

# Interactive models
Most problems in machine learning are formulated as a single model with an optimization problem over a single objective. However, a number of problems consist of a hybrid of several models, each of which passes information to other models but tries to minimize its own private loss function. For example, 
    - GANs: formulate the unsupervised learning problem as a game between two opponents - a generator G which samples from a distribution, and a discriminator D which classifies the samples as real or false. 
        + To make sure the generator has gradients from which to learn even when the discriminator’s classification accuracy is high, the generator’s loss function is usually formulated as maximizing the probability of classifying a sample as true rather than minimizing its probability of being classified false.
    - Actor Critic RL methods (a single Generator takes an action)
        + While most RL algorithms either focus on learning a value function, or a policy directly, AC methods learn both simultaneously - the actor being the policy and the critic being the value function. 
    - Multiple dialog agents in a conversation (Multiple models take turns in generating an utterance)
    - Even a sequence model with a teacher forcing the right output?
This upsets many of the assumptions behind most learning algorithms, SGD optimization usually results in pathological behavior such as oscillations or collapse onto degenerate solutions. it has been hypothesized that the combination of many different local losses underlies the functioning of the brain as well. 
