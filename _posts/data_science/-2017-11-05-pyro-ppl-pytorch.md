---
layout: article
title: Connections betwee supervised, unsupervised, and reinforcement learning
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

- A probabilistic programming language is built on top of the notion of random variables. 
    + So it needs a library of density functions for different distributions. Pyro has such a library at $$pyro.dist$$
    + Pyro's random variables are built using a $$pyro.sample('name', pyro.dist, params)$$
- It needs the ability to integrate deterministic functions with random variables
- It needs an inference engine that would take care of inferece in probabilistic programs. 
    + Pyro can do SVI by defining a $$model$$ (generative) and $$guide$$ (recognition) stochastic functions
    + Inside the generative/recognition models, we need to let Pyro know about all the parameters inside of the decoder/encoder networks. We do this by a call to $$pyro.module()$$.