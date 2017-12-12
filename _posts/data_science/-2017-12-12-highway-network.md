---
layout: article
title: Highway networks
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

# Highway networks
Deep learning is all the rage these days. The idea being that [neural networks]() with many layers can learn representations for the data that makes the desired task (e.g. classification) easier. However, training very deep networks is still a challenging problem. A possible solution is highway networks [1] that provide a way to construct very deep neural networks with hundreds of layers. The core idea is that if we can make the information flow easier through the network, we can increase the depth of the network. An additional insight is that some of the information contained in an input to the network may need to be transformed for constructing better representations while some other part of the information in the input might need to simply be carried to next layer of the network without transformations. If you have ever heard of LSTM [2] networks, this idea will sound familiar. That's because Highway networks are inspired by LSTMs that regulate how much of the input is transformed. They make this adaptive and learnable flow of information possible using a gating mechanism similar to LSTMs. 

A simple feedforward neural network layer consists of a linear transformation of the input, followed by an element-wise nonlinear transformation i.e. $$a = f(W.x+b)$$ (see [this]() for an intuitive explanation). The highway network uses two learnable gates to regulate this flow of information. A "transform gate" learns how much of the input needs to be transformed using the feedforward layer, while a "carry gate" decides how much of the input should be passed without any transformation. The key insight here is that the gates have parameters that are learnable similar to [LSTM gates](), therefore, the optimization procedure can learn how to regulate the information flow in a way that helps the objective. The transform gate is implemented as a simple feedforward neural network layer (i.e. $$T = f(W_T.x+b)$$) whose ouput is multiplied by the original layer output. Usually, (1-T) is used as the carry gate given the intuition that the information that is not transformed needs to be carried (i.e. $$output = a.T + x.(1-T)$$). Therefore, if transform gate learns an identity mapping (i.e. T = 1), all the input is transformed. If the transform gate is closed (T = 0), the input is not transformed at all. 


As an example of use cases for highway networks, is (Bowman et al. 2015) [3], where the authors  mention that they found it necessary to use highway network layers when including feedforward networks between the encoder and the decoder, for the model to learn. Here we investigate what highway networks are and why they are important. We also implement a simple highway network in pytorch.

[1] Srivastava et al., 2015
[2] LSTM
