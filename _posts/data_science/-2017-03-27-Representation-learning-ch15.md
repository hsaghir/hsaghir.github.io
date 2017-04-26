---
layout: article
title: Representation Learning Ch15
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

- Many information processing tasks can be very easy or very difficult depending on how the information is represented. A good representation is one that makes a subsequent learning task easier. Representation learning provides a way to perform unsupervised and semi-supervised learning.

- Learning algorithms share statistical strength across different tasks, including using information from unsupervised tasks to perform supervised tasks. Multiple tasks (some supervised, some unsupervised) can be learned together with some shared internal representation. Most representation learning problems face a tradeoff between preserving asmuch information about the input as possible and attaining nice properties (such as independence).

- To do semi-supervised learning, an elegant approach is to train an autoencoder or generative model at the same time as the supervised model. Examples include the discriminative RBM (Larochelle and Bengio, 2008) and the ladder network (Rasmus et al., 2015), in which the total objective is an explicit sum of the two terms (one using the labels and one only using the input). An alternative simple approach is greedy layer-wise pretraining. 

- Greedy layer-wise unsupervised pretraining relies on a single-layer representation learning algorithm such as an RBM, a single-layer autoencoder, a sparsecoding model, or another model that learns latent representations. Each layer is pretrained using unsupervised learning, taking the output of the previous layer and producing as output a new representation of the data, whose distribution (or its relation to other variables such as categories to predict) is hopefully simpler. It is a greedy algorithm meaning that it optimizes each piece of the solution independently, one piece at a time, rather than jointly optimizing all pieces.

- This pretraining procedure may be used as an initialization for another unsupervised or supervised task. The supervised learning phase may involve training a simple classifier on top of the features learned in the pretraining phase or joint optimization of all layers for fine tuning. Pretraining can also be viewed as a regularizer in the sense that the choice of initial parameters can possibly initialize the system in a location that would be otherwise inaccessible (in some experiments, pretraining decreases test error without decreasing training error). 

- Unsupervised pretraining can help or hurt learning depending on the task. In general, when unsupervised pretraining leads to a better  distance measure in the new representation, it's useful. For example, pretrained word-vectors are much better at distance between similar words than one-hot vectors. 

- Neural networks that receive unsupervised pretraining consistently halt in the same region of function space, while neural networks without pretraining consistently halt in another region