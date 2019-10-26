---
layout: article
title: Deep belief nets, deep Gaussian processes
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---
DBNs are graphical models which learn to extract a deep hierarchical representation of the training data. A DBN is a generative model that mixes directed (hidden layers which form a bayesian net) and undirected (top two layers which form an RBM) connections between variables. The top two layers are an undirected model i.e. an RBM, while the rest of the directed hidden layers are a special bayesian network called sigmoid belief network. In a sigmoid belief network, the probability of a unit to be one, is equal to sigmoid of a linear transformation of the layer above. This effectively solves the explaining away problem. To generate data from a DBN, we would need to do Gibbs sampling from top two layers (RBM) until we converge to an equilibrium state and we can sample from the posterior. Then a sample can be simply propagated through the sigmoid belief net using sigmoids of linear transformation in each layer (i.e. a stochastic proces). DBNs factorize the joint distribution as the conditional probability of observed vector x and the first hidden layer in the sigmoid belief net times the conditional probability of every layer and its top layer in the sigmoid belief net times the joint probability of the top two layers (RBM). The conditional probabilities for each unit in the sigmoid belief net can be calculated as the sigmoid of a linear transformation of its top layer. The joint probability of the RBM can be written in terms of its free energy which is  

Initializing DBNs was also a hard problem, so Hinton et al thought of greedy layer wise initialization to solve initialization for DBNs. They started with a one layer DBN which is an RBM, trained it, used the training weights for initializing the first layer of the sigmoid belief net. Then they used the initializations in the first layer and trained another RBM for the second layer and so on. So they ended up with an RBM at the top and sigmoid layers following. This led to the discovery that the same procedure can be used for pre-training of feedforward nets. The key intuition came from understanding that the probability of the data under an RBM is p(x)=p(x|h)p(h) but the parameters in both p(x|h) and p(h) are the same (i.e. free energy). So let's make new parameters for p(h)  to make model more flexible by adding a new RBM on top of layer h. Continuing this procedure will lead to stacking RBMs on top of each other! Another important intuition here is that we can write the ELBO variational bound for p(x) in an RBM where we marginalize the joint p(x,h), introduce variational distribution q, and use Jensen's inequality for the concave function log p(x) to get the ELBO. Using these two intuitions, we can see that the parameters of each layer are seperate from the next and come up with the greedy layerwise pretraining. Additionally, We can show that this procedure improves the ELBO!

Note that Deep Belief Networks (DBNs) are different from a composition of simple, unsupervised networks such as restricted Boltzmann machines (RBMs) or autoencoders althoght this follows from DBNs. That is called unsupervised pre-training for a feed forward network which was discovered in solving initialization for a DBN. Following is the seminal paper of Hinton et al (2006) on DBNs which got the whole deep learning started.

Learning is difficult in densely-connected, directed belief nets (directed graphical models) that have many hidden layers because it is difficult to infer the conditional distribution of the hidden activities when given a data vector. Variational methods use simple approximations to the true conditional distribution, but the approximations may be poor, especially at the deepest hidden layer where the prior assumes independence. Also, variational learning still requires all of the parameters to be learned together and makes the learning time scale poorly as the number of parameters increases.

The authors describe a model in which the top two hidden layers form an undirected associative memory, and the remaining hidden layers form a directed acyclic graph that converts the representations in the associative memory into observable variables such as the pixels of an image. The paper introduces the idea of a “complementary” prior which exactly cancels the “explaining away” phenomenon (z1->x<-z2) that makes inference difficult in directed models. It shows equivalence between restricted Boltzmann machines and infinite directed networks with tied weights. Another contribution of the paper is introducing a fast, greedy learning algorithm for constructing multi-layer directed networks one layer at a time. Using a variational bound it shows that as each new layer is added, the overall generative model improves


This hybrid model has some attractive features:

0. Solving explaining away problem: In densely connected networks, the posterior distribution over the hidden variables is intractable except in a few special cases such as mixture models or linear models with additive Gaussian noise. MCMC can be used to sample from the posterior, but they are typically very time consuming. Variational methods approximate the true posterior with a more tractable distribution and they can be used to improve a lower bound on the log probability of the training data. But it would be much better to find a way of eliminating explaining away altogether, even in models whose hidden variables have highly correlated effects on the visible variables. It is widely assumed that this is impossible. 

1. There is a fast, greedy learning algorithm that can find a fairly good set of parameters quickly, even in deep networks with millions of parameters and many hidden layers.

2. The learning algorithm is unsupervised but can be applied to labeled data by learning a model that generates both the label and the data.

3. There is a fine-tuning algorithm that learns an excellent generative model.

4. The generative model makes it easy to interpret the distributed representations in the deep hidden layers.

5. The inference required for forming a percept is both fast and accurate.

6. The learning algorithm is local: adjustments to a synapse strength depend only on the states of the presynaptic and post-synaptic neuron.

7. The communication is simple: neurons only need to communicate their stochastic binary states.


References:

Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. "A fast learning algorithm for deep belief nets." Neural computation 18.7 (2006): 1527-1554.



----
