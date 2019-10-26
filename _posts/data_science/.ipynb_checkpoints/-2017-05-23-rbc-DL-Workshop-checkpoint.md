---
layout: article
title: Hamiltonian MCMC intuitions
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

- AlphaGo -> DL/RL/Monte Carlo tree-search

# Representation learning
They did a study to assess the effects of data amount, feature and algo on performance. They found:
- features are super important
- in particular parts of objects (in vision tasks)
- Softmax gets expensive when the number of classes is very big (millions). For example in language models. In such cases, one might use alternatives like tree structures. 


# hyper-parameter selection
a lot of hyper-params involved in training a model such as model arachitecture, hyper-params, normalization, optimization, etc. 
methods:
- cross-validation with a validation set
- systematic search (e.g. grid, random )
    + grid search: for example if we have 2 hyper params. We form a 2d grid with reasonable values (from experience, papers, etc) for each hyperparam. This is very parallelizable but the problem is when we have more hyper-params we get an explosion of dimentions
    + random search: like a grid search but we sample from the formed grid. We usually assume hyper-params are independent random variables. 
- sequential search by forming a model on hyperparam space (taking human out of the loop: a review of bayesian optimization)
    + Bayesian optimization: form a Gaussian process on top of hyper-params. tradeoff exploration and exploitation.
    + hyperband: promissing.

- saddle points are more common than extrema in DNN. 
- SGD converges if you aneal learning rate meaning using some kind of scheduling $$\epsilon_k=(1-\alpha)\epsilon_0+\alpha \epsilon_t; \alpha=k/t$$

# other
- Parameter initialization: important to break symmetry in the initialization where not all params are initialized similarly. 
- a recipe for learning: scale the uniform distribution with range based on input and output numbers in each layer (Golorot, Bengio 2010).
- Family of algo with adaptive LR are pretty robust. start with SGD with momentum and then go to variants like RMSprop with momentum and then Adam and Adaprop.

# Unsupervised learning:
- what is the objective?
    + reconstruction error (GAN, VAE)?
    + max likelihood (ML)?
    + disentangling factors of variation (PCA)?
- Evaluation is the key difficulty of UL/generative models. 

## Autoencoders:
- compact:
    + information bottleneck layer. 
- sparse: 
    + add regularizer of $$KL(q|p)$$ where both are bernouli. You get the bernouli parameter from the activation of units for p and specify the parameter, e.g. 20%, and regularize the distance of the two bernoulies
    + sparse coding: only a decoder in an optimization of inputs to form a sparse representation of the data.
- denoising:
    + just add some noise to input and pass through the autoencoder and measure the reconstruction error with uncorrupted data. 
    + both additive (e.g. iid Gaussian) and multiplicative (dropout-like) noise
- contractive:
    + insensititvity in hidden space. Penalizes the jacobian of the hidden layer as a regularizer. if the gradient of the hidden is small, it means that, hiddens are invariant. 
    + basically encourages the tangents to the manifold to be small. 

- the vector of difference between data and reconstructed data forms a vector field pointing toward the manifold which takes the data back onto the manifold. 
    + There is theory showing the potential energy is equivalent to score function relating autoencoder to probablistic methods. 
    + Also related, we can sample from a DAE using this concept by retrieving the probability distribution from the potential energy
 

## Graphical models 
- RBM is trained with SGD
    + the gradient is estimated using a positive/negative phase for each batch
    + for each batch 
- Three methods for debugging RBM:
    + monitor reconstruction error between data and it's reconstruction 
    + visualize connection comming into hidden units
    + approximate partition function and see whether negative log likelihoods decrease
- pre-training acts as a regularizer lessens overfitting in a model with high capacity while hurting the learning in a model with low capacity

- For classification purposes, a study found that for both RBM and VAE, the performance is best with same width in layers compared to growing/shrikning width with same number of params
- doing pretraining for a deep autoencoder (unsupervised) helps!

### new wave of graphical models in DL (VAE/GAN/..)
- based on directed graphical model
- inject noise only at the top

- The first model was VAE where the inference is possible using SVI due to re-parameterization trick. The second approach is the one of GANs where we have a classification scheme for density estimation using a two sampe test. 

- GAN issues:
    + instability
    + mode-dropping
    + no universal evaluation metric


