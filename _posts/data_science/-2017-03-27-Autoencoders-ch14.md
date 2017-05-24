---
layout: article
title: Autoencoders Ch14
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---


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