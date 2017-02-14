---
layout: article
title: what does it mean to learn a probability distribution?
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

- The classical answer to this is to learn a probability density by defining a parametric model (dist family) and finding parameters that maximize the likelihood($ log p(x)$) which would be the same as minimizing the KL divergence between the model and the unknown real data distribution.

- When the support space of model distributions are low dimensional manifolds, they might not intersect with the true distribution's support which means that KL divergence is not defined. To solve this, typical solution in machine learning is to add high band-width Gaussian noise to help model learn all space involved in the true data distribution. This noise will on the other have an avaraging effect and tends to smooth out the learned distribution leading to blurry samples.

- Alternative approach is to define a fixed random variable and pass it through a deterministic transformation function (usuallly a deep net) to map a simple probability density into a the real data distribution. VAEs and GANs use such transformation funtions but VAEs uses transformation function as an approximate likelihood and approximate posterior that are trained by minimizing KL divergence so it still has previous problems. GANs don't. Advantage of transformation models are that 1- sampling is easier since sampling from a high dimensional density is costly 2- don't need to add noise to samples to cover all data distribution space in the learning phase. 3- we can be more flexible in defining cost functions and don't need to use KL-divergence. 

- The key theoretical problem is then finding proper measures of distance or divergence between distributions that we can use in our cost functions. Distance measures impact convergence in the learning process as the transformation function learns to morph the high-dim space into a shape that is similar to the real-data distribution. If the distance between successive transformations and the final distribution tends to zero we will have convergence. A distance that has a weaker topology (more continuous!?) makes it easier for a sequence of disributions to converge. 

- The WGAN paper introduces the Earth Mover (EM) distance and adopts it to learning in GANs which solve many problems of GAN (don't need maintaining careful balance of generator/classifier, doesn't require careful design of net architecture, solves mode collapsing, can continously estimate EM distance by training discriminator to optimality)!