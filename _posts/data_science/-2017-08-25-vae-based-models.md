---
layout: article
title: The unreasonable elegance of deep generative models
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---



blog post [casmls](https://casmls.github.io/general/2016/09/25/normalizing-flows.html)


## [Improved Variational Inference with Inverse Autoregressive Flow](https://arxiv.org/pdf/1606.04934.pdf)

1. Start with fully-factorized Gaussian approximate posterior and use the VAE reparameterization trick to get y = mu + epsilon*sigma. Mu and sigma are inferred through the encoder network.
2. Let the encoder approximate L(x) and use that to update y.
3. "One step of linear IAF then turns the fully-factorized distribution of y into an arbitrary multivariate Gaussian distribution: z = L(x)·y". apply a step of IAF, and you can turn your fully-factored Gaussian into a conditional multivariate Gaussian which you then fit by optimizing the variational lower bound

you're not really changing the KL term on the latents from the VAE (so you're still optimizing mu and sigma to make y look like a standard normal), but by throwing the "lower triangular inverse Cholesky matrix" into the mix, we add an additional set of parameters L(x) that allow us to transform the (posterior?) distribution over the latents into a multivariate Gaussian without actually having to directly optimize or specify the multivariate Gaussian. 

## [Variational Inference with Normalizing Flows](http://proceedings.mlr.press/v37/rezende15.pdf) 


## [Density estimation using Real NVP](https://arxiv.org/pdf/1605.08803v1.pdf)


## [Made: masked autoencoder for distribution estimation]()


## [Auxiliary Deep Generative Models](https://arxiv.org/abs/1602.05473)


## [DEEP UNSUPERVISED CLUSTERING WITH GAUSSIAN MIXTURE VARIATIONAL AUTOENCODERS](https://arxiv.org/pdf/1611.02648.pdf)

blog post:  http://ruishu.io/2016/12/25/gmvae/

## [ALI - BiGAN](https://arxiv.org/abs/1606.00704, https://arxiv.org/abs/1701.04722)