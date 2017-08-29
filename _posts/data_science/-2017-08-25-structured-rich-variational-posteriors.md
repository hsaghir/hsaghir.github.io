---
layout: article
title: Rich and structured variational posteriors
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

In mean-field variational family, the latent variables are mutually independent and each latent variable $$z_j$$ is governed by its own variational factor, the density $$q_j(z_j)$$ in the total variational density $$q(z) = \prod_j q_j(z_j)$$. One way to expand the family is to add dependencies between the variables. this is called structured variational inference. Another way to expand the family is to consider mixtures of variational densities, i.e., additional latent variables within the variational family.

## more expressive distributions
- Dustin Tran and Rajesh Ranganath have put out an awesome series of papers that allow for more expressive distributions, based on marginalizing out a latent variable, to be used as variational approximations
    + [Hierarchical Variational Models](https://arxiv.org/abs/1511.02386): HVMs augment a variational approximation with a prior on its parameters, which allows it to capture complex structure for both discrete and continuous latent variables.
    + [Variational Gaussian Process](https://arxiv.org/abs/1511.06499):  a Bayesian nonparametric variational family, which adapts its shape to match complex posterior distributions.


## deterministic transformations of the posterior
- Danilo Rezende and Shakir Mohamed introduced the normalizing flows framework [3], which constructs an expressive approximate distribution by composing invertible maps.

    + [Variational Inference with Normalizing Flows](http://proceedings.mlr.press/v37/rezende15.pdf) 
        + blog post [casmls](https://casmls.github.io/general/2016/09/25/normalizing-flows.html)

    + [Improved Variational Inference with Inverse Autoregressive Flow](https://arxiv.org/pdf/1606.04934.pdf)
        1. Start with fully-factorized Gaussian approximate posterior and use the VAE reparameterization trick to get y = mu + epsilon*sigma. Mu and sigma are inferred through the encoder network.
        2. Let the encoder approximate L(x) and use that to update y.
        3. "One step of linear IAF then turns the fully-factorized distribution of y into an arbitrary multivariate Gaussian distribution: z = L(x)·y". apply a step of IAF, and you can turn your fully-factored Gaussian into a conditional multivariate Gaussian which you then fit by optimizing the variational lower bound

        you're not really changing the KL term on the latents from the VAE (so you're still optimizing mu and sigma to make y look like a standard normal), but by throwing the "lower triangular inverse Cholesky matrix" into the mix, we add an additional set of parameters L(x) that allow us to transform the (posterior?) distribution over the latents into a multivariate Gaussian without actually having to directly optimize or specify the multivariate Gaussian. 
    + [Density estimation using Real NVP](https://arxiv.org/pdf/1605.08803v1.pdf)


    + [Made: masked autoencoder for distribution estimation]()




## Adding additional latent variables.

expand the family is to consider mixtures of variational densities, i.e., additional latent variables within the variational family.
    + [Auxiliary Deep Generative Models](https://arxiv.org/abs/1602.05473)

    + [DEEP UNSUPERVISED CLUSTERING WITH GAUSSIAN MIXTURE VARIATIONAL AUTOENCODERS](https://arxiv.org/pdf/1611.02648.pdf)
        - blog post:  http://ruishu.io/2016/12/25/gmvae/


## Gradually improve variational approximation

In our approach, we first specify a simple class of approximations — e.g. Gaussians with diagonal covariance — and then optimize the corresponding variational objective, resulting in the best reachable approximation (in the KL-sense) that a diagonal Gaussian can do. We then expand the variational family in one of two ways: add structure to the covariance, add a component and form a mixture.
    + [IMPROVING VARIATIONAL APPROXIMATIONS](http://andymiller.github.io/2016/11/23/vb.html)


## Implicit posteriors (GAN)

Implicit distributions (transformation models) like GANs can help solve the restricted mean field assumption.
    + [ALI - BiGAN](https://arxiv.org/abs/1606.00704, https://arxiv.org/abs/1701.04722)

    + [Deep and Hierarchical Implicit Models](https://arxiv.org/abs/1702.08896)
        blog post: [](http://dustintran.com/blog/deep-and-hierarchical-implicit-models)

        As a practical example, we show how you can take any standard neural network and turn it into a deep implicit model: simply inject noise into the hidden layers. The hidden units in these layers are now interpreted as latent variables. Further, the induced latent variables are astonishingly flexible, going beyond Gaussians (or exponential families (Ranganath, Tang, Charlin, & Blei, 2015)) to arbitrary probability distributions. Deep generative modeling could not be any simpler!


## Structured state space models
    + [Black box variational inference for state space models](https://arxiv.org/abs/1511.07367)





## Other

Other methods for dealing with restricted mean field posterior.
    + [Integrated non-factorized variational inference](https://papers.nips.cc/paper/5068-integrated-non-factorized-variational-inference.pdf)
        - MCMC can provide samples from exact posterior but is time-consuming and convergence assessment is difficult. Deterministic alternatives include the Laplace approximation, variational methods, and expectation propagation (EP).

        - Inspired by INLA, we propose a hybrid continuous-discrete variational approximation, which enables us to preserve full posterior dependencies and is therefore more accurate than the mean-field variational Bayes (VB) method.

        - To make the variational problem tractable, the variational distribution q(x, θ|y) is commonly required to take a restricted form. For example, mean-field variational Bayes (VB) method assumes the distribution factorizes into a product of marginals, q(x, θ|y) = q(x)q(θ), which ignores the posterior dependencies among different latent variables (including hyperparameters) and therefore impairs the accuracy of the approximate posterior distribution.

        - We propose a hybrid continuous-discrete variational distribution $$q(x|y, θ)q_d(θ|y)$$, where $$q_d(θ|y)$$ is a finite mixture of Dirac-delta distributions. Clearly, $$q_d(θ|y)$$ is an approximation of $$q(θ|y)$$ by discretizing the continuous (typically) low-dimensional parameter space of θ using a grid G with finite grid points.  The use of $$q_d(θ)$$ is equivalent to numerical integration, here used to overcome factorized posteriors in variational inference.

    + [Copula variational inference](https://arxiv.org/pdf/1506.03159.pdf)

        - copula vi is a new variational inference algorithm that augments the mean-field variational distribution with a copula; it captures posterior dependencies among the latent variables. We derived a scalable and generic algorithm for performing inference with this expressive variational distribution.
















