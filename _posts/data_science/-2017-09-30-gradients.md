---
layout: article
title: Estimating Gradients of expectations
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

The problem is to calculate the gradient of an expectation of a funtion $$ \nabla_\theta (E_q(z) [f(z)])=\nabla_\theta( \int q(z)f(z))$$ with respect to parameters $$\theta$$. An example is ELBO where gradient is difficult to compute since the expectation integral is unknown or the ELBO is not differentiable.

## Score function gradient: (both continuous and discrete variables)
To calculate the gradient, we first take the $$\nabla_\theta$$ inside the integral to rewrite it as $$\int \nabla_\theta(q(z)) f(z) dz$$ since only the $$q(z)$$ is a function of $$\theta$$. Then we use the log derivative trick (using the derivative of the logarithm $d (log(u))= d(u)/u$) on the (ELBO) and re-write the integral as an expectation $$\nabla_\theta (E_q(z) [f(z)]) = E_q(z) [\nabla_\theta \log q(z) f(z)]$$. This estimator now only needs the dervative $$\nabla \log q_\theta (z)$$ to estimate the gradient. The expectation will be replaced with a Monte Carlo Average. When the function we want derivative of is log likelihood, we call the derivative $\nabla_\theta \log ⁡p(x;\theta)$ a score function. The expected value of the score function is zero.[](http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/)

The form after applying the log-derivative trick is called the score ratio. This gradient is also called REINFORCE gradient or likelihood ratio. We can then obtain noisy unbiased estimation of this gradients with Monte Carlo. To compute the noisy gradient of the ELBO we sample from variational approximate q(z;v), evaluate gradient of log q(z;v), and then evaluate the log p(x, z) and log q(z). Therefore there is no model specific work and and this is called black box inference. 

The problem with this approach is that sampling rare values can lead to large scores and thus high variance for gradient. There are a few methods that use control variates to help with reducing the variance but with a few more non-restricting assumptions we can find a better method with low variance i.e. pathwise gradients. 

## Path-wise Gradients of the ELBO: (continuous variables)
This method has two more assumptions, the first is assuming the hidden random variable can be reparameterized to represent the random variable z as a function of deterministic variational parameters $v$ and a random variable $\epsilon$, $z=f(\epsilon, v)$. The second is that log p(x, z) and log q(z) are differentiable with respect to z. With reparameterization trick, this amounts to a differentiable deterministic variational function. This method generally has a better behaving variance.


## Concrete gradients -  Gumble Softmax trick (discrete variable)
Estimates the discrete categorical variable with a continuous analog (i.e. softmax) and then uses the path-wise gradient to produce a low-variance but biased gradient. 

Replacing every discrete random variable in a model with a Concrete (continuous estimation of discrete) random variable results in a continuous model where the re-parameterization trick is applicable. The gradients are biased with respect to the discrete model, but can be used effectively to optimize large models. The tightness of the relaxation is controlled by a temperature hyper-parameter. In the low temperature limit, the gradient estimates become unbiased, but the variance of the gradient estimator diverges, so the temperature must be tuned to balance bias and variance.

## MuProp gradient

## Straight-through gradient

## Rebar gradient (discrete variable):

Rebar gradient combines Reinforce gradients with gradients of the Concrete variable through a novel control variate for Reinforce.  It produces a low-variance, and unbiased gradient. 


We sought an estimator that is low variance, unbiased, and does not require tuning additional hyper-parameters. To construct such an estimator, we introduce a simple control variate based on the difference between the REINFORCE and the re-parameterization trick gradient estimators for the relaxed model. This reduces variance, but does not outperform state-of-the-art methods on its own. Our key contribution is to show that it is possible to conditionally marginalize the control variate to significantly improve its effectiveness.









