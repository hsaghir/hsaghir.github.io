---
layout: article
title: Estimating Gradients of expectations
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

The problem is to calculate the gradient of an expectation of a funtion $$ \nabla_\theta (E_q(z) [f(z)])=\nabla_\theta( \int q(z)f(z))$$ with respect to parameters $$\theta$$. An example is ELBO where gradient is difficult to compute since the expectation integral is unknown or the ELBO is not differentiable.

## Score function gradient (likelihood ratio gradient estimator, REINFORCE gradient): (both continuous and discrete variables)
To calculate the gradient, we first take the $$\nabla_\theta$$ inside the integral to rewrite it as $$\int \nabla_\theta(q(z)) f(z) dz$$ since only the $$q(z)$$ is a function of $$\theta$$. Then we use the log derivative trick (using the derivative of the logarithm $d (log(u))= d(u)/u$) on the (ELBO) and re-write the integral as an expectation $$\nabla_\theta (E_q(z) [f(z)]) = E_q(z) [\nabla_\theta \log q(z) f(z)]$$. This estimator now only needs the dervative $$\nabla \log q_\theta (z)$$ to estimate the gradient. The expectation will be replaced with a Monte Carlo Average. When the function we want derivative of is log likelihood, we call the derivative $\nabla_\theta \log ⁡p(x;\theta)$ a score function. The expected value of the score function is zero.[](http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/)

The form after applying the log-derivative trick is called the score ratio. This gradient is also called REINFORCE gradient or likelihood ratio. We can then obtain noisy unbiased estimation of this gradients with Monte Carlo. To compute the noisy gradient of the ELBO we sample from variational approximate q(z;v), evaluate gradient of log q(z;v), and then evaluate the log p(x, z) and log q(z). Therefore there is no model specific work and and this is called black box inference. 

The problem with this approach is that sampling rare values can lead to large scores and thus high variance for gradient. There are a few methods that use control variates to help with reducing the variance but with a few more non-restricting assumptions we can find a better method with low variance i.e. pathwise gradients. 


## Random Directions Gradient Estimator (evolution strategies):
In REINFORCE, We start from a random param set and reapetedly 1) let the network play (let MCMC run) 2) estimate the gradient of expectation of reward using Monte Carlo 3) backprop it and adjust the params using SGD.

In Evolutionary search (ES) we forget all this, we just want to find the best setting of the 1M params of policy net. we start with some random parameters and then repeatedly 1) tweak the guess a bit randomly, and 2) move our guess slightly towards whatever tweaks worked better. 

Concretely in ES, at each step we take a parameter vector w and generate a population of, say, 100 slightly different parameter vectors w1 ... w100 by jittering w with gaussian noise. We then evaluate each one of the 100 candidates independently by running the corresponding policy network in the environment for a while, and add up all the rewards in each case. The updated parameter vector then becomes the weighted sum of the 100 vectors, where each weight is proportional to the total reward! 

Mathematically, you’ll notice that this is also equivalent to estimating the gradient of the expected reward in the parameter space using finite differences, except we only do it along 100 random directions.

- If the function is not differentiable (with non-zero gradients almost everywhere) then you can't do anything much better than sampling (whether by REINFORCE, ES or something else). If you can differentiate, then you probably can't beat backprop which is why path-wise gradient is proposed.


## Simultaneous Perturbation Stochastic Approximation gradient (Continuous variable)

[url](http://mdenil.com/static/papers/2011-mdenil-quantum_deep_learning-nips_workshop.pdf)

Simultaneous Perturbation Stochastic Approximation (SPSA) is an algorithm for approximate gradient based optimization of noisy, differentiable, black box functions [20]. SPSA requires that the objective be differentiable, but unlike traditional stochastic gradient methods it does not require explicit access to the objective gradient.

SPSA estimates the objective gradient at each step using a stochastic variant of the finite difference approximation. Forming the ordinary finite difference approximation is expensive in high dimensions, since it requires 2d evaluations of the objective (where d is the dimensionality of the problem). In contrast, SPSA is able to function with exactly 2 objective evaluations at each step, regardless of the dimensionality.

### Local re-parameterization trick
- in practice the performance of stochastic gradient descent crucially depends on the variance of the gradients. If this variance is too large, stochastic gradient descent will fail to make much progress in any reasonable amount of time. If we calculate the Variance of the log likelihood part of the ELBO, we get two terms, one that depends on the variances of individual datapoints in a minibatch and one that depends on the covariance of variances of data points in the minibatch.

- The total contribution to the variance by variance of each minibatch ELBO is inversely proportional to the minibatch size M. However, the total contribution by the covariances does not decrease with M. In practice, this means that the variance of ELBO in SGVB estimator can be dominated by the covariances for even moderately large M.

- We therefore propose an alternative estimator for which we have Cov [Li,Lj ]=0, so that the variance of our stochastic gradients scales as 1/M. We then make this new estimator computationally efficient by not sampling epsilon directly, but only sampling the intermediate variables f(epsilon) through which the epsilon influences the SGVB ELBO estimate.  By doing so, the global uncertainty in the weights is translated into a form of local uncertainty that is independent across examples and easier to sample. Whenever a source of global noise can be translated to local noise in the intermediate states of computation (epsilon -> f(epsilon)), a local reparameterization can be applied to yield a computationally and statistically efficient gradient estimator.


## Path-wise Gradients of the ELBO: (continuous variables)
This method has two more assumptions, the first is assuming the hidden random variable can be reparameterized to represent the random variable z as a function of deterministic variational parameters $v$ and a random variable $\epsilon$, $z=f(\epsilon, v)$. The second is that log p(x, z) and log q(z) are differentiable with respect to z. With reparameterization trick, this amounts to a differentiable deterministic variational function. This method generally has a better behaving variance.


## Synthetic gradient 
The key insight in the paper is the gradient $$\frac{\partial L}{\partial \theta_i}$$ doesn’t have to be a monolithic expression, but factorizes nicely via chain rule as $$\frac{\partial L}{\partial h_i}\frac{\partial h_i}{\partial \theta_i}$$, and that each of these factors can be computed separately:

1. The factor $$\frac{\partial h_i}{\partial \theta_i}$$, only depends on information local to the layer i. So, that’s available instantaneously.
2. Leaving the factor $$\frac{\partial L}{\partial h_i}$$ to be approximated with an estimate $$\hat{\delta_i}$$. So, now your “instantaneous” update equation looks like this:
$$\theta_i \leftarrow \theta_i - \alpha~\hat{\delta_i}~\frac{\partial h_i}{\partial \theta_i}$$. 

The estimator $$\hat{\delta_i}$$ can be anything, but the most obvious thing is train another mini neural network $$M_{i+1}$$.  The mini neural network has its own parameters to be trained in a supervised setting with the real loss L. The supervision comes from the fact that the accurate loss LL will eventually get computed and will trickle down to serve as supervision to learn the params of M_{i+1}. This parameter update can happen independently and asynchronously, removing any kind of network locking. As expected, this model can produce noisy gradients, but as we know from literature, the presence of noise is actually desirable to make the training robust.


## Concrete gradients -  Gumble Softmax trick (estimating discrete with continuous variable)
Estimates the discrete categorical variable with a continuous analog (i.e. softmax) and then uses the path-wise gradient to produce a low-variance but biased gradient. 

Replacing every discrete random variable in a model with a Concrete (continuous estimation of discrete) random variable results in a continuous model where the re-parameterization trick is applicable. The gradients are biased with respect to the discrete model, but can be used effectively to optimize large models. The tightness of the relaxation is controlled by a temperature hyper-parameter. In the low temperature limit, the gradient estimates become unbiased, but the variance of the gradient estimator diverges, so the temperature must be tuned to balance bias and variance.


## Straight-through gradient
-  Pretend the stochastic variable is identity in the backward pass where you need gradients. 

##Conditioning only on the Markov blanket, Ranganath et al. (2014) / local learning signals, Mnih and Gregor (2014). 
This is the most important variance reduction technique, since the stochastic gradients no longer scale with the number of latent variables but rather with the size of their Markov blanket.


## local expectation gradients by Titsias (2015) 
analytically calculates the expectations over discrete latent variables with small support. 


## MuProp gradient

Writing the gradient of the ELBO as \mathbb{E}_{q(z)}[\nabla_{\lambda} \log q(z; \lambda) f(z)] where f(z)=\log p(x,z) - \log q(z; \lambda)f(z)=logp(x,z)−logq(z;λ). 

Monte Carlo estimates of
$${\mathbb{E}_{q(z)}[\nabla_{\lambda} \log q(z; \lambda) (f(z) - h(z))] + \mu, \quad \mu = \mathbb{E}_{q(z)}[\nabla_{\lambda} \log q(z; \lambda) h(z)]}$$
are still unbiased, and appropriate choices of the control variate h(z)h(z) can reduce the variance of the estimator.

The authors propose a control variate based on a first-order Taylor expansion of f(z) around a fixed value $$\bar z$$​. By doing so, they are able to use gradient information from the model, evaluated at that point:
 $${h(z) = f(\bar z) + f'(\bar z)(z - \bar z)}$$. 
The gradient of the ELBO simplifies to (Eq.3 in the paper)

$${\mathbb{E}_{q(z)}\Big[\nabla_{\lambda} \log q(z; \lambda) \Big(f(z) - f(\bar z) - f'(\bar z)(z - \bar z)\Big)\Big] + f'(\bar z)\nabla_\lambda \mathbb{E}_{q(z)}[z].}$$

Of course, calculating $$f'(\bar z)$$ is not feasible because of the discrete variables. To address this they apply a “deterministic mean-field network” as an approximation. This enables backpropagation over the mean values of the discrete distributions, rather than over samples from the discrete distribution. 


## Rebar gradient (discrete variable):

Rebar gradient combines Reinforce gradients with gradients of the Concrete variable through a novel control variate for Reinforce.  It produces a low-variance, and unbiased gradient. 


We sought an estimator that is low variance, unbiased, and does not require tuning additional hyper-parameters. To construct such an estimator, we introduce a simple control variate based on the difference between the REINFORCE and the re-parameterization trick gradient estimators for the relaxed model. This reduces variance, but does not outperform state-of-the-art methods on its own. Our key contribution is to show that it is possible to conditionally marginalize the control variate to significantly improve its effectiveness.









