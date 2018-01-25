---
layout: article
title: Estimating Gradients of expectations
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---


Back-propagation (Rumelhart & Hinton, 1986), computes exact gradients for deterministic and differentiable objective functions but is not applicable if there is stochasticity or non-differentiable functions involved. That is the case when we want to calculate the gradient of an expectation of a function with respect to parameters $$\theta$$ i.e. $$ \nabla_\theta (E_q(z) [f(z)])=\nabla_\theta( \int q(z)f(z))$$ . An example is ELBO where gradient is difficult to compute since the expectation integral is unknown or the ELBO is not differentiable.



# permutations with Gumbel-Sinkhorn 
Learning permutation latent variable models requires an intractable marginalization over the combinatorial objects. 

The paper approximates discrete maximum-weight matching using the continuous Sinkhorn operator. Sinkhorn operator is attractive because it functions as a simple, easy-to-implement analog of the softmax operator. Gumbel-Sinkhorn is an extension of the Gumbel-Softmax to distributions over latent matchings.
https://openreview.net/pdf?id=Byt3oJ-0W

Notice that choosing a category can always be cast as a maximization problem (e.g. argmax of a softmax on categories). Similarly, one may parameterize the choice of a permutation $$P$$ through a square matrix $$X$$, as the solution to the linear assignment problem with $$P_N$$ denoting the set of permutation matrices. The matching operator can parameterize the hard choice of permutations with an argmax on the inner product of the matrix $$X$$ and the set of $$P_N$$ matrices i.e. $$M(X) = argmax <P,X>$$. They approximate $$M(X)$$ with the Sinkhorn operator. Sinkhorn normalization, or Sinkhorn balancing iteratively normalizes rows and columns of a matrix.


# gradient estimators for discrete variables 

## REINFORCE

$$g_reinforce[f] = f(b) grad_\theta \log p(b|\theta), $$

To calculate the gradient of the expectation, we first take the gradient operator $$\nabla_\theta$$ inside the integral to rewrite it as $$\int \nabla_\theta(q(z)) f(z) dz$$ given that only the $$q(z)$$ is a function of $$\theta$$. The only condition is that $$q(z)$$ be differentiable over $$\theta$$ almost everywhere. 

Then we use the log derivative trick (using the derivative of the logarithm $d (log(u))= d(u)/u$) on the (ELBO) and re-write the integral as an expectation $$\nabla_\theta (E_q(z) [f(z)]) = E_q(z) [\nabla_\theta \log q(z) f(z)]$$. This estimator now only needs the derivative $$\nabla \log q_\theta (z)$$ to estimate the gradient. The expectation will be replaced with a Monte Carlo Average. When the function we want derivative of is log likelihood, we call the derivative $\nabla_\theta \log ⁡p(x;\theta)$ a score function. The expected value of the score function is zero.[](http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/)

The form after applying the log-derivative trick is called the score ratio. This gradient is also called REINFORCE gradient or likelihood ratio. We can then obtain noisy unbiased estimation of this gradients with Monte Carlo. To compute the noisy gradient of the ELBO we sample from variational approximate q(z;v), evaluate gradient of log q(z;v), and then evaluate the log p(x, z) and log q(z). Therefore there is no model specific work and and this is called black box inference. 

The problem with this approach is that sampling rare values can lead to large scores and thus high variance for gradient. There are a few methods that use control variates to help with reducing the variance but with a few more non-restricting assumptions we can find a better method with low variance i.e. pathwise gradients. 


- An example of a non-differentiable function is a loss function that is defined by an expectation over a collection of random variables. For example, in case of a few possible discrete actions in RL, the policy net is a classifier with softmax output over possible categories of actions. We don't have immediate target values, and the reward is delayed. therefore, the loss function is defined by an expectation over a sequence of action random variables. Estimating the gradient of this loss function, using samples, is required so that we can backpropagate through the policy netword and adjust policy net parameters. the loss functions and their gradients are intractable, as they involve either a sum over an exponential number of latent variable configurations, or high-dimensional integrals that have no analytic solution. Monte-Carlo gradient estimators (Reinforce is an example) are common. 

- we sample the softmax output of the policy neural net to get a discrete action, then, take log, then we can multiply the logprob by reward and sum it up. 


- [pytoch implementation](https://github.com/JamesChuanggg/pytorch-REINFORCE/blob/master/reinforce_discrete.py)



``` python
probs = self.model(Variable(state)) # run policy net, get softmax output on categories of possible actions
action = probs.multinomial().data # sample to get action index and choose a category
prob = probs[:, action[0,0]].view(1, -1) # index probs with action selection
log_prob = prob.log() # compute log prob


```




## Gumble Softmax:
Estimates the discrete categorical variable with a continuous analog (i.e. softmax) and then uses the path-wise gradient (reparameterization trick) to produce a low-variance but biased gradient. 

Replacing every discrete random variable in a model with a Concrete (continuous estimation of discrete) random variable results in a continuous model where the re-parameterization trick is applicable. The gradients are biased with respect to the discrete model, but can be used effectively to optimize large models. The tightness of the relaxation is controlled by a temperature hyper-parameter. In the low temperature limit, the gradient estimates become unbiased, but the variance of the gradient estimator diverges, so the temperature must be tuned to balance bias and variance.


``` python 
def sample_gumbel(self, shape, eps=1e-20): 
    """Sample from Gumbel(0, 1)"""
    U = autograd.Variable(t.FloatTensor(shape).uniform_(0,1))
    return -t.log(-t.log(U + eps) + eps)

def gumbel_softmax_sample(self, logits, temperature): 
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + self.sample_gumbel(logits.size())
    return self.softmax( y / temperature)
```

-  If we pretend the stochastic variable is identity in the backward pass where you need gradients, this would be the straight through gradient.



## Rebar gradient (discrete variable):
- The main idea is to use a control variate to reduce the variance of a Monte Carlo (e.g. reinforce) estimator. i.e. $$g_{new} (b) = g(b) - control(b) + E_{p(b)}[control(b)] $$.

Rebar gradient combines Reinforce gradients with gradients of the Concrete variable through a novel control variate for Reinforce.  It produces a low-variance, and unbiased gradient. 

- The main idea is to use reinforce as the gradient estimator, $$g(b)$$, and CONCRETE estimator as the control variate, control(b).

We sought an estimator that is low variance, unbiased, and does not require tuning additional hyper-parameters. To construct such an estimator, we introduce a simple control variate based on the difference between the REINFORCE and the re-parameterization trick gradient estimators for the relaxed model. This reduces variance, but does not outperform state-of-the-art methods on its own. Our key contribution is to show that it is possible to conditionally marginalize the control variate to significantly improve its effectiveness.


an implementation:
- https://github.com/Bonnevie/rebar/blob/master/rebar.py

```
def rebar(params, est_params, noise_u, noise_v, f):
log_temp, log_eta = est_params
```

## Relax gradient (discrete/continuous/blackbox gradient):

- if the function we require gradient of is discrete, then continuous relaxation of it to interpolate values at points where it doesn't exist can use a variety of function. Relax uses a NN to learn that function. 

- The main idea is to use reinforce as the gradient estimator, $$g(b)$$, a reinforce estimator for a learned control variate function, control(b), and the reparameterization gradient for expectation of the control variate. 

It makes a general framework for learning low-variance, unbiased gradient estimators for black-box functions of random variables. Uses gradients of a neural network trained jointly with model parameters.


- https://github.com/duvenaud/relax/blob/master/relax-autograd/relax.py