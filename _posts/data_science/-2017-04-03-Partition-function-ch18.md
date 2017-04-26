---
layout: article
title: Partition function Ch18
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

- The partition function, $$Z(\theta)$$ is an integral or sum over the unnormalized probability of all states, $$Z=\sum_x p_unnormalized (x;\theta) dx$$. It's mostly an intractable integral that is a function of model parameters $$\theta$$.

- Some deep learning models are designed to have a tractable partition function, or do not require calculating the partition function. However, for the rest of models we must confront it. 

## Log-Likelihood Gradient

- In undirected probability models, we can write the log-likelihood as $$P(x)=\log \frac{P_unnormalized (x)}{Z} = \log P_unnormalized (x)-\log Z(\theta)$$. The gradient of log-likelihood  will have both terms since the partition function is also a function of model parameters $$\theta$$. These two gradient terms are called the positive phase (unnormalized probability) and negative phase (partition funcgtion) of learning. 

- Models with no latent variables or with few interactions between latent variables typically have a tractable positive phase (due to conditional independence of variables). The quintessential example of a model with a straightforward positive phase and difficult negative phase is the RBM. If we can calculate the gradient of the log-likelihood, $$\Delta_theta p(x, \theta)= \Delta_theta \log P_unnormalized (x)-\Delta_theta \log Z(\theta)$$,  we would be able to use it for optimization in maximium likelihood learning of parameters. 

- If the positive phase is tractable, we can use Monte Carlo methods to approximate the negative phase (gradient of normalizer) using the following expectation $$\Delta_theta \log Z = E_p(x) [\Delta_theta \log p_unnormalized (x)]. The derivation of this expectation is straight forward. We first write the $$\Delta_theta \log Z= \frac{\Delta_theta Z}{Z}$$ using the gradient of $$\log$$ trick and take the $$\Delta_theta$$ into the sum definition of $$Z$$ i.e. $$\frac{\sum_x \Delta_theta p_unnormalized (x)}{Z}$$. We then use the $$\exp(\log(.))$$ trick and replace $$p_unnormalized (x)$$ and use the gradient of exponential property and get $$\frac{\sum_x \exp(\log p_unnormalized (x)) \Delta_theta p_unnormalized (x)}{Z}$$. Replacing the $$\frac{p_unnormalized (x)}{Z}$$ with the normalized probability $$p(x)$$, we arrive at the expectation on $$p(x)$$.

- Monte Carlo method provides a flexible framework for learning undirected models. In the positive phase, we increase $$\log p_unnormalized (x)$$ for $$x$$ drawn from the data (push up on energy for data manifold). In the negative phase, we decrease the partition function by decreasing $$\log p_unnormalized (x)$$ drawn from the model distribution as points that the model believes in strongly- i.e. halucinations (push down on energy on points not on data manifold). This loop stops when an equilibrium is reached between data distribution and model halucinations.

## Stochastic Maximum Likelihood and Contrastive Divergence

- The naive way of calculating the negative phase is to compute the gradient by burning in a set of Markov chains from a random initialization every time the gradient is needed (each gradient step) which is not feasible. The main cost is buring in the Markov chain from a random initialization, so an obvious fix is to initialize the Markov chains from a distribution that is very close to the model distribution. Contrastive divergence does exactly that by initializing the Markov chain at each step with samples from the data distribution. Initially, the data distribution is not close to the model distribution, so the negative phase is not very accurate, but as the positive phase acts more, the two distributions get closer and the negative phase will get more accurate. 

- CD can only push energy up on low probability regions near the data manifold. It fails to suppress regions of high probability that are far from actual training examples. These regions are called spurious modes. This happens since spurious modes won't be visited by a Markov chains initialized at training points. CD is biased due to the approximation of negative phase. It can be interpreted as discarding the smallest terms of the correct MCMC update gradient, which explains the bias. 

- CD can be used in layer-wise greedy training but not directly in a deeper model. That's because there is no data distribution for hidden units to be used as initialization points. A different strategy that resolves many of the problems with CD is to initialize the Markov chains at each gradient step with their states from the previous gradient step. This approach was first discovered under the name stochastic maximum likelihood (SML) or persistent contrastive divergence (PCD). The basic idea of this approach is that, so long as the steps taken by the stochastic gradient algorithm are small, then the model from the previous step will be similar to the model from the current step. Since the MCMC continues instead of restarting at each gradient step, the chains are free to wander far enough to find all of the model’s modes and thus are more resistant to spurious modes. Moreover, because it is possible to store the state of all of the sampled variables, whether visible or latent, SML provides an initialization point for both the hidden and visible units. 

SML is able to train deep models efficiently, however, SML is vulnerable to becoming inaccurate if the stochastic gradient algorithm can move the model faster than the Markov chain can mix between steps. This can happen if k is too small or $$\epsilon$$ is too large. Subjectively, if the learning rate is too high for the number of Gibbs steps, the human operator will be able to observe that there is much more variance in the negative phase samples across gradient steps rather than across different Markov chains.


## Pseudolikelihood

- Instead of estimating the partition function, we can use models that sidestep the normalizer. Most of these approaches are based on the observation that it is easy to compute ratios of probabilities in an undirected probabilistic model since the partition function cancels out. The pseudolikelihood is based on the observation that conditional probabilities take this ratio-based form, and thus can be computed without knowledge of the partition function.

## Score Matching and Ratio Matching

- Score matching trains a model without estimating Z or its derivatives. The derivatives of a log density with respect to its argument is called its score. Score matching minimizes a loss function which is expected squared difference between the derivatives of the model’s log density with respect to the input and the derivatives of the data’s log density with respect to the input. This objective function avoids the difficulties associated with differentiating the partition function Z because Z is not a function of x. computing the score of the data distribution requires knowledge of the true distribution generating the training data, pdata. Fortunately, minimizing the expected value of equivalent to minimizing the expected value of

- Because score matching requires taking derivatives with respect to x, it is not applicable to models of discrete data. However, the latent variables in the model may be discrete.

- Ratio matching is a variant of score matching that applies specifically to binary data. Ratio matching consists of minimizing the average over examples of an objective function. As with the pseudolikelihood estimator, ratio matching can be thought of as pushing down on all fantasy states that have only one variable different from a training example. Since ratio matching applies specifically to binary data, this means that it acts on all fantasy states within Hamming distance 1 of the data.

## Denoising Score Matching

- Denoising score matching is especially useful because in practice we usually do not have access to the true pdata but rather only an empirical distribution defined by samples from it. Any consistent estimator will, given enough capacity, make pmodel into a set of Dirac distributions centered on the training points. Smoothing by q helps to reduce this problem, at the loss of the asymptotic consistency property.

- several autoencoder training algorithms are
equivalent to score matching or denoising score matching. These autoencoder
training algorithms are therefore a way of overcoming the partition function
problem.

## Noise-Contrastive Estimation

- Most techniques for estimating models with intractable partition functions do not
provide an estimate of the partition function. SML and CD estimate only the
gradient of the log partition function, rather than the partition function itself.
Score matching and pseudolikelihood avoid computing quantities related to the
partition function altogether.
Noise-contrastive estimation (NCE) (Gutmann and Hyvarinen, 2010) takes a
different strategy. In this approach, the probability distribution estimated by the
model is represented explicitly as

log pmodel(x) = log p˜model(x; θ) + c, (18.28)

where c is explicitly introduced as an approximation of − log Z(θ). Rather than
estimating only θ , the noise contrastive estimation procedure treats c as just
another parameter and estimates θ and c simultaneously, using the same algorithm
for both. The resulting log pmodel(x) thus may not correspond exactly to a valid
probability distribution, but will become closer and closer to being valid as the
estimate of c improves.1

- NCE works by reducing the unsupervised learning problem of estimating p(x)
to that of learning a probabilistic binary classifier in which one of the categories
corresponds to the data generated by the model. This supervised learning problem
is constructed in such a way that maximum likelihood estimation in this supervised learning problem defines an asymptotically consistent estimator of the original
problem. Specifically, we introduce a second distribution, the noise distribution pnoise(x).
The noise distribution should be tractable to evaluate and to sample from. We
can now construct a model over both x and a new, binary class variable y. y is a switch variable that determines whether we will generate x from the model or from the noise distribution.

- The special case of NCE where the noise samples
are those generated by the model suggests that maximum likelihood can be
interpreted as a procedure that forces a model to constantly learn to distinguish
reality from its own evolving beliefs, while noise contrastive estimation achieves
some reduced computational cost by only forcing the model to distinguish reality
from a fixed baseline (the noise model).

- Using the supervised task of classifying between training samples and generated
samples (with the model energy function used in defining the classifier) to provide
a gradient on the model was introduced earlier in various forms

- Noise contrastive estimation is based on the idea that a good generative model
should be able to distinguish data from noise. A closely related idea is that
a good generative model should be able to generate samples that no classifier
can distinguish from data. This idea yields generative adversarial networks

## Estimating the Partition Function

- Most techniques upto now describe methods that avoid needing to compute the intractable partition function Z(θ) associated with an undirected graphical model, in this section we discuss several methods for directly estimating the partition function. Estimating the partition function can be important in evaluating the model, monitoring training performance, and comparing models to each other. Ratio of partition functions can be obtained using importance sampling. 

- Two related strategies can be used for estimating partition functions for complex distributions over highdimensional spaces: annealed importance sampling and bridge sampling. Both start with the simple importance sampling strategy and both attempt to overcome the problem of the proposal p0 being too far from p1 by introducing intermediate distributions that attempt to bridge the gap between p0 and p1.

### annealed importance sampling 


### bridge sampling