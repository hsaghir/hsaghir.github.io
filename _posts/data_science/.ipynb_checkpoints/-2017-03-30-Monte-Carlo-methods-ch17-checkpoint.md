---
layout: article
title: Monte Carlo methods Ch17
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

- Randomized algorithms use the ability to sample from a distribution to solve intractable problems numerically. There are two categories: Las Vegas and Monte Carlo algorithms. Las Vegas algorithms always return precise correct answer or report that they failed, but need random resources. Monte Carlo algorithms return uncertain answers with error, but can improve the estimate by expending more resources. 

- Sampling provides a flexible way to approximate many sums and integrals at reduced cost. The idea is to view the sum or integral as if it was an expectation under some distribution and to approximate the expectation by a corresponding average. For example, we use batch gradient with SGD instead of whole dataset gradient which is costly. On the other hand, gradient of partition function is intractable but we can estimate it using MCMC.

- The law of large numbers states that if the samples are i.i.d.,then the average converges almost surely to the expected value and its distribution (uncertainty) converges to a normal distribution with emprical average and variance (emprical variance divided by the number of samples). 

- Therefore, the ability to easily draw samples from distributions is crucial. If it is not feasible to sample from the distribution exactly, we either use importance sampling, or form a sequence of estimators that converge towards the distribution of interest and iteratively update them to get samples from the  (MCMC). 

- The idea is to view the sum or integral, $$\int_x~p(x){f(x)}$$, as if it was an expectation under some distribution. Therefore,  We decompose the original problem to a distribution, $$p(x)$$ and function, $$f(x)$$, and multiply and divide it by a known distribution, $$q(x)$$, that is easy to sample from. We then take the numerator $$q(x)$$ into the definition of expectation ,$$E_q [\frac{p(x)f(x)}{q(x)}]$$. We can draw $$n$$ sample from $$q$$ and average $$\frac{1}{n}*\sum_n{\frac{p(x)f(x)}{q(x)}}$$ as an estimate for the expectation. This is called unbiased importance sampling estimator. Any choice of sampling distribution $$q$$ is valid, however, the variance of the estimator can be greatly sensitive to the choice of $$q$$. Better importance sampling distributions put more weight where the integrand, $$f(x)$$, is larger. 

- We introduce $$q(x)$$ instead of directly using $$p(x)$$ as distribution on the expectation since $$p(x)$$ is usually not known or is not easy to sample from. Additionally, using $$q$$ has the advantage of converting the problem from density estimation to density ratio estimation that does not require normalized $$p$$ or $$q$$ by dividing the estimator by $$\sum_n{\frac{p(x)}{q(x)}}$$. This is called biased importance sampling. 

- If there are samples of q for which $$\frac{p(x)f(x)}{q(x)}$$ is large, the variance of the estimator can get very large. The $$q$$ distribution is usually chosen to be a very simple distribution so that it is easy to sample from. When $$x$$ is high-dimensional, this simplicity in $$q$$ causes it to match $$p(x)$$ or $$p(x)|f(x)|$$ poorly and thus high variance of the estimator. 

- importance sampling has been used to estimate partition function, log-likelihood in deep directed models such as the variational autoencoder, improve the estimate of the gradient of the cost function (Sampling more difficult examples more frequently can reduce the variance of the gradient in such cases). 


#MCMC

- When there is no low variance importance sampling distribution $$q(x)$$, we might use MCMC. 

- In DL context, this usually happens when $$p_model(x)$$ is represented by an undirected model. In these cases, we introduce a mathematical tool called a Markov chain to approximately sample from $$p_model(x)$$.

-The most standard, generic guarantees for MCMC techniques are only applicable when the model does not assign zero probability to any state. In the EBM formulation, every state is guaranteed to have  on-zero probability. MCMC methods are in fact more broadly applicable and can be used with many probability distributions that contain zero probability states.

- The core idea of a Markov chain is to have a state x that begins as an arbitrary value. Over time, we randomly update x repeatedly. Eventually $$x$$ becomes (very nearly) a fair sample from $$p(x)$$ when we reach equilibrium. Formally, a Markov chain is defined by a random state $$x$$ and a transition distribution, $$T$$ specifying the probability that a random update will go to another state if it starts in state $$x$$. Running the Markov chain means repeatedly updating the state x to a new value sampled from $$T$$.

- Applying the Markov chain update repeatedly corresponds to multiplying by the transition matrix,  A, repeatedly. In other words, we can think of the process of running a Markov chain as exponentiating the matrix A. The matrix A has special structure because each of its columns represents a probability distribution. Such matrices are called stochastic matrices. 

- If there is a non-zero probability of transitioning from any state to any other state for some power t, then the Perron-Frobenius theorem (Perron, 1907; Frobenius, 1908) guarantees that the largest eigenvalue is real and equal to 1. Over time, we can see that all of the eigenvalues are exponentiated. This process causes all of the eigenvalues that are not equal to 1 to decay to zero. Under some additional mild conditions, A is guaranteed to have only one eigenvector with eigenvalue 1. The process thus converges to a stationary distribution, sometimes also called the equilibrium distribution. To be a stationary point, v must be an eigenvector with corresponding eigenvalue 1.

- all Markov chain methods consist of repeatedly applying stochastic updates until eventually the state begins to yield samples from the equilibrium distribution. Running the Markov chain until it reaches its equilibrium distribution is called “burning in” the Markov chain. After the chain has reached equilibrium, a sequence of infinitely many samples may be drawn from from the equilibrium distribution. They are identically distributed but any two successive samples will be highly correlated with each other. A finite sequence of samples may thus not be very representative of the equilibrium distribution. One way to mitigate this problem is to return only every n successive samples, so that our estimate of the statistics of the equilibrium distribution is not as biased by the correlation between an MCMC sample and the next several samples. 

- Markov chains are thus expensive to use because of the time required to burn in to the equilibrium distribution and the time required to transition from one sample to another reasonably decorrelated sample after reaching equilibrium. If one desires truly independent samples, one can run multiple Markov chains in parallel. This approach uses extra parallel computation to eliminate latency. The strategy of using only a single Markov chain to generate all samples and the strategy of using one Markov chain for each desired sample are two extremes; deep learning practitioners usually use a number of chains that is similar to the number of examples in a minibatch and then draw as many samples as are needed from this fixed set of Markov chains. A commonly used number of Markov chains is 100.

- Another difficulty is that we do not know in advance how many steps the Markov chain must run before reaching its equilibrium distribution. This length of time is called the mixing time. It is also very difficult to test whether a Markov chain has reached equilibrium. We do not have a precise enough theory for guiding us in answering this question. Theory tells us that the chain will converge, but not much more.

- If we analyze the Markov chain from the point of view of a matrix A acting on a vector of probabilities v, then we know that the chain mixes when $$A^t$$ has effectively lost all of the eigenvalues from A besides the unique eigenvalue of 1. This means that the magnitude of the second largest eigenvalue will determine the mixing time. However, in practice, we cannot actually represent our Markov chain in terms of a matrix. The number of states that our probabilistic model can visit is exponentially large in the number of variables, so it is infeasible to represent v, A, or the eigenvalues of A. Due to these and other obstacles, we usually do not know whether a Markov chain has mixed. Instead, we simply run the Markov chain for an amount of time that we roughly estimate to be sufficient, and use heuristic methods to determine whether the chain has mixed. These heuristic methods include manually inspecting samples or measuring correlations between successive samples.

## Gibbs Sampling

- the problem that the Markov chain was originally introduced to solve is the problem of representing the distribution of a large group of variables. We useda Markov chain to factorize the distribution using a composition of simple componenets that only locally depend of variables. 

- There are two ways to choos $$q$$ distribution in importance sampling through a Markov chain (i.e. transition function $$T(x^'|x)$$). we either (1) learn $$p_model$$ and derive $$T(x^'|x)$$ from it (e.g. Gibbs sampling for EBMs) or (2) directly parametrize $$T(x^'|x)$$ and learn it, so that its stationary distribution implicitly defines the $$p_model$$ (e.g. VAEs).

- Gibbs sampling is a conceptually simple and effective approach to building a Markov chain that samples from model distribution $$p_model(x)$$. Sampling from $$T(x^'|x)$$ is accomplished by selecting one variable $$x_i$$ and sampling it from the Markov chain, $$p_model$$, conditioned on its neighbors in the undirected graph defining the structure of the energy-based model. Several variables may be sampled at the same time if they are conditionally independent given all of their neighbors. For example, in RBMs, all of the hidden units are conditionally independent from each other given all of the visible units and can be sampled simultaneously (block Gibbs sampling).

- Alternate approaches to designing Markov chains to sample from $$p_model$$ are possible. For example, the Metropolis-Hastings algorithm is widely used in other disciplines. In the context of the deep learning approach to undirected modeling, it is rare to use any approach other than Gibbs sampling. Improved sampling techniques are one possible research frontier.

## MCMC mixing

- Ideally, successive samples from a Markov chain designed to sample from p(x ) would be completely independent from each other and would visit many different regions in x space proportional to their probability. Instead, especially in high dimensional cases, MCMC samples become very correlated.

- MCMC methods with slow mixing can be seen as inadvertently performing something resembling noisy gradient descent on the energy function. The chain tends to take small steps with a preference for moves that yield lower energy configurations. When starting from a rather improbable configuration (higher energy than the typical ones from p (x)), the chain tends to gradually reduce the energy of the state and only occasionally move to another mode. Once the chain has found a region of low energy, which we call a mode, the chain will tend to walk around that mode (following a kind of random walk). Once in a while it will step out of that mode and generally return to it or (if it finds an escape route) move towards another mode. The problem is that successful escape routes are rare for many interesting distributions, so the Markov chain will continue to sample the same mode longer than it should. Transitions between two modes that are separated by a high energy barrier are exponentially less likely. The problem arises when there are multiple modes with high probability that are separated by regions of low probability (e.g. classification problems).

- Sometimes this problem can be resolved by finding groups of highly dependent units and updating all of them simultaneously in a block.

- In models with latent variables $$p(x,h)$$, like RBM, we alternate between $$p(x|h)$$ and $$p(h|x)$$. For good mixing we'd like $$p(h|x)$$ to have high entropy to be able to sample from all modes while for representation learning, we'd like $$h$$ and $$x$$ to have high mutual information to be able to reconstruct $$x$$ well. These are confilicting goals. One way to resolve this problem is to make h be a deep representation, that encodes into in such x h a way that a Markov chain in the space of h can mix more easily. It remains however unclear how to exploit this observation to help better train and sample from deep generative models.

- When a distribution has sharp peaks of high probability surrounded by regions of low probability, it is difficult to mix between the different modes of the distribution. We can remedy this by constructing the target distribution in a way that the peaks are not as high and the surrounding valleys are not as low. This can be done by adding a new parameter to the Gibbs probability function (i.e. divide by tempreture) to control how sharp the peak probability is. When the temperature falls to zero the energy-based model becomes deterministic and when the temperature rises to infinity, the distribution becomes uniform. Tempering is a general strategy of mixing between modes rapidly by drawing samples with higher than 1 tempreture. Markov chains based on tempered transitions (Neal, 1994) temporarily sample from higher-temperature distributions in order to mix to different modes, then resume sampling from the unit temperature distribution.

- One possible reason that tempering hasn't been very effective yet is that there are critical temperatures around which the temperature transition must be very slow (as the temperature is gradually reduced) in order for tempering to be effective.

- Despite the difficulty of mixing, Monte Carlo techniques are useful and are often the best tool available. Indeed, they are the primary tool used to confront the intractable partition function of undirected models.