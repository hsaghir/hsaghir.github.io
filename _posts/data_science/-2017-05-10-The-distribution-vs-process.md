---
layout: article
title: Information Geometry
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

- A probability distribution is the reciprocal of energy for a random variable telling how likely the values of that random variable are. You can think of a random variable as a piece of land and the probability distribution as the density of cloud above this land. 

- A random process is a collection of random variables indexed by some mathematical set (each variable uniquly associated with an element of the set). Historically this set has been natural numbers (1,2,..) representing the evolution of some system over time. Each random variable takes values from a single mathematical space known as the state space. A random process can have many different realizations due to its randomness.

- When a process is associated with a ditribution, it usually means that the random variables in the process each have a probability distribution similar to that specific distribution (state space). If the random variables are indexed by the Cartesian plane or some higher-dimensional Euclidean space, then the collection of random variables is usually called a random field instead.

- Based on their properties, stochastic processes can be divided into various categories, which include random walks, martingales, Markov processes, Lévy processes, Gaussian processes, and random fields as well as renewal processes and branching processes. 

- Bernoulli process, is a sequence of independent and identically distributed (iid) random variables of binary states.
- Random walks are stochastic processes that are usually defined as sums of iid random variables or random vectors in Euclidean space.
- The Wiener process (Brownian motion) is a stochastic process with stationary and independent increments that are normally distributed based on the size of the increments (continuous version of the simple random walk).
- Poisson process can be defined as a counting process, which is a stochastic process that represents the random number of points or events up to some time

- Markov processes are stochastic processes, traditionally in discrete or continuous time, that have the Markov property, which means the next value of the Markov process depends on the current value, but it is conditionally independent of the previous values of the stochastic process. In other words, the behavior of the process in the future is stochastically independent of its behavior in the past, given the current state of the process.The Brownian motion process and the Poisson process (in one dimension) are both examples of Markov processes.
- A martingale is a discrete-time or continuous-time stochastic process with the property that the expectation of the next value of a martingale is equal to the current value given all the previous values of the process. A symmetric random walk and a Wiener process (with zero drift) are both examples of martingales
- Lévy processes are types of stochastic processes that can be considered as generalizations of random walks in continuous time.
- A point process is a collection of points randomly located on some mathematical space such as the real line. 


# Bayesian nonparametrics based on Levy processes

- Lévy processes are types of stochastic processes that can be considered as generalizations of random walks in continuous time. The main defining characteristic of these processes is their stationarity property, so they were known as processes with stationary and independent increments. Increaments are all independent of each other, and the distribution of each increment only depends on the difference in time. Important stochastic processes such as the Wiener process, the homogeneous Poisson process (in one dimension), and subordinators are all Lévy processes.

- Gaussian Processes. A key fact of Gaussian processes is that they can be completely defined by their second-order statistics (Variance). Thus, if a Gaussian process is assumed to have mean zero, defining the covariance function completely defines the process' behaviour. Basic aspects that can be defined through the covariance function are the process' stationarity, isotropy, smoothness and periodicity. Gaussian processes translate as taking priors on functions and the smoothness of these priors can be induced by the covariance function. If we expect that for "near-by" input points x and x' their corresponding output points y and y' to be "near-by" also, then the assumption of continuity is present. If we wish to allow for significant displacement then we might choose a rougher covariance function. A common covariance function might be Gaussian Noise: $K_{\text{GN}}(x,x')=\sigma ^{2}\delta _{x,x'}$ where Here $d=x-x'$. The parameter $l$ is the characteristic length-scale of the process (practically, "how close" two points $x$ and $x'$ have to be to influence each other significantly). $δ$ is the Kronecker delta and $σ$ the standard deviation of the noise fluctuations.
