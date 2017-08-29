---
layout: article
title: Time series prediction 
comments: true
image:
  teaser: jupyter-main-logo.svg
---

Traditional methods of time-series analysis are mainly concerned with decomposing a series into:
- Trend: long-term change in the mean level.
- Seasonal: periodicity.
- stochastic: assumed noise model.

After detrending and removing periodicity, traditional time series analysis are concerned with infering the parameters of the assumed noise model. The stochastic approach to time series makes the assumption that a time series is a realization of a stochastic process (i.e. a noise model). This stochastic process can be modeled with a generative model and the statistical time-series modeling is concerned with inferring the properties of the probability model which generated the observed time series.

[confer this for Gaussian latent variable models GLMs](http://www.flutterbys.com.au/stats/tut/tut12.9.html)

This is a generative modeling problems. Three main approaches to generative modeling include 

1. Autoregressive models
2. Latent variable models
3. Transformation models

A stochastic process is said to be strictly stationary if the joint distribution of the process does not change with time.


1. A true autoregressive approach is one where the model directly observes the data without any latent factor e.g. an RNN or an AR(k). Such models don't put any noise assumptions on the data and the learning is easy (might even have a gaurantee!?).

- AR(p): Autoregressive model of order $p$ (deterministic linear combination/linear regression of the past $p$ outputs plus a global Gaussian noise). The partial autocorrelation of an AR(p) process becomes zero at lag $p + 1$ and greater.  partial correlation is the correlation between the residuals RX and RY resulting from the linear regression of X with Z and of Y with Z (z is control variable), so we determine the appropriate maximum lag for the estimation by examining the sample partial autocorrelation function.


2. Traditional autoregressive approaches are all latent variable models with different dependancy structure between latents and observables. 

- MA(q): Moving average model of order $q$ (mean plus linear combination/linear regression of past l Gaussian noises on prediction error $N(0, \gamma)$ which are mutually independant). The role of the noises (random shocks) in the MA model differs from their role in the autoregressive (AR) model in two ways. First, they are propagated to future values of the time series directly through their linear combination. Second, in the MA model a shock affects values only for the current period and $q$ periods into the future; in contrast, in the AR model a shock affects values infinitely far into the future, due to its indirect impact noise at t affects $x_t$ which affects $x_{t+1}$ and so on. The autocorrelation function of an MA(q) process becomes zero at lag q + 1 and greater, so we determine the appropriate maximum lag for the estimation by examining the sample autocorrelation function.

- ARMA(p,q): combination of AR(p) and MA(q). An ARMA model can be represented by a directed graphical model (or Bayes net) with both stochastic (MA) and deterministic (AR) nodes. observable variables (the Y ’s) are represented by deterministic nodes and the error variables (the E’s) are represented by stochastic nodes.

- ARIMA(k, l, d): Order d integrated AR(k) with MA(l) : An ARIMA model for a time series is simply an ARMA model for a preprocessed version of that same time series. This preprocessing consists of d consecutive differencing transformations, where each transformation replaces the observations with the differences between successive observations. For example, when d = 0 an ARIMA model is a regular ARMA model, when d = 1
an ARIMA model is an ARMA model of the differences. 

- many more variations..
- Kalman filter: Linear dynamical system with latent and observation Gaussian noise. 

Non-linear models:
- ARCH(k): Autoregressive Conditional Heteroskedasticity of order k

These models put a strong assumption on the noise terms and there is no learning/inference gaurantee.

Non-parametric approaches !?
Based on their properties, stochastic processes can be divided into various categories, which include random walks, martingales, Markov processes, Lévy processes, Gaussian processes, and random fields as well as renewal processes and branching processes. 

- Bernoulli process, is a sequence of independent and identically distributed (iid) random variables of binary states.
- Random walks are stochastic processes that are usually defined as sums of iid random variables or random vectors in Euclidean space.
- The Wiener process (Brownian motion) is a stochastic process with stationary and independent increments that are normally distributed based on the size of the increments (continuous version of the simple random walk).
- Poisson process can be defined as a counting process, which is a stochastic process that represents the random number of points or events up to some time

- Markov processes are stochastic processes, traditionally in discrete or continuous time, that have the Markov property, which means the next value of the Markov process depends on the current value, but it is conditionally independent of the previous values of the stochastic process. In other words, the behavior of the process in the future is stochastically independent of its behavior in the past, given the current state of the process.The Brownian motion process and the Poisson process (in one dimension) are both examples of Markov processes.
- A martingale is a discrete-time or continuous-time stochastic process with the property that the expectation of the next value of a martingale is equal to the current value given all the previous values of the process. A symmetric random walk and a Wiener process (with zero drift) are both examples of martingales
- Lévy processes are types of stochastic processes that can be considered as generalizations of random walks in continuous time
- A point process is a collection of points randomly located on some mathematical space such as the real line. 

A key fact of Gaussian processes is that they can be completely defined by their second-order statistics (Variance). Thus, if a Gaussian process is assumed to have mean zero, defining the covariance function completely defines the process' behaviour. Basic aspects that can be defined through the covariance function are the process' stationarity, isotropy, smoothness and periodicity. Gaussian processes translate as taking priors on functions and the smoothness of these priors can be induced by the covariance function. If we expect that for "near-by" input points x and x' their corresponding output points y and y' to be "near-by" also, then the assumption of continuity is present. If we wish to allow for significant displacement then we might choose a rougher covariance function. A common covariance function might be Gaussian Noise: $K_{\text{GN}}(x,x')=\sigma ^{2}\delta _{x,x'}$ where Here $d=x-x'$. The parameter $l$ is the characteristic length-scale of the process (practically, "how close" two points $x$ and $x'$ have to be to influence each other significantly). $δ$ is the Kronecker delta and $σ$ the standard deviation of the noise fluctuations.

3. Transformation models are the likes of a GAN where a simple noise model is transformed to a complex distribution. 

can we transform a cleverly designed LDS to the complex distribution of a time series using an RNN/dilated CNN network? This might be interesting from generative/prediction point of view but not causal inference. 


The basic problem is that generative neural network models seem to either be stable but fail to properly capture higher-order correlations in the data distribution (which manifests as blurriness in the image domain), or they are very unstable to train due to having to learn both the distribution and the loss function at the same time, leading to issues like non-stationarity and positive feedbacks. The way GANs capture higher order correlations is to say ‘if there’s any distinguishable statistic from real examples, the discriminator will exploit that’. That is, they try to make things individually indistinguishable from real examples, rather than in the aggregate. The cost of that is the instability arising from not having a joint loss function – the discriminator can make a move that disproportionately harms the generator, and vice versa.

Other methods are stable, but have difficulty in data spaces dominated by higher-order correlations. Variational autoencoders are an example of this. By imposing a certain distribution on the latent space, they provide a principled way to generate new samples from the data distribution which occur at the correct relative probabilities. VAEs tend to suffer blurriness in the output, however, especially when the entropy is high. This is because the user must specify a similarity metric in the data space by hand. That is to say, when training a variational autoencoder, one must say ‘this image is closer to the target than that one’, and that definition of closeness is usually taken to be per-pixel mean-squared-error (MSE). The problem with this is that it implies that the distribution of images should be modeled by a set of independent gaussian distributions for each pixel, whereas perceptually it’s the higher-order correlation between pixels that is more important to a human observer. For example, an image of a circle can be shifted by one pixel to the left and still appear as the same circle, but in terms of per-pixel MSE, this corresponds to a huge fluctuation in the pixels at the edge. Because of this, methods using functional or perceptual information in place of MSE can be used to improve the quality of images generated by a non-adversarial setup (paper). Alex Champandard (@alexjc) has made a number of investigations along these lines as well.