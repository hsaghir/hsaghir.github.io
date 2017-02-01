---
layout: article
title: Unifying Linear Gaussian Models 
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---
Many common techniques for time series analysis (i.e. Factor analysis, PCA, ICA, Mixture of Gaussians, Vector quantization, Kalman Filter, HMM) can be seen as variants of one underlying model! 

Fundamental model is a discrete Linear dynamical system with Gaussian noise:

$x_t+1=Ax_t+w_0,  w_0~N(0, Q)$

$y_t=Cx_t+v_0, v_0~N(0, R)$

A is the transition matrix, C is the generative matrix,  w_0 and v_0 are noise vectors from Gaussian priors which are independant of each other, x, y and uncorrelated from time step to time step (white noise). The noise process does not have any notion of the time. x is then a first-order Gauss-Markov random process. Without noise $w_0$, the process would always shrink or explode exponentially, based on the eigenvalues of A. Without generation noise $v_0$, the latent variable $x_t$ will be observable. Additionally, there is degenracy in the model meaning that all the structure in Q covariance matrix can be moved into A and C matrices and therefore, We can assume diagonal covariance matrix (i.i.d noise). This is not the case for R since $y_t$ are observed and we can't rescale. 

Note the duality of an LTI system. Primal LTI system in state space:

x˙(t) = Ax(t) + Bu(t),
y(t) = Cx(t)

And its Dual LTI system in state space,
x˙(t) = AT x(t) + CT u(t),
y(t) = BT x(t).

The primal system is observable if and only if the dual system in controllable. The primal system is controllable if and only if the dual system in observable.

Linear Gaussian models are popular since the sum of two Gaussians is also Gaussian which means that all time outputs of a linear model would be Gaussian with initial Gaussian input, i.e.:

$p(x_t+1|x_t) ~ N(Ax_t, Q)$

$p(y_t|x_t) ~N(Cx_t, R)$

Due to the markov property, each time point is only dependant on the current state which means the before and after observations are conditionally independant given current state. 

In some cases we know what our hidden variables are and want to estimate them, for example, vision where the hidden variable might be location of an object. In such situations we know estimates of transition and generation matrices based on physics (Sensor Fusion / Filtering / Smoothing) and want to accurately estimate them based on data so we care about inferece (Maximum a posteriori!?). 

In other cases, we want to discover causes and don't have a model (system identification / Learning / Causal Inference) so we care about finding parameters that model the data well (Maximum likelihood!?).

The fundamental algorithm for learning in all such models is the EM algorithm. The objective of the EM algorithm is to maximize the liklihood of observed data in the presence of hidden variables. The basic idea is; given observations and the current model parameters (staring with random), use the solution to the filtering/smoothing problem to estimate the unknown latent states (Expectation /E step). Then use this ficticious complete data (observations and latents) to solve for new model parameters (Maximization - M step). Do this iteratively until convergence!