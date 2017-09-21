---
layout: article
title: Building generative model algorithms
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---



## Non-Parametric Models
- Parametric models assume a finite set of parameters can explain data. Therfore, given the model parameters, future predictions x, are independant of the observed data.

- Non-parametric models assume that the data distribution cannot be defined in terms of such a finite set of parameters. But they can often be defined by assuming an infinite dimensional parameter vector. Usually we think of infinte parameters as a function. Therfore, the amount of information stored in a nonparametric model grows with more data. 

Parametric | Non-parametric | -> Application
polynomial regression | Gaussian processes | -> function approx.
logistic regression | Gaussian process classifiers | -> classification
mixture models, k-means \ Dirichlet process mixtures | -> clustering
hidden Markov models | infinite HMMs | -> time series
factor analysis, probabilistic PCA, PMF | infinite latent factor models | -> feature discovery

### Gaussian Process
- Consider the problem of nonlinear regression: You want to learn a function f with error bars from data D = {X, y}. A Gaussian process defines a distribution over functions p(f) which can be used for Bayesian regression: p(f|D) = p(f)p(D|f)/p(D)

- A multilayer perceptron (neural network) with infinitely many hidden units and Gaussian priors on the weights (bayesian neural net) is a GP (Neal, 1996)


-In a Gaussian process, each data point is considered to be a random variable. We form a similarity matrix between all data points which is used as the covariance matrix for a multivariate Gaussian (uses negative exponensial of Euclidean distance). Since the joint distribution of data is a multivariate Gaussian, assuming that the joint distribution of test data and training data are also a multivariate Gaussian, prediction will be conditional probability of test data given training data from a multivariate Gaussian joint. In a multivariate Gaussian, joint and marginal probabilities can be analytically calculated using linear algebra on the covariance matrix. Therefore, prediction consists of simply performing linear algebra on covariance matrix (similarity matrix) of training data. Note that the choice of the distance measure (i.e. negative exponensial of Euclidean distance) is the modelling prior in a regression problem (e.g. if a linear distance is chosen, then it's linear regression!).

- Other variants of non-parametric models are 1) nearest neighbor regression, where the model would simply store all (x,y) training pairs and then just return an extrapolation of the nearest neighbor y values for each x in the test set. AS another variant, 2) we can simply wrap a parametric model inside a bigger model that scales with training data size; for example in a generalized linear model, we can simply increase the degree of polynomial by extending the data matrix X and number of parameters. 

- Gaussian processes are a very flexible non-parametric model for unknown functions, and are widely used for regression, classification, and many other applications that require inference on functions. Dirichlet processes are a non-parametric model with a long history in statistics and are used for density estimation, clustering, time-series analysis and modelling the topics of documents. To illustrate Dirichlet processes, consider an application to modelling friendships in a social network, where each person can belong to one of many communities. A Dirichlet process makes it possible to have a model whereby the number of inferred communities (that is, clusters) grows with the number of people. The Indian buffet process (IBP)40 is a non-parametric model that can be used for latent feature modelling, learning overlapping clusters, sparse matrix factorization, or to non-parametrically learn the structure of a deep network41. Elaborating the social network modelling example, an IBP-based model allows each person to belong to some subset of a large number of potential communities (for example, as defined by different families, workplaces, schools, hobbies, and so on) rather than a single community, and the probability of friendship between two people depends on the number of overlapping communities they have42. In this case, the latent features of each person correspond to the communities, which are not assumed to be observed directly. The IBP can be thought of as a way of endowing Bayesian non-parametric models with ‘distributed representations’, as popularized in the neural network literature. An interesting link between Bayesian non-parametrics and neural networks is that, under fairly general conditions, a neural network with infinitely many hidden units is equivalent to a Gaussian process.

- Standard variance calculations involve multiplying the covariance matrix by an inverse of the covariance matrix. However, calculating the inverse of a covariance matrix requires a substantial expansion of the matrix dimensions that can rapidly exhaust all available memory. 



## [GP Regression](http://www.gaussianprocess.org/gpml/chapters/RW2.pdf)


- One can think of a Gaussian process as defining a distribution over functions,
and inference taking place directly in the space of functions, the function-space two equivalent views
view.