---
layout: article
title: Similarity matrix, Distance measure, and divergence incarnations
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

# Clustering

In general, two approaches are used for clustering include 1) iteratively fitting a mixture model (e.g., using EM as in k-means) and 2) linking together pairs of training cases that have high affinity (e.g., using spectral methods).


# Distance measures and similarity matrices

- Measures of similarity between two vectors include Euclidean distance, 1-norm, ∞-norm, Cosine measure, Gabriel graph, A measure derived from a consensus matrix, Delaunay triangulation, Hamming distance or variation, a new measure you develop, etc. If a similarity measure doesn't return a value in [0,1] you can always squash them to [0,1] using a sigmoid function or a negative exponential. Conceptually, we like thinking of the data in a "graph-like" way. We can form a similarity matrix (using the distance measure) and use a powerful graph partitioning softwares to cluster the points. For example, spectral clustering algorithms depend on forming a Laplacian matrix based on similarity matrix, do eigen decomposition, and cluster based on eigenvalues.

- Affinity propagation employes a similarity matrix (uses Euclidean distance), performs belief propagation and clusters the data.

- Gaussian Process makes a similarity matrix for data (uses negative exponensial of Euclidean distance). Uses that as the Covariance matrix of a multivariate Gaussian (assuming each data point is a random variable). Since the joint distribution of data is a multivariate Gaussian, assuming that the joint distribution of test data and training data are also a multivariate Gaussian, prediction will be conditional probability of test data given training data from a multivariate Gaussian joint. In a multivariate Gaussian, joint and marginal probabilities can be analytically calculated using linear algebra on the covariance matrix. Therefore, prediction consists of simply performing linear algebra on covariance matrix (similarity matrix) of training data. Note that the choice of the distance measure (i.e. negative exponensial of Euclidean distance) is the modelling prior in a regression problem (e.g. if a linear distance is chosen, then it's linear regression!).

- Linear Gaussian models (LDS, HMM, etc) and Latent state space models Gaussian models (i.e. as in a SVAE), formulate models as factorizations of a multivariate Gaussian on hidden and observable variables. 


# Divergences 

- Approximate entropy employes a similarity matrix (uses Euclidean distance), sums it, normalizes it, and calculates the entropy of the resulting probability distribution (histogram). Does this for vectors of 2 points (embedding dimension=2) and vectors of 3 points (embedding dim=3) and calculates the conditional entropy as their ratio.

- Divergences are not similarity measures since they are not symmetric but they are close enough for many applications. 

- In my time-series clustering algorithm for holder exponents, I used a variation of KL-divergence (i.e. Chi-square measure) for clustering histograms of data from consecutive time windows.

- GANs and VAEs employ KL-divergence as a distance measure between probability distributions that can then be used for forming a loss function that they later try to optimize. WGAN uses a different divergence (Wassestien or earth mover's distance) which is a better divergence and results in significantly better and more stable results. 
