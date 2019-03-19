---
layout: article
title: Correlation Explanation for embedding sparse Categorical variables
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

### Correlation explanation: 

- Essentially, CorEx is an unsupervised model-free latent factor setting trained with an information theoretic objective. The intuition is that if we can explain as much correlation in the data as possible, we can discover structure in the data. 
 
- The advantage of CorEX w.r.t. traditional generative modelling is the model-free aspect of CorEx where we don't need to first define a generative model and then figure out the inference. With model-based generative modelling there are two main problems. First we need to define a generative process (model) beforehand, and second the models we define are generally restricted to our ability to do inference in them. Model-free latent factor approaches like CorEx help alleviate this problem.

- CorEx searches for latent factors so that conditioned on these factors, the correlation in the data are minimized (as measured by multivariate mutual information). In other words, it looks for the simplest explanation that accounts for the most correlation in the data.As a bonus, building on the infomation based foundation leads to a hierarchical representations.

- If we have $$n$$ discrete random variabels $${X_1,...,X_n}$$, total correlation (multivariate mutual information) is defined as the sum of mutual informations between all random variables $$TC(X_G) =  \sum_i H(X_i) - H(X_G)$$. If the joint distribution $$P(X_1,..X_n)$$ factorizes the total correlation is zero therfore, total correlation can be expressed as the KL divergence between the real joint and the factorized joint $$TC(X_G) =  KL(p(X_G) || \prod_i p(x_i))$$. 

-  We introduce latent variables $$Y$$. The total correaltion among the observed group of variables, $$X$$, after condition on $$Y$$ is simply $$TC(X|Y) = \sum_i H(X_i|Y) - H(X|Y)$$. Therefore, the extent to which $$Y$$ explains the total correlation in $$X$$ can be measured by looking at how much the total correlation is reduced after introducing $$Y$$ i.e. the difference between the total correlation in $$X$$ and and total correlation in $$(X|Y)$$ i.e. $$TC(X;Y) = TC(X) - TC(X|Y) = \sum_i I(X_i : Y) - I(X : Y)$$. 

-  This difference forms an objective function that we can optimize to find the latent factors $$Y$$ that best explain the correlations in $$X$$. The bigger this objective, the more $$Y$$ explains correlation in $$X$$. Note that this objective is not symmetric. 

-  Also Note that if the distribution of X conditioned on Y is factorized (i.e. Y explains all correlation in X) the $$T(X|Y)$$ term becomes zero and the objective maximizes since $$TC(X)$$ does not depend on $$Y$$. This implies that the data $$X$$ can be perfectly described by a naive Bayes model with $$Y$$ as parent and $$X$$ as children if we knew $$Y$$. This intuition will help with solving the optimization problem in a tractable and efficient way. 

-  An interesting connection is that in the case where $$Y$$ explains all correlation in $$X$$, $$Y$$ will be explaining all Markov properties (pair-wise correlations) in $$X$$ and therfore, it will form a directed acyclic graphical model.  

- Since the total correlation depends on the joint distribution p(X,Y) and by extension on P(X). If we have $$n$$ binary $${0,1}$$ variables, then the search over all P(Y|X) involves $$2^n$$ variables which is intractable. 

- The paper introduces some restrictions on the objective to make the optimization tractable even for high-dim spaces and small number of samples.

- The output of this process is a set of Y variables that explain the correlations in $$X$$. If we repeat the same process for $$Y$$, we can find a set of $$Z$$ variables that explain the correlations in $$Y$$. This will give us a hierarchy of representations that form a tree generative model for the data.  

- From computational point of view, the main work of the algorithm involves a matrix multiplication followed by a nonlinear element-wise transform. These can be accelerated by GPUs!
- From a theoretical point of view, generalizing CorEx to allow non-tree representations seem feasible.







### MINE (mutual information neural estimation):
- mutual information between two random variables is the ratio density between their joint distribution and the product of their marginals. GANs can do this density ratio estimation


- In practice, MI between two sets of variables using MINE is calculated using a discriminator (regressor to be exact) with two heads for the two sets of variables. By maximizing the distance between regressed value for the original ordering of the two sets of variables and the estimated value for the shuffled ordering of the two sets of variable; the discriminator learns to estimate the mutual information between the two sets of variables. 

- It might be possible to use the [MINE](https://arxiv.org/abs/1801.04062) setting of MI estimation in corex. 