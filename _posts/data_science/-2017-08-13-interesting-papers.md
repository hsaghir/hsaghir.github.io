---
layout: article
title:  Interesting papers main ideas
comments: true
categories: data_science
image:
  teaser: practical/pytorch_logo.png
---


- [Understanding Black-box Predictions via Influence Functions] (http://proceedings.mlr.press/v70/valera17a.html)
  + goal is to understand the effect of training points on a model’s predictions. how would the model’s predictions change if this training point wasn't present?
  + Assume calculating parameters by loss minimization with and without a data point. The difference between values of parameters shows the importance of the point. This is computationally prohibitive to do for every data point
  + Influence function provides an alternative by computing the parameter change if the data point z were perturbed by some small $$\epsilon$$, giving us new parameters. Assuming convexity and twice differentiability of the loss function, it is proven that the influence of the upweighting of the datapoint z on the parameters is given by Hessian gradient product of the loss function. Essentially this is determined by forming a quadratic approximation to the loss function around parameter values and taking a single Newton step.
  +  Since removing a point z is the same as upweighting it by 1/n, we can linearly approximate the parameter change due to removing z by computing 1/n * influence function, without retraining the model.
  +  Next, we apply the chain rule to measure how upweighting z changes functions of parameters.

-[Automatic Discovery of the Statistical Types of Variables in a Dataset](http://proceedings.mlr.press/v70/valera17a/valera17a.pdf)
  + Goal is to automatically discover the statistical types (i.e. numerical, categorical, etc) and appropriate likelihood (noise) models for, the variables in a dataset. 
  + Often, the variable types and noise model are assumed to be known. For example, a common approach is to model continuous data as Gaussian variables, and discrete data as categorical variables.
  + method automatically distinguishes among real-valued, positive real-valued and interval data as types of continuous variables, and among categorical, ordinal and count data as types of discrete variables.
  + probabilistic model assumptions:
    * There is a latent structure in the data and can be captured by a low-rank representation, such that conditioning on it, the likelihood model factorizes for both number of objects and attributes.
    * noise model for each attribute can be expressed as a mixture of noise models for all existing data types in the data. The weight is the probability of the attribute belonging to the corresponding data type.
  +  an efficient MCMC inference algorithm is used to infer both the low-rank representation and the weight of each noise model for each attribute in data.
  +  model:
    *  There are n observations and each observation is d-dimensional (an nxd matrix). There is a local vector of K latent variables, $$z_nk$$, (i.e. an nxk matrix) and a global weight vector, $$b_dk$$ (i.e. an dxk matrix).
    *  a matrix factorization problem where $$X_nd b_dk = Z_nk$$. The d attributes are assumed to be conditionally independent given the low rank representation $$Z_nk$$.
    *  we assume each attribute is a mixture of a set of distributions for possible data types. The weights, w, determine the mixture coefficient and is desired. 
    *  ![alt text](/images/interesting_papers/Valera17_data_type_inference.png "model")

##  [Direct feedback alignment]()
- The idea is to use random matrices instead of inverse weight matrices in backprop. Interestingly it works. Reminds me of reservoir computing ideas. 




