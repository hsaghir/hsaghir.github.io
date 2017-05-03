---
layout: article
title: Approximate Inference - Ch19
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---


- A model usually consists of two sets of visible and hidden variables. Inference is the problem of computing $$p(h|v)$$ or taking an expectation on it. This is usually necessary for maximum likelihood learning. 

- In graphical models like RBM and pPCA, where there is only a single hidden layer, inference is easy. 

- Interactable inference (too computationally hard to calculate,  omits the gains of having a probabilistic graphical model) in DL usually arises from interaction beteween latent variables in a structured graphical model. It may be due to direct interactions in an undirected graph (produces large cliques of latent vars) or the "explaining away" effect in directed graphs. 

- Inference can be described as an optimization problem and many approaches bypass exact inference in favor of tractability of variational inference. 

- This conversion of expectation to optimization is done throught ELBO. 