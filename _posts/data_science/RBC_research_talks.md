---
layout: article
title: RBC research talks
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---


## Chris
- graphical models/message passing/combinatorial optimization
- main problem in inference is a marginalization problem
- message passing is basically the distributive law $$ab+ac=a(b+c)$$
- the key idea of message passing and belief prop is calculating things once and reuse them, i.e. dynamic programmig
- belief prop is exact marginalization-> if graph is a tree 
- if graph has loops -> approximate solution / initialize messages and iterate until convergecs.

## Kry
- dim-reduction -> pca/mds/isomap/ile/tsne
- an observation:
  + recall in continuous space is similar to continuity of function
  + precision in continuous space is similar to one-to-one mapping
- a corrolary from topology says we can't preserve both continuity/one-to-one when we perform a dimensionality reduction mapping.