---
layout: article
title: Few Shot Learning
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---



Few-Shot Learning
=================

Classification models that perform well after seeing very few samples from each class can be achieved in a number of ways:

-   A good weight initialization ([Finn et al., 2017](https://arxiv.org/abs/1703.03400)) so that subsequent learning coverges rapidly,
-   Weight initialization and an optimizer ([Ravi & Larochelle, 2016](https://openreview.net/forum?id=rJY0-Kcll)) that builds on weight update rules meta-learned by gradient descent ([Andrychowicz et al., 2016](https://arxiv.org/abs/1606.04474)),
-   Learning an embedding of examples that facilitates classification by a linear model ([Snell et al., 2017](https://arxiv.org/abs/1703.05175)) or k-nearest neighbor ([Vinyals et al., 2016](https://arxiv.org/abs/1606.04080)), and
-   Using external memory ([Santoro et al., 2016](https://arxiv.org/abs/1605.06065)) to rapidly bind information in the data.

For deep generative models, an autoencoder can be extended to learn statistics of datasets ([Edwards & Storkey, 2016](https://arxiv.org/abs/1606.02185)), rather than datapoints.

[Omniglot](https://github.com/brendenlake/omniglot) is a popular data set for evaluation.

References
----------

-   2017 July 18, Chelsea Finn, Pieter Abbeel, and Sergey Levine. [Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/abs/1703.03400). *arXiv:1703.03400*. [blog](http://bair.berkeley.edu/blog/2017/07/18/learning-to-learn/).
-   2017 March 15, Jake Snell, Kevin Swersky, and Richard S. Zemel. [Prototypical Networks for Few-shot Learning](https://arxiv.org/abs/1703.05175). *arXiv:1703.05175*.
-   2016 November 5, Sachin Ravi and Hugo Larochelle. [Optimization as a Model for Few-Shot Learning](https://openreview.net/forum?id=rJY0-Kcll). *OpenReview*.
-   2016 June 13, Oriol Vinyals, Charles Blundell, Timothy Lillicrap, Koray Kavukcuoglu, and Daan Wierstra. [Matching Networks for One Shot Learning](https://arxiv.org/abs/1606.04080). *arXiv:1606.04080*.
-   2016 June 7, Harrison Edwards and Amos Storkey. [Towards a Neural Statistician](https://arxiv.org/abs/1606.02185). *arXiv:1606.02185*.
-   2016 May 19, Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy Lillicrap. [One-shot Learning with Memory-Augmented Neural Networks](https://arxiv.org/abs/1605.06065). *arXiv:1605.06065*.
-   2015 December 11, Brenden M. Lake, Ruslan Salakhutdinov, Joshua B. Tenenbaum. [Human-level concept learning through probabilistic program induction](http://science.sciencemag.org/content/350/6266/1332). *Science*, 350(6266):1332-1338.
