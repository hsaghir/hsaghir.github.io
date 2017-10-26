---
layout: article
title: Elements of statistical learning theory
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---



## AdaBoost

- A weak classifier is one whose error rate is only slightly better than random guessing. The purpose of boosting is to **sequentially** apply the weak classification algorithm to repeatedly **modified versions of the data** [observations?](RL analogy) thereby producing a sequence of weak classifiers $$G_m(x),m = 1, 2,... ,M$$ [Policy?](RL analogy). The predictions from all of them are then combined through a weighted average (vote) to produce the final prediction.

- The data modifications at each boosting step consist of applying weights w1,w2,... ,wN to each of the training observations (xi,yi), i = 1, 2,... ,N [envinronment Model is simply defined as data with different weights](RL analogy!). 
    + Initially all of the weights are set to $$w_i = \frac{1}{N}$$, so that the first step simply trains the classifier on the data in the usual manner. 
    + For each successive iteration m = 2, 3,... ,M: 
        * Fit a classifier $$G_m(x)$$ to the training data using weights $$w_i$$.
        * Compute the classifier's normalized weighted error.
        * Compute classifier's voting weight/ coefficient $$\alpha_m = \log((1 − err_m)/err_m)$$.
        * Compute new sample weights based on misclassified instances.  Set $$w_i \to w_i . exp[\alpha_m · mis-classified_i]$$
    + at the end take the weighted $$\alpha_m$$ average of votes between m classifiers $$G_m$$.

the observation weights are individually modified and the classification algorithm is reapplied to the weighted observations. At step $$m$$, those observations that were misclassified by the classifier $$G_{m−1}(x)$$ induced at the previous step have their weights increased, whereas the weights are decreased for those that were classified correctly. 





