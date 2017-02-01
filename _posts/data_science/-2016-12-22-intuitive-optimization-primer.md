---
layout: article
title: Probabilistic graphical models and bayesian inference
comments: true
image:
  teaser: jupyter-main-logo.svg
---

There are a few classes of optimization algorithms that are frequently used in machine learning. 

- Gradient-based:
  1st order - (SGD, Adagrad, Adam,  Batch Gradient Descent)
  2nd order - (either by computing the Hessian or approximating it e.g. Newton method, conjugate gradient, scaled conjugate gradient, HFO)

- Inference Algorithms:
  Expectation Maximization (EM)
  Message passing

Search based techniques:
genetic algorithms, simulated annealing, Morkov Chain Monte Carlo?l 
These techniques usually don't require the function being optimised to be differentiable, they try to find a solution by sampling from a probability distribution.






You've defined your neural net architecture. How the heck do you train it? The basic workhorse for neural net training is [stochastic gradient descent (SGD)](https://metacademy.org/concepts/stochastic_gradient_descent), where one visits a single training example at a time (or a "minibatch" of training examples), and takes a small step to reduce the loss on those examples. This requires computing the [gradient](https://metacademy.org/concepts/gradient) of the loss function, which can be done using [backpropagation](https://metacademy.org/concepts/backpropagation). Be sure to [check your gradient computations](http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization) with finite differences to make sure you've derived them correctly. SGD is conceptually simple and easy to implement, and with a bit of tuning, can work very well in practice.

There is a broad class of optimization problems known as [convex optimization](https://metacademy.org/concepts/convex_optimization), where SGD and other local search algorithms are guaranteed to find the global optimum. This occurs because the function being optimized is "bowl shaped" (convex) and local improvements in the optimization function work towards the global optimum. Much of machine learning research is focused on trying to formulate things as convex optimization problems. Unfortunately, deep neural net training is usually not convex, so you are only guaranteed to find a local optimum. This is a bit disappointing, but ultimately it's [something we can live with](http://videolectures.net/eml07_lecun_wia/). For most feed-forward networks and generative networks, the local optima tend to be pretty reasonable. (Recurrent neural nets are a different story --- more on that below.)

A bigger problem than local optima is that the curvature of the loss function can be pretty extreme. While neural net training isn't convex, the problem of curvature also shows up for convex problems, and many of the techniques for dealing with it are borrowed from convex optimization. As general background, it's useful to read the following sections of Boyd and Vandenberghe's book, [Convex Optimization](http://www.stanford.edu/~boyd/cvxbook/):

-   [Sections 9.2-9.3](http://www.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=477) talk about gradient descent, the canonical first-order optimization method (i.e. a method which only uses first derivatives)
-   [Section 9.5](http://www.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=498) talks about Newton's method, the canonical second-order optimization method (i.e. a method which accounts for second derivatives, or curvature)

While Newton's method is very good at dealing with curvature, it is impractical for large-scale neural net training for two reasons. First, it is a batch method, so it requires visiting every training example in order to make a single step. Second, it requires constructing and inverting the Hessian matrix, whose dimension is the number of parameters. ([Matrix inversion](https://metacademy.org/concepts/computing_matrix_inverses) is only practical up to tens of thousands of parameters, whereas neural nets typically have millions.) Still, it serves as an idealized second-order training method which one can try to approximate. Practical algorithms for doing so include:

-   [conjugate gradient](http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)
-   limited memory BFGS

Compared with most neural net models, training RBMs introduces another complication: computing the objective function requires computing the partition function, and computing the gradient requires performing [inference](https://metacademy.org/concepts/inference_in_mrfs). Both of these problems are [intractable](https://metacademy.org/concepts/complexity_of_inference). (This is true for [learning Markov random fields (MRFs)](https://metacademy.org/concepts/mrf_parameter_learning) more generally.) [Contrastive divergence](http://learning.cs.toronto.edu/~hinton/csc2535/readings/nccd.pdf)and [persistent contrastive divergence](http://www.cs.utoronto.ca/~tijmen/pcd/pcd.pdf) are widely used approximations to the gradient which often work quite well in practice. Evaluating the models remains a difficult problem, though. One can [estimate the model likelihood](http://www.cs.utoronto.ca/~rsalakhu/papers/dbn_ais.pdf) using [annealed importance sampling](https://metacademy.org/concepts/annealed_importance_sampling), but this is delicate, and failures in estimation tend to overstate the model's performance.

Even once you understand the math behind these algorithms, the devil's in the details. Here are some good practical guides for getting these algorithms to work in practice:

-   G. Hinton. [A practical guide to training restricted Boltzmann machines.](http://www.csri.utoronto.ca/~hinton/absps/guideTR.pdf) 2010.
-   J. Martens and I. Sutskever. [Training deep and recurrent networks with Hessian-free optimization.](http://www.cs.utoronto.ca/~ilya/pubs/2012/HF_for_dnns_and_rnns.pdf) Neural Networks: Tricks of the Trade, 2012.
-   Y. Bengio. [Practical recommendations for gradient-based training of deep architectures.](http://arxiv.org/pdf/1206.5533) Neural Networks: Tricks of the Trade, 2012.
-   L. Bottou. [Stochastic gradient descent tricks.](http://research.microsoft.com/pubs/192769/tricks-2012.pdf) Neural Networks: Tricks of the Trade, 2012.