---
layout: article
title: Math tricks in ML
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---


This is a post inspired by Shakir Mohammad's [Machine learning trick of day series](http://blog.shakirm.com/ml-series/trick-of-the-day/). I found it super-useful, so I decided to summarize the intuitions behind these mathematical tricks here:


1. Replica trick. A useful trick for the computation of log-normalising constants. With this trick we can provide theoretical insights into many of the models we find today, and predictions for the outcomes that we should see in experiments.

2. Gaussian Integral trick. A instance of a class of variable augmentation strategies that allows us to introduce auxiliary variables that allows for easier inference. This particular trick allows quadratic functions of discrete variables to be represented using a continuous underlying representation for which Monte Carlo analysis is easier.

3. Hutchinson's trick. Hutchinson's estimator allows us to compute a stochastic, unbiased estimator of the trace of a matrix. It forms one instance of a diverse set of randomised algorithms for matrix algebra that we can use to scale up our machine learning systems.

4. Reparameterisation tricks. We can often reparameterise random variables in our problems using a mechanism by which they are generated. This is especially useful in deriving unbiased gradient estimators that are used for stochastic optimisation problems that appear throughout machine learning.

5. Log derivative trick. An ability to flexibly manipulate probabilities is essential in machine learning. We can do this using the score function that then allows us to develop alternative gradient estimators for  the stochastic optimisation problems that we encountered using reparameterisation methods.

6. Tricks with sticks. The analogy of breaking a stick is a powerful tool that helps us to reason about how probability can be assigned to a set of discrete categories. Using this tool, we shall develop new sampling methods, loss functions for optimisation, and ways to specify highly-flexible models.

7. Kernel tricks 

8. Identity trick

9. Gumbel-max trick

10. log-sum-exp trick.

## Other Math tricks 

there are many mathematical “tricks” involved in Machine Learning, whether or not they are explicitly stated. The following inspired by this [list](https://danieltakeshi.github.io/2017/05/06/mathematical-tricks-commonly-used-in-machine-learning-and-statistics)

11. Cauchy-Schwarz
12. Integrating Probabilities into Expectations
13. Introducing an Independent Copy
14. Jensen’s Inequality
15. Law of Iterated Expectation
16. Lipschitz Functions
17. Markov’s Inequality
18. Norm Properties
19. Series Expansions (e.g. Taylor’s)
20. Stirling’s Approximation
21. Symmetrization
22. Take a Derivative
23. Union Bound
24. Variational Representations
25. the law of iterated expectation, i.e. $$E[E[X|Y]]=E[X]$$.




