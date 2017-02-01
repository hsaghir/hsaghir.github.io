---
layout: article
title: Bayesian machine learning overview
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

Bayesian statistics is a branch of statistics where quantities of interest (such as parameters of a statistical model) are treated as random variables, and one draws conclusions by analyzing the posterior distribution over these quantities given the observed data. While the core ideas are decades or even centuries old, Bayesian ideas have had a big impact in machine learning in the past 20 years or so because of the flexibility they provide in building structured models of real world phenomena. Algorithmic advances and increasing computational resources have made it possible to fit rich, highly structured models which were previously considered intractable.

This roadmap is meant to give pointers to a lot of the key ideas in Bayesian machine learning. If you're considering applying Bayesian techniques to some problem, you should learn everything in the "core topics" section. Even if you just want to use a software package such as [BUGS](http://www.mrc-bsu.cam.ac.uk/bugs/), [Infer.NET](http://research.microsoft.com/en-us/um/cambridge/projects/infernet/), or [Stan](http://mc-stan.org/), this background will help you figure out the right questions to ask. Also, if the software doesn't immediately solve your problem, you'll need to have a rough mental model of the underlying algorithms in order to figure out why.

If you're considering doing research in Bayesian machine learning, the core topics and many of the advanced topics are part of the background you're assumed to have, and the papers won't necessarily provide citations. There's no need to go through everything here in linear order (the whole point of Metacademy is to prevent that!), but hopefully this roadmap will help you learn these things as you need them. If you wind up doing research in Bayesian machine learning, you'll probably wind up learning all of these topics at some point.

Core topics
===========

This section covers the core concepts in Bayesian machine learning. If you want to use this set of tools, I think it's worth learning everything in this section.

Central problems
----------------

What is Bayesian machine learning? Generally, Bayesian methods are trying to solve one of the following problems:

-   [parameter estimation](https://metacademy.org/concepts/bayesian_parameter_estimation). Suppose you have a statistical model of some domain, and you want to use it to make predictions. Or maybe you think the parameters of the model are meaningful, and you want to fit them in order to learn something about the world. The Bayesian approach is to compute or approximate the posterior distribution over the parameters given the observed data.
    -   Often, you want to use the model to choose actions. [Bayesian decision theory](https://metacademy.org/concepts/bayesian_decision_theory) is a framework for doing this.
-   [model comparison](https://metacademy.org/concepts/bayesian_model_comparison): You may have several different models under consideration, and you want to know which is the best match to your data. A common case is that you have several models of the same form of differing complexities, and you want to trade off the complexity with the degree of fit.
    -   Rather than choosing a single model, you can define a prior over the models themselves, and average the predictions with respect to the posterior over models. This is known as [Bayesian model averaging](https://metacademy.org/concepts/bayesian_model_averaging).

It's also worth learning the basics of [Bayesian networks](https://metacademy.org/concepts/bayesian_networks) (Bayes nets), since the notation is used frequently when talking about Bayesian models. Also, because Bayesian methods treat the model parameters as random variables, we can represent the Bayesian inference problems themselves as Bayes nets!

The readings for this section will tell you enough to understand *what problems Bayesian methods are meant to address*, but won't tell you how to actually solve them in general. That is what the rest of this roadmap is for.

Non-Bayesian techniques
-----------------------

As background, it's useful to understand how to fit generative models in a non-Bayesian way. One reason is that these techniques can be considerably simpler to implement, and often they're good enough for your goals. Also, the Bayesian techniques bear close similarities to these, so they're often helpful analogues for reasoning about Bayesian techniques.

Most basically, you should understand the notion of [generalization](https://metacademy.org/concepts/generalization), or how well a machine learning algorithm performs on data it hasn't seen before. This is fundamental to evaluating any sort of machine learning algorithm. You should also understand the following techniques:

-   [maximum likelihood](https://metacademy.org/concepts/maximum_likelihood), a criterion for fitting the parameters of a generative model
-   [regularization](https://metacademy.org/concepts/regularization), a method for preventing overfitting
-   [the EM algorithm](https://metacademy.org/concepts/expectation_maximization), an algorithm for fitting generative models where each data point has associated latent (or unobserved) variables

Basic inference algorithms
--------------------------

In general, Bayesian inference requires answering questions about the posterior distribution over a model's parameters (and possibly latent variables) given the observed data. For some simple models, these questions can be answered analytically. However, most of the time, there is no analytic solution, and we need to compute the answers approximately.

If you need to implement your own Bayesian inference algorithm, the following are probably the simplest options:

-   [MAP estimation](https://metacademy.org/concepts/map_parameter_estimation), where you approximate the posterior with a point estimate on the optimal parameters. This replaces an integration problem with an optimization problem. This doesn't mean the problem is easy, since the optimization problem is often itself intractable. However, it often simplifies things, because software packages for optimization tend to be more general and robust than software packages for sampling.
-   [Gibbs sampling](https://metacademy.org/concepts/gibbs_sampling), an iterative procedure where each random variable is sampled from its conditional distribution given the remaining ones. The result is (hopefully) an approximate sample from the posterior distribution.

You should also understand the following general classes of techniques, which include the majority of the Bayesian inference algorithms used in practice. Their general formulations are too generic to be relied on most of the time, but there are a lot of special cases which are very powerful:

-   [Markov chain Monte Carlo](https://metacademy.org/concepts/markov_chain_monte_carlo), a general class of sampling-based algorithms based on running Markov chains over the parameters whose stationary distribution is the posterior distribution.
    -   In particular, [Metropolis-Hastings](https://metacademy.org/concepts/metropolis_hastings) (M-H) is a recipe for constructing valid MCMC chains. Most practical MCMC algorithms, including Gibbs sampling, are special cases of M-H.
-   [Variational inference](https://metacademy.org/concepts/variational_inference), a class of techniques which try to approximate the intractable posterior distribution with a tractable distribution. Generally, the parameters of the tractable approximation are chosen to minimize some measure of its distance from the true posterior.

Models
------

The following are some simple examples of generative models to which Bayesian techniques are often applied.

-   [mixture of Gaussians](https://metacademy.org/concepts/mixture_of_gaussians), a model where each data point belongs to one of several "clusters," or groups, and the data points within each cluster are Gaussian distributed. Fitting this model often lets you infer a meaningful grouping of the data points.
-   [factor analysis](https://metacademy.org/concepts/factor_analysis), a model where each data point is approximated as a linear function of a lower dimensional representation. The idea is that each dimension of the latent space corresponds to a meaningful factor, or dimension of variation, in the data.
-   [hidden Markov models](https://metacademy.org/concepts/hidden_markov_models), a model for time series data, where there is a latent discrete state which evolves over time.

While Bayesian techniques are most closely associated with generative models, it's also possible to apply them in a discriminative setting, where we try to directly model the conditional distribution of the targets given the observations. The canonical example of this is [Bayesian linear regression](https://metacademy.org/concepts/bayesian_linear_regression).

Bayesian model comparison
-------------------------

The section on inference algorithms gave you tools for approximating posterior inference. What about model comparison? Unfortunately, most of the algorithms are fairly involved, and you probably don't want to implement them yourself until you're comfortable with the advanced inference algorithms described below. However, there are two fairly crude approximations which are simple to implement:

-   the [Bayesian information criterion (BIC)](https://metacademy.org/concepts/bayesian_information_criterion), which simply takes the value of the MAP solution and adds a penalty proportional to the number of parameters
-   the [Laplace approximation](https://metacademy.org/concepts/laplace_approximation), which approximates the posterior distribution with a Gaussian centered around the MAP estimate.

Advanced topics
===============

This section covers more advanced topics in Bayesian machine learning. You can learn about the topics here in any order.

Models
------

The "core topics" section listed a few commonly used generative models. Most datasets don't fit those structures exactly, however. The power of Bayesian modeling comes from the flexibility it provides to build models for many different kinds of data. Here are some more models, in no particular order.

-   [logistic regression](https://metacademy.org/concepts/bayesian_logistic_regression), a discriminative model for predicting binary targets given input features
-   [Bayesian networks](https://metacademy.org/concepts/bayesian_networks) (Bayes nets). Roughly speaking, Bayes nets are directed graphs which encode patterns of probabilistic dependencies between different random variables, and are typically chosen to represent the causal relationships between the variables. While Bayes nets can be learned in a non-Bayesian way, Bayesian techniques can be used to learn both the [parameters](https://metacademy.org/concepts/bayes_net_parameter_learning) and [structure](https://metacademy.org/concepts/bayes_net_structure_learning) (the set of edges) of the network.
    -   [Linear-Gaussian models](https://metacademy.org/concepts/linear_gaussian_models) are an important special case where the variables of the network are all jointly Gaussian. Inference in these networks is often tractable even in cases where it's intractable for discrete networks with the same structure.
-   [latent Dirichlet allocation](https://metacademy.org/concepts/latent_dirichlet_allocation), a "topic model," where a set of documents (e.g. web pages) are each assumed to be composed of some number of topics, such as computers or sports. Related models include [nonnegative matrix factorization](https://metacademy.org/concepts/nonnegative_matrix_factorization) and [probabilistic latent semantic analysis](https://metacademy.org/concepts/probabilistic_lsa).
-   [linear dynamical systems](https://metacademy.org/concepts/linear_dynamical_systems), a time series model where a low-dimensional gaussian latent state evolves over time, and the observations are noisy linear functions of the latent states. This can be thought of as a continuous version of the HMM. Inference in this model can be performed exactly using the Kalman [filter](https://metacademy.org/concepts/kalman_filter) and [smoother](https://metacademy.org/concepts/kalman_smoother).
-   [sparse coding](https://metacademy.org/concepts/sparse_coding), a model where each data point is modeled as a linear combination of a small number of elements drawn from a larger dictionary. When applied to natural image patches, the learned dictionary resembles the receptive fields of neurons in the primary visual cortex. See also a closely related model called [independent component analysis](https://metacademy.org/concepts/independent_component_analysis).

Bayesian nonparametrics
-----------------------

All of the models described above are *parametric*, in that they are represented in terms of a fixed, finite number of parameters. This is problematic, since it means one needs to choose a parameter for, e.g., the number of clusters, and this is rarely known in advance.

This problem may not seem so bad for the models described above, because for simple models such as clustering, one can typically choose good parameters using cross-validation. However, many widely used models are far more complex, involving many independent clustering problems, where the numbers of clusters can vary from a handful to thousands.

Bayesian nonparametrics is an ongoing research area within machine learning and statistics which sidesteps this problem by defining models which are *infinitely complex*. We cannot explicitly represent infinite objects in their entirety, of course, but the key insight is that for a finite dataset, we can still perform posterior inference in the models while only explicitly representing a finite portion of them.

Here are some of the most important building blocks which are used to construct Bayesian nonparametric models:

-   [Gaussian processes](https://metacademy.org/concepts/gaussian_processes) are priors over functions such that the values sampled at any finite set of points are jointly Gaussian. In many cases, posterior inference is tractable. This is probably the default thing to use if you want to put a prior over functions.
-   the [Chinese restaurant process](https://metacademy.org/concepts/chinese_restaurant_process), which is a prior over partitions of an infinite set of objects.
    -   This is most commonly [used in clustering models](https://metacademy.org/concepts/crp_clustering) when one doesn't want to specify the number of components in advance. The inference algorithms are fairly simple and well understood, so there's no reason not to use a CRP model in place of a finite clustering model.
    -   This process can equivalently be viewed as [Dirichlet process](https://metacademy.org/concepts/dirichlet_process).
-   the [hierarchical Dirichlet process](https://metacademy.org/concepts/hierarchical_dirichlet_process), which involves a set of Dirichlet processes which share the same base measure, and the base measure is itself drawn from a Dirichlet process.
-   the [Indian buffet process](https://metacademy.org/concepts/indian_buffet_process), a prior over infinite binary matrices such that each row of the matrix has only a finite number of 1's. This is most commonly used in models where each object can have various attributes. I.e., rows of the matrix correspond to objects, columns correspond to attributes, and an entry is 1 if the object has the attribute.
    -   The simplest example is probably the [IBP linear-Gaussian model](https://metacademy.org/concepts/ibp_linear_gaussian_model), where the observed data are linear functions of the attributes.
    -   The IBP can also be viewed in terms of the [beta process](https://metacademy.org/concepts/beta_process). Essentially, the beta process is to the IBP as the Dirichlet process is to the CRP.
-   [Dirichlet diffusion trees](https://metacademy.org/concepts/dirichlet_diffusion_trees), a hierarchical clustering model, where the data points cluster at different levels of granularity. I.e., there may be a few coarse-grained clusters, but these themselves might decompose into more fine-grained clusters.
-   the [Pitman-Yor process](https://metacademy.org/concepts/pitman_yor_process), which is like the CRP, but has a more heavy-tailed distribution (in particular, a power law) over cluster sizes. I.e., you'd expect to find a few very large clusters, and a large number of smaller clusters. Power law distributions are a better fit to many real-world datasets than the exponential distributions favored by the CRP.

Sampling algorithms
-------------------

From the "core topics" section, you've already learned two examples of sampling algorithms: [Gibbs sampling](https://metacademy.org/concepts/gibbs_sampling) and [Metropolis-Hastings](https://metacademy.org/concepts/metropolis_hastings) (M-H). Gibbs sampling covers a lot of the simple situations, but there are a lot of models for which you can't even compute the updates. Even for models where it is applicable, it can mix very slowly if different variables are tightly coupled. M-H is more general, but the general formulation provides little guidance about how to choose the proposals, and the proposals often need to be chosen very carefully to achieve good mixing.

Here are some more advanced MCMC algorithms which often perform much better in particular situations:

-   [collapsed Gibbs sampling](https://metacademy.org/concepts/collapsed_gibbs_sampling), where a subset of the variables are marginalized (or collapsed) out analytically, and Gibbs sampling is performed over the remaining variables. For instance, when fitting a CRP clustering model, we often marginalize out the cluster parameters and perform Gibbs sampling over the cluster assignments. This can dramatically improve the mixing, since the assignments and cluster parameters are tightly coupled.
-   [Hamiltonian Monte Carlo](https://metacademy.org/concepts/hamiltonian_monte_carlo) (HMC), an instance of M-H for continuous spaces which uses the gradient of the log probability to choose promising directions to explore. This is the algorithm that powers [Stan](http://mc-stan.org/).
-   [slice sampling](https://metacademy.org/concepts/slice_sampling), an auxiliary variable method for sampling from one-dimensional distributions. Its key selling point is that the algorithm doesn't require specifying any parameters. Because of this, it is often combined with other algorithms such as HMC which would otherwise require specifying step size parameters.
-   [reversible jump MCMC](https://metacademy.org/concepts/reversible_jump_mcmc), a way of constructing M-H proposals between spaces of differing dimensionality. The most common use case is Bayesian model averaging.

While the majority of sampling algorithms used in practice are MCMC algorithms, sequential Monte Carlo (SMC) is another class of techniques based on approximately sampling from a sequence of related distributions.

-   The most common example is probably the [particle filter](https://metacademy.org/concepts/particle_filter), an inference algorithm typically applied to time series models. It accounts for observations one time step at a time, and at each step, the posterior over the latent state is represented with a set of particles.
-   [Annealed importance sampling](https://metacademy.org/concepts/annealed_importance_sampling) (AIS) is another SMC method which gradually "anneals" from an easy initial distribution (such as the prior) to an intractable target distribution (such as the posterior) by passing through a sequence of intermediate distributions. An MCMC transition is performed with respect to each of the intermediate distributions. Since mixing is generally faster near the initial distribution, this is supposed to help the sampler avoid getting stuck in local modes.
    -   The algorithm computes a set of weights which can also be used to [estimate the marginal likelihood](https://metacademy.org/concepts/ais_partition_function). If enough intermediate distributions are used, the variance of the weights is small, and therefore they yield an accurate estimate of the marginal likelihood.

Variational inference
---------------------

Variational inference is another class of approximate inference techniques based on optimization rather than sampling. The idea is to approximate the intractable posterior distribution with a tractable approximation. The parameters of the approximate distribution are chosen to minimize some measure of distance (usually [KL divergence](https://metacademy.org/concepts/kl_divergence)) between the approximation and the posterior.

It's hard to make any general statements about the tradeoffs between variational inference and sampling, because each of these is a broad category that includes many particular algorithms, both simple and sophisticated. However, here are some general rules of thumb:

-   Variational inference algorithms involve different implementation challenges from sampling algorithms:
    -   They are harder, in that they may require lengthy mathematical derivations to determine the update rules.
    -   However, once implemented, variational Bayes can be easier to test, because one can employ the standard checks for optimization code (gradient checking, local optimum tests, etc.)
    -   Also, most variational inference algorithms converge to (local) optima, which eliminates the need to check convergence diagnostics.
-   The output of most variational inference algorithms is a distribution, rather than samples.
    -   To answer many queries, such as the expectation or variance of a model parameter, one can simply check the variational distribution. With sampling methods, by contrast, one often needs to collect large numbers of samples, which can be expensive.
    -   However, with variational methods, the accuracy of the approximation is limited by the expressiveness of the approximating class, and it's not always obvious how different the approximating distribution is from the posterior. By contrast, if you run a sampling algorithm long enough, eventually you will get accurate results.

Here are some important examples of variational inference algorithms:

-   [variational Bayes](https://metacademy.org/concepts/variational_bayes), the application of variational inference to Bayesian models where the posterior distribution over parameters cannot be represented exactly. If the model also includes latent variables, then [variational Bayes EM](https://metacademy.org/concepts/variational_bayes_em) can be used.
-   the [mean field approximation](https://metacademy.org/concepts/mean_field_approximation), where the approximating distribution has a particularly simple form: all of the variables are assumed to be independent.
    -   Mean field can also be [viewed in terms of convex duality](https://metacademy.org/concepts/variational_inference_convex_duality), which leads to different generalizations from the usual interpretation.
-   [expectation propagation](https://metacademy.org/concepts/expectation_propagation), an approximation to loopy belief propagation. It sends approximate messages which represent only the expectations of certain sufficient statistics of the relevant variables.

And here are some canonical examples where variational inference techniques are applied. While you're unlikely to use these particular models, they provide a guide for how variational techniques can be applied to Bayesian models more generally:

-   [linear regression](https://metacademy.org/concepts/variational_linear_regression)
-   [logistic regression](https://metacademy.org/concepts/variational_logistic_regression)
-   [mixture of Gaussians](https://metacademy.org/concepts/variational_mixture_of_gaussians)
-   [exponential family models](https://metacademy.org/concepts/variational_exponential_family)

Belief propagation
------------------

Belief propagation is another family of inference algorithms intended for graphical models such as [Bayes nets](https://metacademy.org/concepts/bayesian_networks) and [Markov random fields](https://metacademy.org/concepts/markov_random_fields) (MRFs). The variables in the model "pass messages" to each other which summarize information about the joint distribution over other variables. There are two general forms of belief propagation:

-   When applied to tree-structured graphical models, BP performs exact posterior inference. There are two particular forms:
    -   the [sum-product algorithm](https://metacademy.org/concepts/sum_product_on_trees), which computes the marginal distribution of each individual variable (and also over all pairs of neighboring variables).
    -   the [max-product algorithm](https://metacademy.org/concepts/max_product_on_trees), which computes the most likely joint assignment to all of the variables
-   It's also possible to apply the same message passing rules in a graph which isn't tree-structured. This doesn't give exact results, and in fact lacks even basic guarantees such as convergence to a fixed point, but often it works pretty well in practice. This is often called [loopy belief propagation](https://metacademy.org/concepts/loopy_belief_propagation) to distinguish it from the tree-structured versions, but confusingly, some research communities simply refer to this as "belief propagation."
    -   Loopy BP can be [interpreted as a variational inference algorithm](https://metacademy.org/concepts/loopy_bp_as_variational).

The [junction tree algorithm](https://metacademy.org/concepts/junction_trees) gives a way of applying exact BP to non-tree-structured graphs by defining coarser-grained "super-variables" with respect to which the graph is tree-structured.

The most common special case of BP on trees is the [forward-backward algorithm](https://metacademy.org/concepts/forward_backward_algorithm) for HMMs. [Kalman smoothing](https://metacademy.org/concepts/kalman_as_forward_backward) is also a special case of the forward-backward algorithm, and therefore of BP as well.

BP is widely used in computer vision and information theory, where the inference problems tend to have a regular structure. In Bayesian machine learning, BP isn't used very often on its own, but it can be a powerful component in the context of a variational or sampling-based algorithm.

Theory
------

Finally, here are some theoretical issues involved in Bayesian methods.

-   Defining a Bayesian model requires choosing priors for the parameters. If we don't have strong prior beliefs about the parameters, we may want to choose [uninformative priors](https://metacademy.org/concepts/uninformative_priors). One common choice is the [Jeffreys prior](https://metacademy.org/concepts/jeffreys_prior).
-   How much data do you need to accurately estimate the parameters of your model? The [asymptotics of maximum likelihood](https://metacademy.org/concepts/asymptotics_of_maximum_likelihood) provide a lot of insight into this question, since for finite models, the posterior distribution has similar asymptotic behavior to the distribution of maximum likelihood estimates.


Connections of NNs to other machine learning methods

Many neural net models can be seen as nonlinear generalizations of "shallow" models. Feed-forward neural nets are essentially nonlinear analogues of algorithms like logistic regression. Autoencoders can be seen as nonlinear analogues of dimensionality reduction algorithms like PCA.
RBMs with all Gaussian units are equivalent to Factor analysis. RBMs can also be generalized to other exponential family distributions.
Kernel methods are another set of techniques for converting linear algorithms into nonlinear ones. There is actually a surprising relationship between neural nets and kernels: Bayesian neural nets converge to Gaussian processes (a kernelized regression model) in the limit of infinitely many hidden units. (See Chapter 2 of Radford Neal's Ph.D. thesis. Background: Gaussian processes)