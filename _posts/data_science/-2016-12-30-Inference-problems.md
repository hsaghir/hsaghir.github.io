---
layout: article
title: An intuitive primer on Inference
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

Finding insights in data, prediction, modeling the world and everything that we can do with data involves finding the probability distribution underlying our data P(x). Therfore, We are interested in finding the true probability distribution of our data P(x) which is unknown. We usually use the scientific method of probabilistic pipline to solve this problem i.e.:

1. Knowledge and questions we want answered
2. make assumptions and formulate a model (using probabilitic graphical models, deep nets as functions, etc)
3. Fit the model with data, find the parameters of the model, find patterns in data (Inference)
4. predict and explore
5. criticize and revise the model


The statistical modelling procedure involves introduction of some hidden variables, z, and a mixing procedure for them in a way which we believe will lead to an estimation for the true probability distribution of the data P(x). These hidden variables and their structure construct our model. Therefore, the new problem under our model is the joint distribution P(x,z) of the hidden variables z, and the observed variables x. 

The joint distribution P(x,z) can be thought of as a combination of other simpler probability distributions P(x,z)=P(x|z)P(z). The way they are combined is through a hierarchy of component distributions. So first a sample from the top distribution over hidden variables P(z) (i.e. prior) chooses the component that should produce data, then the corresponding component P(x|z) (i.e. likelihood) produces a sample x. This makes it easier to express the complex probability distribution of the observed data P(x) using a model P(x,z)=P(x|z)P(z). It is important to note that the real procedure for producing a sample x is unknown to us and the model is merely an attempt to find an estimation for the true distribution.

At the beginning, the probability of choosing a component in the mixture is based on a very crude assumption for the general shape of such a distribution (i.e. prior P(z)) and we don't know the specific value of the parameters in the assumed structure for the probability model. We would like to find these unknowns based on the data. Finding the parameters to form a posterior belief is called inference and is the key algorithmic problem which tells us what model says about the data. In the probabilistic pipleline, we will then criticize the model we have built and might modify it and solve the problem again until we arrive at a model that satisfies our needs. 

Inference about the unknowns is done by finding the posterior P(z|x) through the Bayesian rule $$P(z|x)=\frac{P(x,z)}{P(x)}$$. To calculate the posterior we need $$P(x)$$ which is the probability that the model assings to the data. It is calculated by summing out all possible configurations of the hidden variables using the model $$P(x)= \int_x P(x,z)$$. For most interesting problems this integral is intractable so the exact inference problem (calculating the posterior) faces a problem. To solve the inference problem two approximation approaches are taken:

1. Approximate the integral $$P(x)= \int_x P(x,z)$$ using numerical methods like Quadrature(doesn't work in high dimensions) or MCMC. 
2. Bypass the calculation of the integral involved in calculation of exact posterior by approximating the posterior directly. We pick a family of distributions over the latent variables with its own parameters and maximize the ELBO to fit the family as closely as possible to the real posterior. We use this approximate posterior as a proxy for the real one.

Here is a list of strategies for solving the inference problem:

1. Exact Inference (For very simple problems where normalizer integral is tractable)
2. Evidence estimation (Estimating P(x) instead of its analytical form)
3. Density ratio estimation (Avoiding density estimation by calculating ratio of two densities)
4. Gradient ratio estimation (Instead of estimating ratios, we estimate gradients of log densities)

Other types of inference problems we might encounter are 5) Moment computation $E[f(z)|x] =\int f(z)p(z|x)dz$, 6) Prediction $p(xt+1) =\int p(xt+1|xt)p(xt)dxt$, and 7) Hypothesis Testing $B = log p(x|H1) - log p(x|H2)$. These are usually solved using the same strategies layed out above.

Before getting into the nuts and bolts of solving the inference problem, it is useful to have an intuition of the problem. When we setup a Bayesian inference problem with N unknowns, we are implicitly creating an N dimensional space of parameters for the prior distributions to exist in. Associated is an additional dimension, that reflects the prior probability of a particular point in the N dimensional space. Surfaces describe our prior distributions on the unknowns, and after incorporating our observed data X, the surface of the space changes by pulling and stretching the fabric of the prior surface to reflect where the true parameters likely live. The tendency of the observed data to push up the posterior probability in certain areas is checked by the prior probability distribution, so that lower prior probability means more resistance. More data means more pulling and stretching, and our original shape becomes mangled or insignificant compared to the newly formed shape. Less data, and our original shape is more present. Regardless, the resulting surface describes the posterior distribution.


## Inference Strategies:

### Exact inference
Exact inference is not possible unless the priors are conjugate so that the posteriors are also conjugate and they can mathematically be derived exactly. Using the mean field assumption with conjugate priors can help with exact inference. The mean-field assumption is a fully factorized posterior. Each factor is the same family as the model’s complete conditional. if the prior is conjugate, probability propagation algorithms can be used, but in case of non-conjugate priors black box variational inference should be used.

#### Probability propagation and factor graphs:
In undirected graph case, there is one and only one path between each pair of nodes. In a directed trees, there is one node that hass no parent, root, and all other nodes have exactly one parent. 
Here we introduce an algorithm for probabilistic inference known as the sum-product/belief-propagation applicable to tree-like graphs. This algorithm involves asimoke  update equation, i.e. a sum over a product of potentials, which is applied once for each outgoing edge at each node

#### Laplace approximation is an approximation of the posterior using simple functions.


### Evidence estimation

For most interesting models, the denominator of posterior is not tractable since it involves integrating out any global and local variables, z so the integral is intractable and requires approximation. 

$p(x) = \int p(x, z)dz$

Being able to estimate the evidence enables model scoring/ comparison/ selection, moment estimation, normalisation, posterior computation, and prediction. 

Traditional solution to approximating the evidence and the posterior involves resampling techniques including Gibbs sampling which is a variant of MCMC based on sampling from the posterior. Gibbs sampling is based on randomness and sampling, has strict conjugacy requirements, require many iterations and sampling, and it's not very easy to gauge whether the Markov Chain has converged to start collecting samples from the posterior. There are other MCMC algorithms (like Metropolis Hastings) that do not require conjugacy but still suffer from the need for thousands of samplings and convergence measurement deficiency.

#### Variational Inference

Variatonal inference turns the problem of inference into optimization. It started in 80s, by bringing ideas like mean field from statistical physics to probabilistic methods. In the 90s, Jordan et al developed it further to generalize to more inference problems and in parallel Hinton et al developed mean field for neural nets and connected it to EM which lead to VI for HMMs and Mixtures. Modern VI touches many areas like, probabilistic programming, neural nets, reinforcement learning, convex optimization, bayesian statistics and many applications. 

Basic idea is to transform the integral into an expectation over a simple, known distribution using the variational approximate (ELBO). Writing the log-likelihood for the evidence $log p(x)=log \int p(x,z)$ and introducing the variational approximate $log p(x)= log \int p(x,z)*q(z|x)/q(z|x)$, we can move the $log$ inside the integral using Jensen's inequality and use expectation on q(z|x) instead of the integral to obtain the ELBO:

$p(x)> E[log p(x|z)] - KL[q(z|x)||p(z)]$ 

The ELBO is a lower bound to the marginal or model evidence. The first term measures how well samples from q(z|x) are able to explain the data x (Reconstruction cost). The second term ensures that the explanation of the data q(z|x) doesn’t deviate too far from prior beliefs p(z) which is a mechanism for realising Ockham’s razor (Regularization).

In implementing a parametric approach to density estimation, there are two main decisions that have to be made. The first decision is to specify the parametric form of the density function (variational distribution q). This is essentially the same as specifying the hypothesis space of a learning algorithm. 

The second decision that has to be made is how to learn a parametric model based on training data. There are three main approaches to this problem. The first approach, called maximum likelihood, says that one should choose parameter values that maximize the components that directly sample data (i.e. the likelihood or the probability of the data under the model, this corresponding to the first term in the ELBO, or an error function). In a sense, they provide the best possible fit to the data. As a result, there is a tendency towards overfitting. If we have a small amount of data, we can come to some quick conclusions due to small sample size.

Baysian approach is a belief system that suggests every variable including parameters should be beliefs (probability dist) not a single value. It provides an alternative approach to maximum likelihood, that does not make such a heavy commitment towards a single set of parameter values, and incorporates prior knowledge. In the Bayesian approach, all parameter values are considered possible, even after learning. No single set of parameter values is selected. Instead of choosing the set of parameters that maximize the likelihood, we maintain a probability distribution over the set of parameter values. The ELBO represents Bayesian approach where the second term represents balancing a prior distribution with the model's explanation of the data. The third approach, maximum a-posteriori (MAP), is a compromise between the maximum likelihood and the Bayesian belief system. The MAP estimate is the single set of parameters that maximize the probability under the posterior and is found by solving a penalized likelihood problem. However, it remedies the overfitting problem a bit by veering away from the maximum likelihood estimate (MLE), if MLE has low probability under the prior. As more data are seen, the prior term will be swallowed by the likelihood term, and the estimate will look more and more like the MLE.

Note that the loss function (i.e. the distance measure) in optimization is the root of the all problems since optimization's only objective is to reduce the loss. If the loss is not properly defined, the model can't learn well and if the loss function does not consider the inherent noise of the data (i.e. regularization), the model will eventually overfit to noise in the data and reduce generalization. Therefore, the loss function (distance measure) is very important and the reason why GANs work so well is that they don't explicitly define a loss function and learn it instead! The reason that Bayesian approach prevents overfitting is because they don't optimize anything, but instead marginalise (integrate) over all possible choices. The problem then lies in the choice of proper prior beliefs regarding the model.

#### Variational inference details

Variational Inference turns the inference into optimization. It posits a variational family of distributions over the latent variables,  and fit the variational parameters to be close (in KL sense or another divergence like BP, EP, etc) to the exact posterior. KL is intractable (only possible exactly if q is simple enough and compatible with the prior), so VI optimizes the evidence lower bound (ELBO) instead which is a lower bound on log p(x). Maximizing the ELBO is equivalent to minimizing the KL, note that the ELBO is not convex. The ELBO trades off two terms, the first term prefers q(.) to place its mass on the MAP estimate. The second term encourages q(.) to be diffuse.

To optimize the ELBO, Traditional VI uses coordinate ascent which iteratively update each parameter, holding others fixed. Classical VI is inefficient since they do some local computation for each data point. Aggregate these computations to re-estimate global structure and Repeat. In particular, variational inference in a typical model where a local latent variable is introduced for every observation (z->x) would involve introducing variational distributions for each observation, but doing so would require a lot of parameters, growing linearly with observations. Furthermore, we could not quickly infer x given a previously unseen observation. We will therefore perform amortised inference where we introduce an inference network for all observations instead.

Stochastic variational inference (SVI) scales VI to massive data. Additionally, SVI enables VI on a wide class of difficult models and enable VI with elaborate and flexible families of approximations. Stochastic Optimization replaces the gradient with cheaper noisy estimates and is guaranteed to converge to a local optimum. Example is SGD where the gradient is replaced with the gradient of a stochastic sample batch. The variational inferene recipe is:
1. Start with a model
2. Choose a variational approximation (variational family)
3. Write down the ELBO and compute the expectation (integral). 
4. Take ELBO derivative 
5. Optimize using the GD/SGD update rule

We usually get stuck in step 3, calculating the expectation (integral) since it's intractable. We refer to black box variational Inference to compute ELBO gradients without calculating its expectation. The way it works is to combine steps 3 and 4 above to calculate the gradient of expectation in a single step using variational methods instead of exact method of 3 then 4. Three main ideas for computing the gradient are score function gradient, pathwise gradients, and amortised inference. 

- Score function gradient: The problem is to calculate the gradient of an expectation of a funtion $$ \nabla_\theta (E_q(z) [f(z)])=\nabla_\theta( \int q(z)f(z))$$ with respect to parameters $$\theta$$. The function here is ELBO but gradient is difficult to compute since the integral is unknown or the ELBO is not differentiable. To calculate the gradient, we first take the $$\nabla_\theta$$ inside the integral to rewrite it as $$\int \nabla_\theta(q(z)) f(z) dz$$ since only the $$q(z)$$ is a function of $$\theta$$. Then we use the log derivative trick (using the derivative of the logarithm $d (log(u))= d(u)/u$) on the (ELBO) and re-write the integral as an expectation $$\nabla_\theta (E_q(z) [f(z)]) = E_q(z) [\nabla_\theta \log q(z) f(z)]$$. This estimator now only needs the dervative $$\nabla \log q_\theta (z)$$ to estimate the gradient. The expectation will be replaced with a Monte Carlo Average. When the function we want derivative of is log likelihood, we call the derivative $\nabla_\theta \log ⁡p(x;\theta)$ a score function. The expected value of the score function is zero.[](http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/)

- The form after applying the log-derivative trick is called the score ratio. This gradient is also called REINFORCE gradient or likelihood ratio. We can then obtain noisy unbiased estimation of this gradients with Monte Carlo. To compute the noisy gradient of the ELBO we sample from variational approximate q(z;v), evaluate gradient of log q(z;v), and then evaluate the log p(x, z) and log q(z). Therefore there is no model specific work and and this is called black box inference. The problem with this approach is that sampling rare values can lead to large scores and thus high variance for gradient. There are a few methods that help with reducing the variance but with a few more non-restricting assumptions we can find a better method with low variance i.e. pathwise gradients. 

- Pathwise Gradients of the ELBO: This method has two more assumptions, the first is assuming the hidden random variable can be reparameterized to represent the random variable z as a function of deterministic variational parameters $v$ and a random variable $\epsilon$, $z=f(\epsilon, v)$. The second is that log p(x, z) and log q(z) are differentiable with respect to z. With reparameterization trick, this amounts to a differentiable deterministic variational function. This method generally has a better behaving variance. 

- Amortised inference: Pathwise gradients would need to estimate a value for each data sample in the training. The basic idea of amortised inference is to learn a mapping from data to variational parameters to remove the computational cost of calculation for every data point. In stochastic variation inference, after random sampling, setting local parameters also involves an intractable expectation which should be calculated with another stochastic optimization for each data point. This is also the case in classic inference like an EM algorithm, the learned distribution/parameters in the E-step is forgotten after the M-step update. In amortised inference, an inference network/tree/basis function might be used as the inference function from data to variational parameters to amortise the cost of inference. So in a sense, this way there is some sort of memory in the inferene and the learned params are not forgotten each time but rather updated in each step and thus they amortise the cost of inference. Amortized inference is faster, but admits a smaller class of approximations, the size of which depends on the flexibility of f.

In summary, for variational inference, if log p(x, z) is z-differentiable:
- Try out an approximation q that is reparameterizable (end to end differentiable model)
If log p(x, z) is not z differentiable:
- Use score function estimator with control variates
- Add further variance reductions based on experimental evidence



#### Advanced ideas in VI (Auxiliary Variational Method, Variational Inference with Normalizing Flows, Hierarchical Variational Models, Auxiliary Deep Generative Models):


Mean-field or fully-factorised approximate posterior distributions q(z|x) is usually not sufficient for modelling real world data (Complex dependencies, Non-Gaussian distributions, Multiple modes) but traditionally we've been restricted to them due to limitations in solving the optimization problem. This is the same challenge encountered in the problem of specifying a model of the data (prior) itself which was restricted by our limitations in solving the inference problem (e.g. conjugate priors).

However, advances in probabilistic modelling (probabilistic programming), scalable inference through stochastic optimization, and blackbox variational inference (non-conjugate models, Monte Carlo gradient estimators, and amortised inference) have recently enabled us to design more flexible models and approximate posterior distributions q(z|x) for our models. So the goal is to build richer approximate posterior distributions and maintain the computational efficiency and scalability. A few examples follow.

- In Gaussian Approximate Posteriors, Structure of covariance $\sum$ describes dependency. Mean field assumption (i.e. diagonal covariance matrix as in a VAE) is least expressive but computatoinally efficient. Full covariance is richest, but computationally expensive. There are ideas around using linear algebra to efficiently decompose the covariance matrix, for example using a tri-diagonal covariance for an LDS approximate posterior. A limitation of Gaussian variational distributions is that the posterior is always Gaussian which doesn't always reflect real world distributions. 

- Autoregressive posterior distributions impose an ordering and non-linear dependency on all preceding hidden variables. Therfore the Joint-distribution is non-Gaussian and very expressive, although conditional distributions are Gaussian. For example compare Gaussian mean field posterior (VAE) vs. auto-regressive posterior (DRAW).

- More Structured posteriors introduce additional variables that induce dependencies, but that remain tractable and efficient for example and LDS/HMM/etc assumption on the posterior in an SVAE.

- In order to design richer approximate posteriors, we can introduce new variables that help to form a richer approximate posterior (where q(z|x) is the marginalization of those new variables). We need to be able to adapt bound (ELBO) to compute entropy or a bound, and maintain computational efficiency to be linear in number of latent variables. There are two main approaches for doing this. First is change-of-variables including Normalising flows and invertible transforms. The second is auxiliary variables which involves Entropy bounds, and Monte Carlo sampling.

In approximations using Change-of-variables, the distribution flows through a sequence of invertible transforms. We begin with an initial distribution q0(z0|x) and apply a sequence of K invertible functions fk that will reshape the posterior to more complex shapes. We Begin with a fully-factorised Gaussian and improve by change of variables. Triangular Jacobians allow for computational efficiency. These include Planar flow, real NVP, and inverse autoregressive flow. These models have a linear time computation of the determinant and its gradient. 

Another possible approach is using hierarchical approximate posteriors. The basic idea is to use a hierarchical model for the approximate posterior (We can use latent variables, like we density models). In that case, the new variables are stochastic variables rather than deterministic in the change-of-variables approach and both continuous and discrete latent variables can be modelled.

The Auxiliary-variable methods add exra latent variables in parallel to existing hiddens and don't change the original model. They capture structure of correlated variables because they turn the posterior into a mixture of distributions q(z|x, a). The richer posterior can be a mixture model, normalising flow, Gaussian process, etc but needs to have easy sampling for evaluation of bound and gradients. 

#### Variational inference from first principles 

Given a model of latent and observed variables p(x, z), variational inference posits a family of distributions over its latent variables and then finds the member of that family closest to the posterior, p(z | x). This is typically formalized as minimizing a Kullback-Leibler (kl) divergence from the approximating family q(·) to the posterior p(·). However, while the kl(q k p) objective offers many beneficial computational properties, it is ultimately designed for convenience; it sacrifices many desirable statistical properties of the resultant approximation. When optimizing kl, there are two issues with the posterior approximation that we highlight. First, it typically underestimates the variance of the posterior. Second, it can result in degenerate solutions that zero out the probability of certain configurations of the latent variables. While both of these issues can be partially circumvented by using more expressive approximating families, they ultimately stem from the choice of the objective. Under the kl divergence, we pay a large price when q(·) is big where p(·) is tiny; this price becomes infinite when q(·) has larger support than p(·).

In "Operator Variational Inference", authors revisit variational inference from its core principle as an optimization problem. We use operators—mappings from functions to functions—to design variational objectives, explicitly trading off computational properties of the optimization with statistical properties of the approximation. We use operators to formalize the basic properties needed for variational inference algorithms. We further outline how to use them to define new variational objectives; as one example, we design a variational objective using a Langevin-Stein operator. the Langevin-Stein objective enjoys two good properties. First, it is amenable to data subsampling, which allows inference to scale to massive data. Second, it permits rich approximating families, called variational programs, which do not require analytically tractable densities. This greatly expands the class of variational families and the fidelity
of the resulting approximation.


#### MCMC inituition
We should explore the deformed posterior space generated by our prior surface and observed data to find the posterior mountain. However, we cannot naively search due to curse of dimensionality. The idea behind the Markov Chain in MCMC is to perform an intelligent search of the space. Sampling from the Markov Chain is similar to repeatedly asking "How likely is this pebble (sample/trace) I found to be from the mountain (unknown distribution) I am searching for?", and completes its task by returning thousands of accepted pebbles in hopes of reconstructing the original mountain. MCMC searches by exploring nearby positions and moving into areas with higher probability (hill climbing). With the thousands of samples, we can reconstruct the posterior surface by organizing them in a histogram.

The Monte Carlo part comes in when we have found accepted samples from the unknown distribution. We can use Monte Carlo estimates to estimate statistics on the unknown distribution. For example an average on samples to estimate an expectation.

Several Markov chain methods are available for sampling from a posterior distribution. Two important examples are the Gibbs sampler and the Metropolis algorithm. In addition, several strategies are available for constructing hybrid algorithms (for example Hamilonian MC).

#### MCMC Algorithm
It's a method, for sampling from an untractable distribution that we don't know and computing statistics using the samples. MCMC involves setting up a random sampling process for navigating to different states based on the transition matrix of the markov chain until the dynamical system settles down on a final stable state (stationary distribution), i.e. start at state 0, randomly switch to another state based on transition matrix, repeat until convegence at which time we can sample from the unknown distribution. If the transition matrix has a transition probability >0 for every two state and each state has a self probability >0, then the convergence is gauranteed to a unique distribution.

Therefore to sample from an unknown distribution p, we need to construct a markov chain T whose unique stationary distribution is p. Then we sample till we converge, at which time, we would be sampling from the unknown distribution. At that point we collect samples and compute our desired statistics using Monte Carlo resampling/simulation! 

##### Gibbs (Boltzman) sampling:
MCMC for gaphical models are done through Gibbs chain:

$$ p(x) \propto \exp(−U(x)/T) $$

Probability $$p(x)$$ of a system to be in the state $$x$$ depends on the energy of the state $$U(x)$$ and temperature $$T$$. Any distribution can be rewritten as Gibbs canonical distribution, but for many problems such energy-based distributions appear very naturally. Algorithms to perform MCMC using Gibbs distribution:

1.Start at current position.
2.Propose moving to a new position (sample).
3.Accept/Reject the new position based on the position's adherence to the data and prior distributions (check new point's probability agianst Gibbs probability distribution e.g. reject probability lower than a threshold ).
4.If you accept: 
A)Move to the new position. Return to Step 1.
B)Else: Do not move to new position. Return to Step 1.
5.After a large number of iterations, return all accepted positions.

This way we move in the general drection towards the regions where the posterior distributions exist. Once we reach the posterior distribution, we can easily collect samples as they likely all belong to the posterior distribution.

Note that the samples from the posterior are dependent samples and not independent samples. So if samples are used for a Monte Carlo estimate, this dependence has to be taken care of in the estimator. 


#### HMC Algorithm
Hamiltonian Monte Carlo use the intuition of the movement of a physical system. Instead of moving randomly, we assume a momentum and move according to equations of motions of a physical system derived from the Hamiltonian of the system. we obtain a new HMC sample as follows:

1. sample a new velocity from a univariate Gaussian distribution
2. perform n leapfrog steps according to the equations of motion to obtain the new state 
3. perform accept/reject move of the new state
4.If you accept: 
A)Move to the new position. Return to Step 1.
B)Else: Do not move to new position. Return to Step 1.
5.After a large number of iterations, return all accepted positions.


#### Monte Carlo Resampling/Simulation:
- Suppose you have some dataset in hand.
- We are interested in the real distribution that produced this data (density estimation).

- Solution, draw a new “sample” of data from your dataset. Repeat that many times so you have a lot of new simulated “samples”. This is called Monte Carlo resampling/Simulation!

- Resampling methods can be parametric (model based) or non-parametric.
- The fundamental assumption is that all information about the real distribution contained in the original dataset is also contained in the distribution of these simulated samples.
- Another way to think about this is that if the dataset you have in your hands is a reasonable representation of the population, then the parameter estimates produced from running a model on a series of resampled data sets will provide a good approximation of the distribution of that statistics in the population.

#### Resampling techniques(Bootstrap): 

- Begin with a dataset of size N
- Generate a simulated sample of size N by drawing from your dataset independently (uniformly) and with replacement.
- Compute and save the statistic of interest.
- Repeat this process many times (e.g. 1,000).
- Treat the distribution of your estimated statistics (e.g. mean) as an estimate of the real distribution of that statistic from population.

This approach is better than assuming a normal distribution for statistics of interest and directly compute from dataset (e.g. mean/variance/confidence_interval/etc) as in classical stat but is obviously way more costly! If the dataset is not representative of the real distribution, the simulated distribution of any statistics computed from that dataset will also probably not accurately reflect the population (Small samples, biased samples, or bad luck). Resampling one observation at a time with replacement assumes the data points in your observed sample are independent. If they are not, the simple bootstrap will not work. Fortunately the bootstrap can be adjusted to accommodate the dependency structure in the original sample. If the data is clustered (spatially correlated, multi-level, etc.) the solution is to resample clusters of observations one at a time with replacement rather than individual observations. If the data is time-serial dependent, this is harder because any sub-set you select still “breaks” the dependency in the sample, but methods are being developed.

It doesn't work well in data with serial correlation in the residual (time series). Models with heteroskedasticity when the form of the heteroskedasticity is unknown. One approach here is to sample pairs (on Y and X) rather than leaving X fixed in repeated samples. Simultaneous equation models because you have to bootstrap all of the endogenous variables in the model.

#### Posterior sampling
- Instead of drawing one single number, we draw a vector of numbers (one for each coefficient)
- Set a key variable in the model
- Calculate a quantity of interest (e.g. Expectation) with each set of simulated coefficients
- Update key variable
- repeat

#### Non-parametric density estimation

- Kernel density estimation(KDE): ntuitively, KDE has the effect of smoothing out each data point into a smooth bump, whose the shape is determined by the kernel function (Gaussian, spherical, etc). Then, KDE sums over all these bumps to obtain a density estimator. At regions with many observations, because there will be many bumps around, KDE will yield a large value. On the other hand, for regions with only a few observations, the density value from summing over the bumps is low, because only have a few bumps contribute to the density estimate. KDE has a smoothing parameter, $$h$$ that controls how the much effect each point has. When $$h$$ is too small, there are many wiggles in the density estimate. When $$h$$ is too large, we smooth out important features. When $$h$$ is at the correct amount, we can see a clear picture of the underlying density. We choose $$h$$ that minimizes the total amount of mean squared error of the density estimate from the real density. KDE can be used to estimate not only the underlying density function but also geometric (and topological) structures related to the density. Because geometric and topological structures generally involve the gradient and Hessian matrix (and it's eigen-decomposition) of the density function. Gradient and Hessian eigenvalues can determine local maxima (modes) of a density function. One can use the local modes to cluster data points; this is called mode clustering or mean-shift clustering. Level sets are regions for which the density value is equal to or above a particular level. Another interesting geometric structures are ridges. A cluster tree is formed by gradually decreasing level set value from a large number. As level sets decrease, a mode appears and with more decrease, new connected components will be created.

- A persistent diagram is a diagram summarizing the topological features of a density function p. The construction of a persistent diagram is very similar to that of a cluster tree, but now we focus on not only the connected components but also higher-order topological structures, such as loops and voids. 

- When the dimension of the data d is large, KDE poses
several challenges. First, KDE (and most other nonparametric density estimators) suffers
severely from the so-called curse of dimensionality. One way
to solve this problem is to find density surrogates that can be estimated easily and to
switch our parameter of interest to a density surrogate. Another issue of KDE in multi-dimensions is visualization. When d > 3,
we can no longer see the entire KDE, and we must therefore use visualization tools to
explore our density estimates.



### Density ratio estimation (including adversarial training)
In statistical pattern recognition, it is important to avoid density estimation (evidence marginal integral) since density estimation is often more difficult than pattern recognition itself. Following this idea—known as Vapnik’s principle, a statistical data processing framework that employs the ratio of two probability density functions has been developed recently and is gathering a lot of attention in the machine learning and data mining communities. The main idea is to estimate a ratio of real data distribution and model data distribution $$\frac{p(x)}{q(x)}$$ instead of computing two densities that are hard. The ELBO in variational inference can be written in terms of the ratio. Introducing the variational posterior into the marginal integral of the joint results in the ELBO being $$E[log p(x,z)- log q(z|x)]$$. By subtracting emprical distribution on the observations, $$p(x)$$ which is a constant and doesn't change optimization we have the ELBO using ratio as $$E[\log \frac{p(x,z)}{q(x,z)}]= E[\log r(x,z)]$$. The density ratio $$r(x)$$ is the core quantity for hypothesis testing, motivated by either the Neyman-Pearson lemma or the Bayesian posterior evidence, appearing as the likelihood ratio or the Bayes factor, respectively. likelihood-free inference can be done through estimating density ratios and using them as the driving principle for learning in implicit generative models (transformation models). Four main ways of calculating the ratio:

1. Probabilistic classification: We can frame it as the problem of classifying the real data $$p(x)$$ and the data produced by the model $$q(x)$$. We use a label of (+1) for the numerator and label (-1) for denumerator so the ratio will be $$r(x)=\frac{p(x|+1)}{q(x|-1)}$$. Using Bayesian rule this will be $$r(x)=(\frac{p(-1)}{p(+1)})*(\frac{p(+1|x)}{p(-1|x)})$$. The first ratio is simply the ratio of the number of data in each class and the second ratio is given by the ratio of classification accuracies. simple and elegant! 

- This is what happens in GAN. So if there are $N1$ data real points and $N2$ generated data points and the classifer classifies the real data points with probability $$D$$, then the ratio is $$r(x)= (N2/N1) * (D/(D-1))$$. Given the classifer, we can develop a loss function for training using logarithmic loss for binary classification. Using some simple math we get the GAN loss function as: 

$$L= \pi E[-log D(x,\phi)]+ (1-\pi) E[-log (1-D(G(z,\theta),\phi))], pi=p(+1|x)$$

In practice, the expectations are computed by Monte Carlo integration using samples from $$p$$ and $$q$$. This loss specifies a bi-level optimisation by forming a ratio loss and a generative loss, using which we perform an alternating optimisation. The ratio loss is formed by extracting all terms in the loss related to the ratio function parameters $$\phi$$, and minimise the resulting objective. For the generative loss, all terms related to the model parameters $$\theta$$ are extracted, and maximized.

$$min L_D= \pi E[-log D(x,\phi)]+ (1-\pi) E[-log (1-D(x,\phi))]$$

$$min L_G= E[log (1-D(G(z,\theta)))]$$

The ratio loss is minimised since it acts as a surrogate negative log-likelihood; the generative loss is minimised since we wish to minimise the probability of the negative (generated-data) class. We first train the discriminator by minimising $$L_D$$ while keeping $$G$$ fixed, and then we fix $$D$$ and take a gradient step to minimise $$L_G$$.

2. moment matching: if all the infinite statistical moments of two distributions are the same the distributions are the same. So the idea is to set the moments of the numenator distribution (p(x)) equal to the moments of a transformed version of the denumerator (r(x)q(x)). This makes it possible to calculate the ratio r(x).

3. Ratio matching: basic idea is to directly match a density ratio model r(x) to the true density ratio under some divergence. A kernel is usually used for this density estimation problem plus a distance measure (e.g. KL divergence) to measure how close the estimation of r(x) is to the true estimation. So it's variational in some sense. Loosely speaking, this is what happens in variational Autoencoders!

4. Divergence minimization: Another approach to two sample testing and density ratio estimation is to use the divergence between the true density p and the model q, and use this as an objective to drive learning of the generative model. f-GANs use the KL divergence as a special case and are equipped with an exploitable variational formulation (i.e. the variational lower bound). There is no discriminator in this formulation, and this role is taken by the ratio function. We minimise the ratio loss, since we wish to minimise the negative of the variational lower bound; we minimise the generative loss since we wish to drive the ratio to one.

5. Maximum mean discrepancy(MMD): is a nonparametric way to measure dissimilarity between two probability distributions. Just like any metric of dissimilarity between distributions, MMD can be used as an objective function for generative modelling.  The MMD criterion also uses the concept of an 'adversarial' function f that discriminates between samples from Q and P. However, instead of it being a binary classifier constrained to predict 0 or 1, here f can be any function chosen from some function class. MMD uses functions from a kernel Hilbert space as discriminatory functions. The discrimination is measured not in terms of binary classification accuracy as above, but as the difference between the expected value of f under P and Q. The idea is: if P and Q are exactly the same, there should be no function whose expectations differ under Q and P. In GAN, the maximisation over f is carried out via stochastic gradient descent, here it can be done analytically. One could design a kernel which has a deep neural network in it, and use the MMD objective!?

6. Instead of estimating ratios, we estimate gradients of log densities. For this, we can use[ denoising as a surrogate task](http://www.inference.vc/variational-inference-using-implicit-models-part-iv-denoisers-instead-of-discriminators/). denoisers estimate gradients directly, and therefore we might get better estimates than first estimating likelihood ratios and then taking the derivative of those.

 

### Energy-based training (alternative to GAN)

"Optimizing the Latent Space of Generative Networks" is a new paper from FAIR that describes the GLO model (Generative Latent Optimization).

Slightly less short story: GLO, like GAN and VAE, is a way to train a generative model under uncertainty on the output.

Short story: GLO is a generative model in which a set of latent variables is optimized at training time to minimize a distance between a training sample and a reconstruction of it produced by the generator. This alleviates the need to train a discriminator as in GAN.

A generative model must be able to generate a whole series of different outputs, for example, different faces, or different bedroom images.
Generally, a set of latent variables Z is drawn at random every time the model needs to generate an output. These latent variables are fed to a generator G that produces an output Y(e.g. an image) Y=G(Z).

Different drawings of the latent variable result in different images being produced, and the latent variable can be seen as parameterizing the set of outputs.

In GAN, the latent variable Z is drawn at random during training, and a discriminator is trained to tell if the generated output looks like it's been drawn from the same distribution as the training set.
In GLO, the latent variable Z is optimized during training so as to minimize some distance measure between the generated sample and the training sample Z* = min_z = Distance(Y,G(Z)). The parameters of the generator are adjusted after this minimization. The learning process is really a joint optimization of the distance with respect to Z and to the parameters of G, averaged on a training set of samples.

After training, Z can be sampled from their allowed set to produce new samples. Nice examples are shown in the paper.

GLO belongs to a wide category of energy-based latent variable models: define a parameterized energy function E(Y,Z), define a "free energy" F(Y) = min_z E(Y,Z). Then find the parameters that minimize F(Y) averaged over your training set, making sure to put some constraints on Z so that F(Y) doesn't become uniformly flat (and takes high values outside of the region of high data density). This basic model is at the basis of sparse modeling, sparse auto-encoders, and the "predictive sparse decomposition" model. In these models, the energy contains a term that forces Z to be sparse, and the reconstruction of Y from Z is linear. In GLO, the reconstruction is computed by a deconvolutional net.



## Causal inference

- Two types of studies are possible, one is interventional studies where in a controlled environment, we introduce an intervention. Causality inference is directly possible due to the intervention. However, we might not have access to interventions. In such cases, we want to perform causal inference using only observation data. 