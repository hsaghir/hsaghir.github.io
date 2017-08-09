---
layout: article
title: Operator Variational Inference
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

## Variational inference from first principles 

Given a model of latent and observed variables p(x, z), variational inference posits a family of distributions over its latent variables and then finds the member of that family closest to the posterior, p(z | x). This is typically formalized as minimizing a Kullback-Leibler (kl) divergence from the approximating family q(·) to the posterior p(·). However, while the kl(q k p) objective offers many beneficial computational properties, it is ultimately designed for convenience; it sacrifices many desirable statistical properties of the resultant approximation. When optimizing kl, there are two issues with the posterior approximation that we highlight. First, it typically underestimates the variance of the posterior. Second, it can result in degenerate solutions that zero out the probability of certain configurations of the latent variables. While both of these issues can be partially circumvented by using more expressive approximating families, they ultimately stem from the choice of the objective. Under the kl divergence, we pay a large price when q(·) is big where p(·) is tiny; this price becomes infinite when q(·) has larger support than p(·).

In "Operator Variational Inference", authors revisit variational inference from its core principle as an optimization problem. We use operators—mappings from functions to functions—to design variational objectives, explicitly trading off computational properties of the optimization with statistical properties of the approximation. We use operators to formalize the basic properties needed for variational inference algorithms. We further outline how to use them to define new variational objectives; as one example, we design a variational objective using a Langevin-Stein operator. the Langevin-Stein objective enjoys two good properties. First, it is amenable to data subsampling, which allows inference to scale to massive data. Second, it permits rich approximating families, called variational programs, which do not require analytically tractable densities. This greatly expands the class of variational families and the fidelity
of the resulting approximation.



## Classic KL- Variational Inference

Variatonal inference turns the problem of inference into optimization. It started in 80s, by bringing ideas like mean field from statistical physics to probabilistic methods. In the 90s, Jordan et al developed it further to generalize to more inference problems and in parallel Hinton et al developed mean field for neural nets and connected it to EM which lead to VI for HMMs and Mixtures. Modern VI touches many areas like, probabilistic programming, neural nets, reinforcement learning, convex optimization, bayesian statistics and many applications. 

Basic idea is to transform the integral into an expectation over a simple, known distribution using the variational approximate (ELBO). Writing the log-likelihood for the evidence $log p(x)=log \int p(x,z)$ and introducing the variational approximate $log p(x)= log \int p(x,z)*q(z|x)/q(z|x)$, we can move the $log$ inside the integral using Jensen's inequality and use expectation on q(z|x) instead of the integral to obtain the ELBO:

$p(x)> E[log p(x|z)] - KL[q(z|x)||p(z)]$ 

The ELBO is a lower bound to the marginal or model evidence. The first term measures how well samples from q(z|x) are able to explain the data x (Reconstruction cost). The second term ensures that the explanation of the data q(z|x) doesn’t deviate too far from prior beliefs p(z) which is a mechanism for realising Ockham’s razor (Regularization).

In implementing a parametric approach to density estimation, there are two main decisions that have to be made. The first decision is to specify the parametric form of the density function (variational distribution q). This is essentially the same as specifying the hypothesis space of a learning algorithm. 

The second decision that has to be made is how to learn a parametric model based on training data. There are three main approaches to this problem. The first approach, called maximum likelihood, says that one should choose parameter values that maximize the components that directly sample data (i.e. the likelihood or the probability of the data under the model, this corresponding to the first term in the ELBO, or an error function). In a sense, they provide the best possible fit to the data. As a result, there is a tendency towards overfitting. If we have a small amount of data, we can come to some quick conclusions due to small sample size.

Baysian approach is a belief system that suggests every variable including parameters should be beliefs (probability dist) not a single value. It provides an alternative approach to maximum likelihood, that does not make such a heavy commitment towards a single set of parameter values, and incorporates prior knowledge. In the Bayesian approach, all parameter values are considered possible, even after learning. No single set of parameter values is selected. Instead of choosing the set of parameters that maximize the likelihood, we maintain a probability distribution over the set of parameter values. The ELBO represents Bayesian approach where the second term represents balancing a prior distribution with the model's explanation of the data. The third approach, maximum a-posteriori (MAP), is a compromise between the maximum likelihood and the Bayesian belief system. The MAP estimate is the single set of parameters that maximize the probability under the posterior and is found by solving a penalized likelihood problem. However, it remedies the overfitting problem a bit by veering away from the maximum likelihood estimate (MLE), if MLE has low probability under the prior. As more data are seen, the prior term will be swallowed by the likelihood term, and the estimate will look more and more like the MLE.

Note that the loss function (i.e. the distance measure) in optimization is the root of the all problems since optimization's only objective is to reduce the loss. If the loss is not properly defined, the model can't learn well and if the loss function does not consider the inherent noise of the data (i.e. regularization), the model will eventually overfit to noise in the data and reduce generalization. Therefore, the loss function (distance measure) is very important and the reason why GANs work so well is that they don't explicitly define a loss function and learn it instead! The reason that Bayesian approach prevents overfitting is because they don't optimize anything, but instead marginalise (integrate) over all possible choices. The problem then lies in the choice of proper prior beliefs regarding the model.

### Variational inference details

Variational Inference turns the inference into optimization. It posits a variational family of distributions over the latent variables,  and fit the variational parameters to be close (in KL sense or another divergence like BP, EP, etc) to the exact posterior. KL is intractable (only possible exactly if q is simple enough and compatible with the prior), so VI optimizes the evidence lower bound (ELBO) instead which is a lower bound on log p(x). Maximizing the ELBO is equivalent to minimizing the KL, note that the ELBO is not convex. The ELBO trades off two terms, the first term prefers q(.) to place its mass on the MAP estimate. The second term encourages q(.) to be diffuse.

To optimize the ELBO, Traditional VI uses coordinate ascent which iteratively update each parameter, holding others fixed. Classical VI is inefficient since they do some local computation for each data point. Aggregate these computations to re-estimate global structure and Repeat. In particular, variational inference in a typical model where a local latent variable is introduced for every observation (z->x) would involve introducing variational distributions for each observation, but doing so would require a lot of parameters, growing linearly with observations. Furthermore, we could not quickly infer x given a previously unseen observation. We will therefore perform amortised inference where we introduce an inference network for all observations instead.

Stochastic variational inference (SVI) scales VI to massive data. Additionally, SVI enables VI on a wide class of difficult models and enable VI with elaborate and flexible families of approximations. Stochastic Optimization replaces the gradient with cheaper noisy estimates and is guaranteed to converge to a local optimum. Example is SGD where the gradient is replaced with the gradient of a stochastic sample batch. The variational inferene recipe is:
1. Start with a model
2. Choose a variational approximation (variational family)
3. Write down the ELBO and compute the expectation (integral). 
4. Take ELBO derivative 
5. Optimize using the GD/SGD update rule

We usually get stuck in step 3, calculating the expectation (integral) since it's intractable. We refer to black box variational Inference to compute ELBO gradients without calculating its expectation. The way it works is to combine steps 3 and 4 above to calculate the gradient of expectation in a single step using variational methods instead of exact method of 3 then 4. Three main ideas for computing the gradient are score function gradient, pathwise gradients, and amortised inference. 

- Score function gradient: The problem is to calculate the gradient of an expectation of a funtion $ grad(E_q(z) [f(z)])=grad( \int q(z)f(z))$. The function here is ELBO but gradient is difficult to compute since the integral is unknown or the ELBO is not differentiable. To calculate the gradient of the ELBO, we use some simple algebra (i.e. the log derivative trick, using the property of the derivative of the logarithm $d (log(u))= d(u)/u$ ) on the ELBO and re-write it using a function estimator (i.e. score function). The gradient of the log likelihood, i.e. $∇log ⁡p(x;θ)$ is called a score function. The expected value of the score function is zero.

- The form after applying the log-derivative trick is called the score ratio. This gradient is also called REINFORCE gradient or likelihood ratio. We can then obtain noisy unbiased estimation of this gradients with Monte Carlo. To compute the noisy gradient of the ELBO we sample from variational approximate q(z;v), evaluate gradient of log q(z;v), and then evaluate the log p(x, z) and log q(z). Therefore there is no model specific work and and this is called black box inference. The problem with this approach is that sampling rare values can lead to large scores and thus high variance for gradient. There are a few methods that help with reducing the variance but with a few more non-restricting assumptions we can find a better method with low variance i.e. pathwise gradients. 

- Pathwise Gradients of the ELBO: This method has two more assumptions, the first is assuming the hidden random variable can be reparameterized to represent the random variable z as a function of deterministic variational parameters $v$ and a random variable $\epsilon$, $z=f(\epsilon, v)$. The second is that log p(x, z) and log q(z) are differentiable with respect to z. With reparameterization trick, this amounts to a differentiable deterministic variational function. This method generally has a better behaving variance. 

- Amortised inference: Pathwise gradients would need to estimate a value for each data sample in the training. The basic idea of amortised inference is to learn a mapping from data to variational parameters to remove the computational cost of calculation for every data point. In stochastic variation inference, after random sampling, setting local parameters also involves an intractable expectation which should be calculated with another stochastic optimization for each data point. This is also the case in classic inference like an EM algorithm, the learned distribution/parameters in the E-step is forgotten after the M-step update. In amortised inference, an inference network/tree/basis function might be used as the inference function from data to variational parameters to amortise the cost of inference. So in a sense, this way there is some sort of memory in the inferene and the learned params are not forgotten each time but rather updated in each step and thus they amortise the cost of inference. Amortized inference is faster, but admits a smaller class of approximations, the size of which depends on the flexibility of f.

In summary, for variational inference, if log p(x, z) is z differentiable:
- Try out an approximation q that is reparameterizable (end to end differentiable model)
If log p(x, z) is not z differentiable:
- Use score function estimator with control variates
- Add further variance reductions based on experimental evidence



### Advanced ideas in VI (Auxiliary Variational Method, Variational Inference with Normalizing Flows, Hierarchical Variational Models, Auxiliary Deep Generative Models):


Mean-field or fully-factorised approximate posterior distributions q(z|x) is usually not sufficient for modelling real world data (Complex dependencies, Non-Gaussian distributions, Multiple modes) but traditionally we've been restricted to them due to limitations in solving the optimization problem. This is the same challenge encountered in the problem of specifying a model of the data (prior) itself which was restricted by our limitations in solving the inference problem (e.g. conjugate priors).

However, advances in probabilistic modelling (probabilistic programming), scalable inference through stochastic optimization, and blackbox variational inference (non-conjugate models, Monte Carlo gradient estimators, and amortised inference) have recently enabled us to design more flexible models and approximate posterior distributions q(z|x) for our models. So the goal is to build richer approximate posterior distributions and maintain the computational efficiency and scalability. A few examples follow.

- In Gaussian Approximate Posteriors, Structure of covariance $\sum$ describes dependency. Mean field assumption (i.e. diagonal covariance matrix as in a VAE) is least expressive but computatoinally efficient. Full covariance is richest, but computationally expensive. There are ideas around using linear algebra to efficiently decompose the covariance matrix, for example using a tri-diagonal covariance for an LDS approximate posterior. A limitation of Gaussian variational distributions is that the posterior is always Gaussian which doesn't always reflect real world distributions. 

- Autoregressive posterior distributions impose an ordering and non-linear dependency on all preceding hidden variables. Therfore the Joint-distribution is non-Gaussian and very expressive, although conditional distributions are Gaussian. For example compare Gaussian mean field posterior (VAE) vs. auto-regressive posterior (DRAW).

- More Structured posteriors introduce additional variables that induce dependencies, but that remain tractable and efficient for example and LDS/HMM/etc assumption on the posterior in an SVAE.

- In order to design richer approximate posteriors, we can introduce new variables that help to form a richer approximate posterior (where q(z|x) is the marginalization of those new variables). We need to be able to adapt bound (ELBO) to compute entropy or a bound, and maintain computational efficiency to be linear in number of latent variables. There are two main approaches for doing this. First is change-of-variables including Normalising flows and invertible transforms. The second is auxiliary variables which involves Entropy bounds, and Monte Carlo sampling.

In approximations using Change-of-variables, the distribution flows through a sequence of invertible transforms. We begin with an initial distribution q0(z0|x) and apply a sequence of K invertible functions fk that will reshape the posterior to more complex shapes. We Begin with a fully-factorised Gaussian and improve by change of variables. Triangular Jacobians allow for computational efficiency. These include Planar flow, real NVP, and inverse autoregressive flow. These models have a linear time computation of the determinant and its gradient. 

Another possible approach is using hierarchical approximate posteriors. The basic idea is to use a hierarchical model for the approximate posterior (We can use latent variables, like we density models). In that case, the new variables are stochastic variables rather than deterministic in the change-of-variables approach and both continuous and discrete latent variables can be modelled.

The Auxiliary-variable methods add exra latent variables in parallel to existing hiddens and don't change the original model. They capture structure of correlated variables because they turn the posterior into a mixture of distributions q(z|x, a). The richer posterior can be a mixture model, normalising flow, Gaussian process, etc but needs to have easy sampling for evaluation of bound and gradients. 


