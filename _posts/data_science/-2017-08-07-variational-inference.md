---
layout: article
title: Variational inference
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

# Variational inference

For most interesting models, the denominator of posterior is not tractable. We appeal to approximate posterior inference.

Traditional solution to the inference problem involves Gibbs sampling which is a variant of MCMC based on sampling from the posterior. Gibbs sampling is based on randomness and sampling, has strict conjugacy requirements, require many iterations and sampling, and it's not very easy to gauge whether the Markov Chain has converged to start collecting samples from the posterior. 

## Background 

Variatonal inference turns the problem of inference into optimization. It started in 80s, by bringing ideas like mean field from statistical physics to probabilistic methods. In the 90s, Jordan et al developed it further to generalize to more inference problems and in parallel Hinton et al developed mean field for neural nets and connected it to EM which lead to VI for HMMs and Mixtures. Modern VI touches many areas like, probabilistic programming, neural nets, reinforcement learning, convex optimization, bayesian statistics and many applications. 

In implementing a parametric approach to density estimation, there are two main decisions that have to be made. The first decision is to specify the parametric form of the density function (variational distribution q). This is essentially the same as specifying the hypothesis space of a learning algorithm. The second decision that has to be made is how to learn a parametric model based on training data. There are three main approaches to this problem i.e. Maximum likelihood, Bayesian inference and MAP inference. 

Variational Inference turns Bayesian inference into optimization. Given a model of latent and observed variables $$p(X, Z)$$, variational inference posits a family of distributions over its latent variables and then finds the member of that family closest to the posterior, $$p(Z|X)$$. This is typically formalized as minimizing the Kullback-Leibler (KL) divergence from the approximating family $$q(·)$$ to the posterior $$p(·)$$.

### ELBO

- The ELBO is an equivalent objective to KL divergence between the true and variational posteriors. Since the KL divergence is intractable, we instead minimize the ELBO equivalent objective.

- Basic idea is to transform the integral into an expectation over a simple, known distribution using the variational approximate (ELBO). Writing the log-likelihood for the evidence $log p(x)=log \int p(x,z)$ and introducing the variational approximate $log p(x)= log \int p(x,z)*q(z|x)/q(z|x)$, we can move the $log$ inside the integral using Jensen's inequality and use expectation on q(z|x) instead of the integral to obtain the ELBO:

$p(x)> E[log p(x|z)] - KL[q(z|x)||p(z)]$ 

The ELBO is a lower bound to the marginal or model evidence. The first term measures how well samples from q(z|x) are able to explain the data x (Reconstruction cost). The second term ensures that the explanation of the data q(z|x) doesn’t deviate too far from prior beliefs p(z) which is a mechanism for realising Ockham’s razor (Regularization).

#### ELBO derivation

- The ELBO is determined from introducing a variational distribution, $$q$$, to the marginal log likelihood, i.e. $$\log \ p(x)=\log \int_z p(x,z) * \frac{q(z|x)}{q(z|x)}$$. We use the log likelihood to be able to use the concavity of the $$\log$$ function and employ Jensen's equation to move the $$\log$$ inside the integral i.e. $$\log \ p(x) > \int_z \log\ (p(x,z) * \frac{q(z|x)}{q(z|x)})$$ and then use the definition of expectation on $$q$$ (the nominator $$q$$ goes into the definition of the expectation on $$q$$ to write that as the ELBO) $$\log \ p(x) > ELBO(z) = E_q [- \log\ q(z|x) + \log \ p(x,z)]$$. The difference between the ELBO and the marginal $$p(x)$$ which converts the inequality to an equality is the distance between the real posterior and the approximate posterior i.e. $$KL[q(z|x)\ | \ p(z|x)]$$. Alternatively, the distance between the ELBO and the KL term is the log-normalizer $$p(x)$$. Replace the $$p(z|x)$$ with Bayesian formula to see how. 

Now that we have a defined a loss function, we need the gradient of the loss function, $$\delta E_q[-\log q(z \vert x)+p(x,z)]$$ to be able to use it for optimization with SGD. The gradient isn't easy to derive analytically but we can estimate it using MCMC to directly sample from $$q(z \vert x)$$ and estimate the gradient. This approach generally exhibits large variance since MCMC might sample from rare values.

This is where the re-parameterization trick we discussed above comes in. We assume that the random variable $$z$$ is a deterministic function of $$x$$ and a known $$\epsilon$$ ($$\epsilon$$ are iid samples) that injects randomness $$z=g(x,\epsilon)$$. This re-parameterization converts the undifferentiable random variable $$z$$, to a differentiable function of $$x$$ and a decoupled source of randomness. Therefore, using this re-parameterization, we can estimate the gradient of the ELBO as $$\delta E_\epsilon [\delta -\log\ q(g(x,\epsilon) \vert x) + \delta p(x,g(x,\epsilon))]$$. This estimate to the gradient has been empirically shown to have much less variance and is called "Stochastic Gradient Variational Bayes (SGVB)". The SGVB is also called a black-box inference method (similar to MCMC estimate of the gradient) which simply means it doesn't care what functions we use in the generative and inference network as long as we can calculate the gradient at samples of $$\epsilon$$. We can use SGVB with a separate set of parameters for each observation however that's costly and inefficient. We usually choose to "amortize" the inference with deep networks (to learn a single complex function for all observation to latent mappings). All the terms of the ELBO are differentiable now if we choose deep networks as our likelihood and approximate posterior functions. Therefore, we have an end-to-end differentiable model. Following depiction shows amortized SGVB re-parameterization in a VAE.

<img src="/images/VAE_intuitions/vae_structure.jpg" alt="Simple VAE structure with reparameterization" width="350" height="350">


#### Stochastic optimization in Variational inference/learning

To optimize the ELBO, Traditional VI uses coordinate ascent which iteratively update each parameter, holding others fixed. Classical VI is inefficient since they do some local computation for each data point. Aggregate these computations to re-estimate global structure and Repeat. In particular, variational inference in a typical model where a local latent variable is introduced for every observation (z->x) would involve introducing variational distributions for each observation, but doing so would require a lot of parameters, growing linearly with observations. Furthermore, we could not quickly infer x given a previously unseen observation. We will therefore perform amortised inference where we introduce an inference network for all observations instead.

- Stochastic variational inference (SVI) scales VI to massive data. Additionally, SVI enables VI on a wide class of difficult models and enable VI with elaborate and flexible families of approximations. Stochastic Optimization replaces the gradient with cheaper noisy estimates and is guaranteed to converge to a local optimum. Example is SGD where the gradient is replaced with the gradient of a stochastic sample batch. The variational inferene recipe is:
1. Start with a model
2. Choose a variational approximation (variational family)
3. Write down the ELBO and compute the expectation (integral). 
4. Take ELBO derivative 
5. Optimize using the SGD update rule

- We usually get stuck in step 3, calculating the expectation (integral) since it's intractable. However, what we need for optimization is actually the gradient of the ELBO not the ELBO itself. We refer to black box variational Inference to compute ELBO gradients without calculating its expectation. The way it works is to combine steps 3 and 4 above to calculate the gradient of expectation in a single step using variational methods instead of exact method of 3 then 4. 

- This is a general problem in optimization where we want to evaluate the gradient of expectation of a parameterized function, $$f_\theta$$, w.r.t. a parameterized distribution, $$q_\psi$$. The gradient is with respect to the parameters of the distribution $$\nabla_\psi E_{q_\psi (z)}[f_\theta (z)] = \nabla \int q_\psi (z) f_\theta (z)$$. There are two main ways for evaluating the gradient depending on whether we differentiate the function (pathwise gradients) or the distribution density function (score function gradient).

- Score function gradient: The problem is to calculate the gradient of an expectation of a funtion $$ \nabla_\theta (E_q(z) [f(z)])=\nabla_\theta( \int q(z)f(z))$$ with respect to parameters $$\theta$$. The function here is ELBO but gradient is difficult to compute since the integral is unknown or the ELBO is not differentiable. To calculate the gradient, we first take the $$\nabla_\theta$$ inside the integral to rewrite it as $$\int \nabla_\theta(q(z)) f(z) dz$$ since only the $$q(z)$$ is a function of $$\theta$$. Then we use the log derivative trick (using the derivative of the logarithm $d (log(u))= d(u)/u$) on the (ELBO) and re-write the integral as an expectation $$\nabla_\theta (E_q(z) [f(z)]) = E_q(z) [\nabla_\theta \log q(z) f(z)]$$. This estimator now only needs the dervative $$\nabla \log q_\theta (z)$$ to estimate the gradient. The expectation will be replaced with a Monte Carlo Average. When the function we want derivative of is log likelihood, we call the derivative $\nabla_\theta \log ⁡p(x;\theta)$ a score function. The expected value of the score function is zero.[](http://blog.shakirm.com/2015/11/machine-learning-trick-of-the-day-5-log-derivative-trick/)

- The form after applying the log-derivative trick is called the score ratio. This gradient is also called REINFORCE gradient or likelihood ratio. We can then obtain noisy unbiased estimation of this gradients with Monte Carlo. To compute the noisy gradient of the ELBO we sample from variational approximate q(z;v), evaluate gradient of log q(z;v), and then evaluate the log p(x, z) and log q(z). Therefore there is no model specific work and and this is called black box inference. The problem with this approach is that sampling rare values can lead to large scores and thus high variance for gradient. There are a few methods that help with reducing the variance but with a few more non-restricting assumptions we can find a better method with low variance i.e. pathwise gradients. 

- Pathwise Gradients of the ELBO: This method has two more assumptions, the first is assuming the hidden random variable can be reparameterized to represent the random variable z as a function of deterministic variational parameters $v$ and a random variable $\epsilon$, $z=f(\epsilon, v)$. The second is that log p(x, z) and log q(z) are differentiable with respect to z. With reparameterization trick, this amounts to a differentiable deterministic variational function. This method generally has a better behaving variance. 

In summary, for variational inference, if log p(x, z) is z-differentiable:
- Try out an approximation q that is reparameterizable (end to end differentiable model)
- If log p(x, z) is not z differentiable:
    + Use score function estimator with control variates
    + Add further variance reductions based on experimental evidence

### Parameterization using deep nets and amortizing inference

- we represent the conditional distribution with a deep neural network $$p_\theta (x|z) = N(DNN_\theta (x))$$ to enable arbitarily complex distribution on $$X$$.

- Amortised inference: Pathwise gradients would need to estimate a value for each data sample in the training. The basic idea of amortised inference is to learn a mapping from data to variational parameters to remove the computational cost of calculation for every data point. In stochastic variation inference, after random sampling, setting local parameters also involves an intractable expectation which should be calculated with another stochastic optimization for each data point. This is also the case in classic inference like an EM algorithm, the learned distribution/parameters in the E-step is forgotten after the M-step update. In amortised inference, an inference network/tree/basis function might be used as the inference function from data to variational parameters to amortise the cost of inference. So in a sense, this way there is some sort of memory in the inferene and the learned params are not forgotten each time but rather updated in each step and thus they amortise the cost of inference. Amortized inference is faster, but admits a smaller class of approximations, the size of which depends on the flexibility of f.


## Advanced ideas in VI (Auxiliary Variational Method, Variational Inference with Normalizing Flows, Hierarchical Variational Models, Auxiliary Deep Generative Models):


### Limitations of mean-field VI (Gaussian posterior)
Mean-field or fully-factorised approximate posterior distributions q(z|x) is usually not sufficient for modelling real world data (Complex dependencies, Non-Gaussian distributions, Multiple modes) but traditionally we've been restricted to them due to limitations in solving the optimization problem. This is the same challenge encountered in the problem of specifying a model of the data (prior) itself which was restricted by our limitations in solving the inference problem (e.g. conjugate priors).

However, advances in probabilistic modelling (probabilistic programming), scalable inference through stochastic optimization, and blackbox variational inference (non-conjugate models, Monte Carlo gradient estimators, and amortised inference) have recently enabled us to design more flexible models and approximate posterior distributions q(z|x) for our models. So the goal is to build richer approximate posterior distributions and maintain the computational efficiency and scalability. A few examples follow.

- In Gaussian Approximate Posteriors, Structure of covariance $\sum$ describes dependency. Mean field assumption (i.e. diagonal covariance matrix as in a VAE) is least expressive but computatoinally efficient. Full covariance is richest, but computationally expensive. There are ideas around using linear algebra to efficiently decompose the covariance matrix, for example using a tri-diagonal covariance for an LDS approximate posterior. A limitation of Gaussian variational distributions is that the posterior is always Gaussian which doesn't always reflect real world distributions. 

### Designing richer approximate posteriors
- In order to design richer approximate posteriors, we can introduce new variables that help to form a richer approximate posterior (where q(z|x) is the marginalization of those new variables). We need to be able to adapt bound (ELBO) to compute entropy or a bound, and maintain computational efficiency to be linear in number of latent variables. There are two main approaches for doing this. First is change-of-variables including Normalising flows and invertible transforms. The second is auxiliary variables which involves Entropy bounds, and Monte Carlo sampling.

#### transformations (normalizaing, autoregressive and NVP flows)
In approximations using Change-of-variables, the distribution flows through a sequence of **deterministic invertible** transforms. We begin with an initial distribution q0(z0|x) and apply a sequence of K invertible functions $$f_k$$ that will reshape the posterior to more complex shapes. We employ a class of transformations for which the determinant of the Jacobian can be computed in linear time.

We Begin with a fully-factorised Gaussian and improve by change of variables. Triangular Jacobians allow for computational efficiency. These include Planar flow, real NVP, and inverse autoregressive flow. These models have a linear time computation of the determinant and its gradient. 

$$\log q_K(z_K) =  \log q_0(z_K) - \prod_i ^ K \log(J_i(z_{i-1}))^-1$$

Replacing the above into the ELBO for the new approximate posterior gives the ELBO for the transformed posterior:

$$E_q0 [\log p(x,z_K)] - E_q0[\log q_0(z_0)] + E_q0 [\sum \log(J_i(Z_{i-1}))]$$

- Autoregressive posterior distributions impose an ordering and non-linear dependency on all preceding hidden variables. Therfore the Joint-distribution is non-Gaussian and very expressive, although conditional distributions are Gaussian. For example compare Gaussian mean field posterior (VAE) vs. auto-regressive posterior (DRAW).

#### hierarchical approximate posteriors
Another possible approach is using hierarchical approximate posteriors. The basic idea is to use a hierarchical model for the approximate posterior (We can use latent variables, like with density models). In that case, the new variables are stochastic variables rather than deterministic in the change-of-variables approach and both continuous and discrete latent variables can be modelled.

The Auxiliary-variable methods add exra latent variables in parallel to existing hiddens and don't change the original model. They capture structure of correlated variables because they turn the posterior into a mixture of distributions q(z|x, a). The richer posterior can be a mixture model, normalising flow, Gaussian process, etc but needs to have easy sampling for evaluation of bound and gradients. 

- More Structured posteriors introduce additional variables that induce dependencies, but that remain tractable and efficient for example and LDS/HMM/etc assumption on the posterior in an SVAE.


#### Variational inference from first principles (Operator variational inference)

Given a model of latent and observed variables p(x, z), variational inference posits a family of distributions over its latent variables and then finds the member of that family closest to the posterior, p(z | x). This is typically formalized as minimizing a Kullback-Leibler (kl) divergence from the approximating family q(·) to the posterior p(·). However, while the kl(q k p) objective offers many beneficial computational properties, it is ultimately designed for convenience; it sacrifices many desirable statistical properties of the resultant approximation. When optimizing kl, there are two issues with the posterior approximation that we highlight. First, it typically underestimates the variance of the posterior. Second, it can result in degenerate solutions that zero out the probability of certain configurations of the latent variables. While both of these issues can be partially circumvented by using more expressive approximating families, they ultimately stem from the choice of the objective. Under the kl divergence, we pay a large price when q(·) is big where p(·) is tiny; this price becomes infinite when q(·) has larger support than p(·).

In "Operator Variational Inference", authors revisit variational inference from its core principle as an optimization problem. We use operators—mappings from functions to functions—to design variational objectives, explicitly trading off computational properties of the optimization with statistical properties of the approximation. We use operators to formalize the basic properties needed for variational inference algorithms. We further outline how to use them to define new variational objectives; as one example, we design a variational objective using a Langevin-Stein operator. the Langevin-Stein objective enjoys two good properties. First, it is amenable to data subsampling, which allows inference to scale to massive data. Second, it permits rich approximating families, called variational programs, which do not require analytically tractable densities. This greatly expands the class of variational families and the fidelity
of the resulting approximation.


# To read

[1] Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. In International Conference on Learning Representations.

[2] Rezende, D. J., Mohamed, S., & Wierstra, D. (2014). Stochastic Backpropagation and Approximate Inference in Deep Generative Models. In International Conference on Machine Learning.

[3] Ranganath, R., Gerrish, S., & Blei, D. M. (2014). Black Box Variational Inference. In Artificial Intelligence and Statistics.

[4] Mnih, A., & Gregor, K. (2014). Neural Variational Inference and Learning in Belief Networks. In International Conference on Machine Learning.

[5] Rezende, D. J., & Mohamed, S. (2015). Variational Inference with Normalizing Flows. In International Conference on Machine Learning.

[6] Salimans, T., Kingma, D. P., & Welling, M. (2015). Markov Chain Monte Carlo and Variational Inference: Bridging the Gap. In International Conference on Machine Learning.

[7] Tran, D., Ranganath, R., & Blei, D. M. (2016). The Variational Gaussian Process. In International Conference on Learning Representations.

[8] Ranganath, R., Tran, D., & Blei, D. M. (2016). Hierarchical Variational Models. In International Conference on Machine Learning.

[9] Maaløe, L., Sønderby, C. K., Sønderby, S. K., & Winther, O. (2016). Auxiliary Deep Generative Models. In International Conference on Machine Learning.

[10] Johnson, M. J., Duvenaud, D., Wiltschko, A. B., Datta, S. R., & Adams, R. P. (2016). Composing graphical models with neural networks for structured representations and fast inference. In Neural Information Processing Systems.

[11] Ranganath, R., Altosaar, J., Tran, D., & Blei, D. M. (2016). Operator Variational Inference. In Neural Information Processing Systems.

[12] Gelman, A., Vehtari, A., Jylänki, P., Sivula, T., Tran, D., Sahai, S., … Robert, C. (2017). Expectation propagation as a way of life: A framework for Bayesian inference on partitioned data. ArXiv.org.