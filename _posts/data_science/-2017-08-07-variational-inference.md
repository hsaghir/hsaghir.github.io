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


-  The gap in the Jensen’s inequality is exactly the KL divergence. Therefore minimizing the Kullback-Leibler (KL) divergence is equivalent to maximizing the ELBO.

The ELBO is a lower bound to the marginal or model evidence. The first term measures how well samples from q(z|x) are able to explain the data x (Reconstruction cost). The second term ensures that the explanation of the data q(z|x) doesn’t deviate too far from prior beliefs p(z) which is a mechanism for realising Ockham’s razor (Regularization).

- In mean-field variational family, the latent variables are mutually independent and each latent variable $$z_j$$ is governed by its own variational factor, the density $$q_j(z_j)$$ in the total variational density $$q(z) = \prod_j q_j(z_j)$$. One way to expand the family is to add dependencies between the variables. this is called structured variational inference. Another way to expand the family is to consider mixtures of variational densities, i.e., additional latent variables within the variational family.


#### ELBO derivation

- The ELBO is determined from introducing a variational distribution, $$q$$, to the marginal log likelihood, i.e. $$\log \ p(x)=\log \int_z p(x,z) * \frac{q(z|x)}{q(z|x)}$$. We use the log likelihood to be able to use the concavity of the $$\log$$ function and employ Jensen's equation to move the $$\log$$ inside the integral i.e. $$\log \ p(x) > \int_z \log\ (p(x,z) * \frac{q(z|x)}{q(z|x)})$$ and then use the definition of expectation on $$q$$ (the nominator $$q$$ goes into the definition of the expectation on $$q$$ to write that as the ELBO) $$\log \ p(x) > ELBO(z) = E_q [- \log\ q(z|x) + \log \ p(x,z)]$$. The difference between the ELBO and the marginal $$p(x)$$ which converts the inequality to an equality is the distance between the real posterior and the approximate posterior i.e. $$KL[q(z|x)\ | \ p(z|x)]$$. Alternatively, the distance between the ELBO and the KL term is the log-normalizer $$p(x)$$. Replace the $$p(z|x)$$ with Bayesian formula to see how. 

Now that we have a defined a loss function, we need the gradient of the loss function, $$\delta E_q[-\log q(z \vert x)+p(x,z)]$$ to be able to use it for optimization with SGD. The gradient isn't easy to derive analytically but we can estimate it using MCMC to directly sample from $$q(z \vert x)$$ and estimate the gradient. This approach generally exhibits large variance since MCMC might sample from rare values.

This is where the re-parameterization trick we discussed above comes in. We assume that the random variable $$z$$ is a deterministic function of $$x$$ and a known $$\epsilon$$ ($$\epsilon$$ are iid samples from unit Gaussian) that injects randomness $$z=g(x,\epsilon)$$. This re-parameterization converts the undifferentiable random variable $$z$$, to a differentiable function of $$x$$ and a decoupled source of randomness. Therefore, using this re-parameterization, we can estimate the gradient of the ELBO as $$\delta E_\epsilon [\delta -\log\ q(g(x,\epsilon) \vert x) + \delta p(x,g(x,\epsilon))]$$. This estimate to the gradient has been empirically shown to have much less variance and is called "Stochastic Gradient Variational Bayes (SGVB)". The SGVB is a black-box inference method (similar to REINFORCE estimate of the gradient) which simply means it doesn't care what functions we use in the generative and inference network as long as we can calculate the gradient at samples of $$\epsilon$$. We can use SGVB with a separate set of parameters for each observation however that's costly and inefficient. We usually choose to "amortize" the inference with deep networks (i.e. learn a single complex function for all observation to latent mappings). If we choose deep networks as our likelihood and approximate posterior functions, all terms of the ELBO will be differentiable. Therefore, we have an end-to-end differentiable model. The following depiction shows amortized SGVB re-parameterization in a VAE.

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
  If log p(x, z) is not z differentiable:
- Use score function estimator with control variates
- Add further variance reductions based on experimental evidence


##### Reparameterization trick
The reparameterization trick helps with a low variance estimate of the gradient for continuous variables. However, encounters problems for discrete and categorical variables since we cannot backpropagate gradients through discrete nodes in the computational graph. The Gumbel-softmax trick helps in the reparameterization of categorical variables:
- a nice reparameterization for a discrete (or categorical) variable comes from the Gumbel distribution. The reparameterized categorical variable is a function of a sample from the Gumbel distribution i.e. $$y \approx \argmax_k (\log \alpha_k + G_k)$$. 
    + Draw a sample noise from Gumble distribution by transforming a sample from the uniform distribution i.e. $$G = -\log(-\log(U))$$ where $$U \approx Unif[0,1]$$
    + add it to category weights $$\log \alpha_k$$ 
    + take the value that produces the maximum as a categorical sample
- Althought the resulting reparameterized function is not continuous, it can be approximated with the softmax function indexed with a temperature parameter. 
    + apply softmax to the produced values from Gumble instead of just taking the maximum. 

### Parameterization using deep nets and amortizing inference

- we represent the conditional distribution with a deep neural network $$p_\theta (x|z) = N(DNN_\theta (x))$$ to enable arbitarily complex distribution on $$X$$. 

- To optimize the ELBO, Traditional VI uses coordinate ascent which iteratively update each parameter, holding others fixed. Classical VI is inefficient since they do some local computation for each data point. Aggregate these computations to re-estimate global structure and Repeat. In particular, variational inference in a typical model where a local latent variable is introduced for every observation (z->x) would involve introducing variational distributions for each observation, but doing so would require a lot of parameters, growing linearly with observations. Furthermore, we could not quickly infer x given a previously unseen observation. We will therefore perform amortised inference where we introduce an inference network for all observations instead.

- Amortised inference: Pathwise gradients would need to estimate a value for each data sample in the training. The basic idea of amortised inference is to learn a mapping from data to variational parameters to remove the computational cost of calculation for every data point. In stochastic variation inference, after random sampling, setting local parameters also involves an intractable expectation which should be calculated with another stochastic optimization for each data point. This is also the case in classic inference like an EM algorithm, the learned distribution/parameters in the E-step is forgotten after the M-step update. In amortised inference, an inference network/tree/basis function might be used as the inference function from data to variational parameters to amortise the cost of inference. So in a sense, this way there is some sort of memory in the inferene and the learned params are not forgotten each time but rather updated in each step and thus they amortise the cost of inference. Amortized inference is faster, but admits a smaller class of approximations, the size of which depends on the flexibility of f.


## Advanced ideas in VI:
Improvements in building deep latent-variable generative models have primarily focused on deriving better algorithms and objectives for training that go beyond the ELBO, and identifying more powerful and flexible architectures. 
- On the algorithmic side, 
    + (1) new objectives have been presented to speed up training, learn better features, and combat some of the deficiencies of the ELBO.
        * [Importance weighted autoencoders](Burda et al., 2015),
        * [Renyi divergence variational inference](Li & Turner, 2016),
        * [Operator variational inference](Ranganath et al., 2016a)
        * [Stein variational gradient] (Liu and Wang, 2016)
    + (2) adversarial algorithms for variational inference(http://www.inference.vc/variational-inference-using-implicit-models/)
        * [prior contrastive, Adversarial autoencoders, Unifying VAE & GAN ](Makhzani & Frey 2016, Mescheder et al. 2017)
        * [joint contrastive, ALI, BiGAN](Dumoulin et al, 2016, Donahue et al, 2016)
        * [both above] (Karaletsos, 2016, Mohamed & Lakshminarayanan, 2016, Huszar F. 2017).
        * [Using denoisers instead of discriminator](http://www.inference.vc/variational-inference-using-implicit-models-part-iv-denoisers-instead-of-discriminators/)
        * [implicit p and q, Deep hierarchical implicit models](http://dustintran.com/blog/deep-and-hierarchical-implicit-models)
- On the architectural side, there are three directions of focus: 
    + (1) more complex encoders that extend the variational family 
        * [normalizing flows](Rezende & SMohamed, 2015),
        * [autoregressive flows](Kingma et al., 2016),
        * [Hierarchical variational models](Ranganath et al., 2016b), etc
    + (2) more complex priors:
        * [Vae with a vampprior] (Tomczak & Welling, 2017), 
        * [NADE](Larochelle & Murray, 2011b), 
        * [DGM with stick-breaking priors](Nalisnick & Smyth, 2016)
    + (3) more complex decoders: 
        * [Variational lossy autoencoder](Chen et al., 2016),
        * [PixelVAE](Gulrajani et al., 2016),
        * [PixelGAN autoencoders](Makhzani & Frey, 2017)
- On the theory side,
    + (1) Information theoretic analysis:
        * [amount of information that the latent variable contains about the input](Alexander A. Alemi et al 2017)

These extensions have allowed us to scale VI to many exciting applications, but it remains unclear how these different pieces of complexity fit together and interact.

### Limitations of mean-field VI (Gaussian posterior)
- mean field assumption for the approximate posterior of a vanilla VAE means that we are assuming a multivariate Guassian shape for the approximate posterior with a diagonal covariance matrix (a hyper-sphere). meaning that we are assuming dimensions of the hyper-sphere are completely independent of each other. Even an LDS still has a multivariate Gaussian approximate posterior but now with tri-diagonal covariance matrix (a hyper-elipse).

- Factorization assumptions on the posterior have two parts. First we might assume that the posterior for a single data point has certain factorization structure. Then we usually assume that the posterior for each data point $$q(z_i|x_i)$$ is independant of the posterior for the next data point $$q(z_{i+1}|x_{i+1})$$. This second factorization assumption is usually refered to as mean field assumption which results in the approximate posterior being $$q(Z|X) = \prod_i q(z_i|x_i)$$. Even in a deep LDS or a hidden markov model, there are still mean field factorization assumptions on blocks of data point posteriors i.e. $$q(Z|X) = \prod_i q(z_i, z_{i+1}|x_i, x_{i+1})$$. Due to these factorization assumptions, we can usually break the expectation on the approximate posterior in the ELBO down to sample-wise averages.

Mean-field or fully-factorised approximate posterior distributions q(z|x) is usually not sufficient for modelling real world data (Complex dependencies, Non-Gaussian distributions, Multiple modes) but traditionally we've been restricted to them due to limitations in solving the optimization problem. This is the same challenge encountered in the problem of specifying a model of the data (prior) itself which was restricted by our limitations in solving the inference problem (e.g. conjugate priors).

However, advances in probabilistic modelling (probabilistic programming), scalable inference through stochastic optimization, and blackbox variational inference (non-conjugate models, Monte Carlo gradient estimators, and amortised inference) have recently enabled us to design more flexible models and approximate posterior distributions q(z|x) for our models. So the goal is to build richer approximate posterior distributions and maintain the computational efficiency and scalability. A few examples follow.

- In Gaussian Approximate Posteriors, Structure of covariance $\sum$ describes dependency. Mean field assumption (i.e. diagonal covariance matrix as in a VAE) is least expressive but computatoinally efficient. Full covariance is richest, but computationally expensive. There are ideas around using linear algebra to efficiently decompose the covariance matrix, for example using a tri-diagonal covariance for an LDS approximate posterior. A limitation of Gaussian variational distributions is that the posterior is always Gaussian which doesn't always reflect real world distributions. 

### Adversarial training in VI

there are two ways implicit models (following the definition above) can be used in variational inference, and which can be important to distinguish.

The first is as you describe in the example of Bayesian logistic regression, where the variational distribution is an implicit model. This enables the most expressive posterior approximation in the sense that q no longer requires a tractable density (which is a silly requirement but appears again and again for historical and ease of optimization reasons). We pursued it in Operator Variational Inference (https://arxiv.org/abs/1610.... ), termed a "variational program". Others including Qiang Liu, Dilin Wang, Jason Lee, and Yingzhen Li are currently pursuing this in the context of Stein variational gradient descent, and there's also the fantastic workshop paper by Theo you referenced above.

The second is inference when the probability model (p) is implicit. This is related to the papers you mentioned above on adversarial auto encoders and BiGANs/ALI, both of which do a form of latent variable inference beyond point estimation (although not necessarily posterior inference). There's also recent works on variational inference + ABC which are very relevant, such as VI with intractable likelihood (https://arxiv.org/abs/1503.... ) and automatic variational ABC (https://arxiv.org/abs/1606.... ).

A natural idea of course is to consider algorithms that enable [both implicit p and implicit q](http://dustintran.com/blog/deep-and-hierarchical-implicit-models). This is something that we're working on, hopefully to be done by ICML Like you, I've also found that thinking from the perspective of ratio estimation, and beyond the usual GAN setup, to be very insightful.

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

The Auxiliary-variable methods add exra latent variables in parallel to existing hiddens and don't change the original model. They capture structure of correlated variables because they turn the posterior into a mixture of distributions q(z|x, a). The richer posterior can be a mixture model, normalising flow, Gaussian process, etc but needs to have easy sampling for evaluation of bound and gradients. 

- More Structured posteriors introduce additional variables that induce dependencies, but that remain tractable and efficient for example and LDS/HMM/etc assumption on the posterior in an SVAE.


#### Variational inference from first principles (Operator variational inference)

Given a model of latent and observed variables p(x, z), variational inference posits a family of distributions over its latent variables and then finds the member of that family closest to the posterior, p(z | x). This is typically formalized as minimizing a Kullback-Leibler (kl) divergence from the approximating family q(·) to the posterior p(·). However, while the kl(q k p) objective offers many beneficial computational properties, it is ultimately designed for convenience; it sacrifices many desirable statistical properties of the resultant approximation. When optimizing kl, there are two issues with the posterior approximation that we highlight. First, it typically underestimates the variance of the posterior. Second, it can result in degenerate solutions that zero out the probability of certain configurations of the latent variables. While both of these issues can be partially circumvented by using more expressive approximating families, they ultimately stem from the choice of the objective. Under the kl divergence, we pay a large price when q(·) is big where p(·) is tiny; this price becomes infinite when q(·) has larger support than p(·).

In "Operator Variational Inference", authors revisit variational inference from its core principle as an optimization problem. We use operators—mappings from functions to functions—to design variational objectives, explicitly trading off computational properties of the optimization with statistical properties of the approximation. We use operators to formalize the basic properties needed for variational inference algorithms. We further outline how to use them to define new variational objectives; as one example, we design a variational objective using a Langevin-Stein operator. the Langevin-Stein objective enjoys two good properties. First, it is amenable to data subsampling, which allows inference to scale to massive data. Second, it permits rich approximating families, called variational programs, which do not require analytically tractable densities. This greatly expands the class of variational families and the fidelity of the resulting approximation.



### Variational inference for Bayesian Neural Networks



### Information theory perspective:

Goal is to learn a representation that contains a particular amount of information and from which the input can be reconstructed as well as possible. A good representation $$Z$$ must contain information about the input $$X$$. The mutual information is:

$$I(X;Z)=\mathbb {E} _{X,Z}[SI(x,Z)]=\sum _{x,z}p(x,z)\log {\frac {p(x,z)}{p(x)\,p(z)}}$$

This is representational mutual information, which is distinguishable from the generative mutual information. Unfortunately, this is hard to compute, since we do not have access to the true data density $$p(x)$$, and computing the marginal $$p(z)=\sum_x p(x,z)$$ can
be challenging. However, the following tractable lower and upper bounds can be derived as $$H-D<I<R$$ where R is rate and D is distortion. H here is the data complexity and a theoretical limit. We cannot use less than H nats to communicate data X in anyway. When $$R=0$$, then mutual information of representation and data is zero ($$I=0$$) and $$Z$$ doesn't know anything about $$X$$. This is the case where encoder hasn't encoded any information into $$Z$$. However, even without learning a representation $$Z$$, decoder alone can reduce distortion using an autoregressive setting upto the theoretical limit of$$ H$$. When $$D=0$$, we have a zero distortion setting, we can perfectly encode and decode our data; where lowest possible rate is  $$H$$, the entropy of the data. We can use a less efficient code $$Z$$ to get the same reconstruction performance which means higher rates at fixed distortion.

This reminds me of CorEx where if we have $$n$$ discrete random variabels $${X_1,...,X_n}$$, total correlation (multivariate mutual information) is defined as the sum of mutual informations between all random variables $$TC(X_G) =  \sum_i H(X_i) - H(X_G)$$. If the joint distribution $$P(X_1,..X_n)$$ factorizes the total correlation is zero therfore, total correlation can be expressed as the KL divergence between the real joint and the factorized joint $$TC(X_G) =  KL(p(X_G) || \prod_i p(x_i))$$. 

We then introduce latent variables $$Y$$. The total correaltion among the observed group of variables, $$X$$, after condition on $$Y$$ is simply $$TC(X|Y) = \sum_i H(X_i|Y) - H(X|Y)$$. Therefore, the extent to which $$Y$$ explains the total correlation in $$X$$ can be measured by looking at how much the total correlation is reduced after introducing $$Y$$ i.e. the difference between the total correlation in $$X$$ and and total correlation in $$(X|Y)$$ i.e. $$TC(X;Y) = TC(X) - TC(X|Y) = \sum_i I(X_i : Y) - I(X : Y)$$. This difference forms an objective function that we can optimize to find the latent factors $$Y$$ that best explain the correlations in $$X$$. The bigger this objective, the more $$Y$$ explains correlation in $$X$$. Note that this objective is not symmetric. 

Since the total correlation depends on the joint distribution $$p(X,Z)$$ and by extension on $$P(X)$$. If we have $$n$$ binary $${0,1}$$ variables, then the search over all $$P(Z|X)$$ involves $$2^n$$ variables which is intractable. The paper introduces some restrictions on the objective to make the optimization tractable. The VAE framework can be combined with CorEx here. 



# References

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