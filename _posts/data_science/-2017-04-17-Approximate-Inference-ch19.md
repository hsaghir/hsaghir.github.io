---
layout: article
title: Approximate Inference - Ch19
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---


- A model usually consists of two sets of visible and hidden variables. Inference is the problem of computing $$p(h|v)$$ or taking an expectation on it. This is usually necessary for maximum likelihood learning. 

- In some graphical models with a particular structure like RBM and pPCA, where there is only a single hidden layer, inference is easy meaning that $$p(h|v)$$ can be calculated by multiplying the visible variables by the weight matrix. 

- Intractable inference (too computationally hard to calculate, omits the gains of having a probabilistic graphical model) in DL usually arises from interaction between latent variables in a structured graphical model. It may be due to direct interactions between hidden variables in an undirected graph (which produces large cliques of latent vars) or the "explaining away" effect in directed graphs. 

- We bypass the intractable exact calculation of the normalizer by approximating the partition function as explained in chapter 18. An alternative approach to solving the inference problem involves direct approximation of the posterior (i.e. variational inference). We start with the definition of the data distribution $$p(x)=\int_z p(x,z)$$ and introduce a family of distributions, $$q$$, on hidden variables to act as an approximation for the posterior. 
- We write down the ELBO using the concavity of the log function. Using the Jensen's inequality, we replace the equality with an inequality $$ \log p(x) > E_q[log p(x, Z)] − E_q[log q(Z)]$$ and choose a family of variational distributions such that the expectations are computable. For example, in mean field variational inference, we assume that each variable is independent and that the variational family factorizes and expectations are computable. In real posteriors however, variables are not independent. These dependencies are often what makes the posterior difficult to work with.

- Then, we maximize the ELBO to find the parameters that give as tight a bound as possible on the marginal probability of x. By maximizing the lower bound (ELBO), we effectively bypass the calculation of the normalizer for the computation of the posterior. 

- Different forms of approximate inference use different approximate optimization methods to find the best $$q$$. We can make the optimization procedure less expensive by restricting the family of distributions $$q$$ or by using an imperfect optimization procedure that may not completely maximize the ELBO but merely increase it by a significant amount. Also the divergence metric we use to measure the distance between an initial distribution and the desired distribution is important. For example, KL divergence connects to variational inference while other divergences connect to other approximate inference techniques like belief propagation, expectation propagation, etc. 

- The ELBO is determined from introducing a variational distribution $$q$$, on lower bound on the marginal log likelihood, i.e. $$\log \ p(x)=\log \int_z p(x,z) * \frac{q(z|x)}{q(z|x)}$$. We use the log-likelihood to be able to use the concavity of the $$\log$$ function and employ Jensen's inequality to move the $$\log$$ inside the integral i.e. $$\log \ p(x) > \int_z \log\ (p(x,z) * \frac{q(z|x)}{q(z|x)})$$ and then use the definition of expectation on $$q$$ (the nominator $$q$$ goes into the definition of the expectation on $$q$$ to write that as the ELBO) $$\log \ p(x) > ELBO(z) = E_q [- \log\ q(z|x) + \log \ p(x,z)]$$. The difference between the ELBO and the marginal $$p(x)$$ which converts the inequality to an equality is the distance between the real posterior and the approximate posterior i.e. $$KL[q(z|x)\ | \ p(z|x)]$$. Or alternatively, the distance between the ELBO and the KL term is the log normalizer $$p(x)$$. Replace the $$p(z|x)$$ with Bayesian formula to see how. 

- Note that in the above derivation of the ELBO, the first term is the entropy of the variational posterior and second term is log of joint distribution. However we usually write joint distribution as $$p(x,z)=p(x|z)p(z)$$ to rewrite the ELBO as $$ E_q[\log\ p(x|z)+KL(q(z|x)\ | \ p(z))]$$. This derivation is much closer to the typical machine learning literature in deep networks. The first term is log likelihood (i.e. reconstruction cost) while the second term is KL divergence between the prior and the posterior (i.e a regularization term that won't allow posterior to deviate much from the prior). Also note that if we only use the first term as our cost function, the learning with correspond to maximum likelihood learning that does not include regularization and might over fit.

- Now that we have defined a loss function (ELBO), we need the gradient of the loss function, $$\delta E_q[-\log q(z \vert x)+p(x,z)]$$ to be able to use it for optimization with SGD. The gradient isn't easy to derive analytically but we can estimate it using MCMC to directly sample from $$q(z \vert x)$$ and estimate the gradient. This approach generally exhibits large variance since MCMC might sample from rare values. This is where the re-parameterization trick comes in and reduces the variance by decoupling the random and deterministic parts of the model and making it differentiable. 

## Expectation Maximization (EM)

- EM is not an approach to approximate inference, but rather an approach to learning that uses an approximate posterior. EM maximizes the ELBO to find parameters for latent variable models by introducing an approximate posterior $$q$$. 
- Algorithmically, EM repeatedly uses an estimate of parameters as an approximate posterior with which it calculates an update (by maximizing ELBO w.r.t. the parameters); Then it uses the update to calculate new estimates for parameters:

-  Formally, EM consists of two steps that are repeated till convergence:
    +  Expectation step: we set the approximate posterior $$q$$ to the value of the posterior $$p$$ at that beginning of that step i.e. $$q= p(h \vert x;\theta^0)$$. 
    +  Maximization step: We maximize the ELBO with respect to parameters using an optimization algorithm (for example calculate gradient and use SGD update rule). 
    +  Repeat!

- This is similar to coordinate ascent for maximizing the ELBO; in the expectation step we maximize the ELBO w.r.t. $$q$$ using averages as an approximation of the expectation. Then we fix $$q$$ and  maximize the ELBO w.r.t. parameters. Repeat!

- SGD on latent variable model can be seen as a special case of EM where the maximization step is taking a single gradient step. Other optimization algorithms such as the Newton method can take a larger M step all the way to to global maximum. 

- Even though the E-step involves exact inference, we can think of the EM  as using approximate inference in some sense. Specifically, the M-step assumes that the same value of $$q$$ can be used for all values of parameters. This will introduce a gap between the ELBO and the true posterior as the M-step moves further and further away from the value initial value of parameters used in the E-step. Fortunately, the E-step reduces the gap to zero again as we enter the loop for the next time.

- One insight of EM is that there is the basic structure of the learning process, in which we update the model parameters to improve the likelihood of a completed dataset, where all missing variables have their values provided by an estimate of the posterior distribution. This particular insight is not unique to the EM algorithm. For example, using gradient descent to maximize the log-likelihood also has this same property; the log-likelihood gradient computations require taking expectations with respect to the posterior distribution over the hidden units.

- Another insight of the EM algorithms is that we can continue to use the one value of $$q$$ even after an update to obtain larger M steps.

## Maximum A Posteriori (MAP) inference 
- MAP inference estimates the most likely value of missing values as point estimates instead of calculating their distributions. We can derive MAP estimate from the ELBO, if we restrict the approximate posterior family $$q$$ to a point (Dirac) distribution $$q~\delta(h-\mu)$$.
- MAP inference is commonly used in DL as a learning method and a feature extractor mostly in sparse coding models. In sparse coding, the inference problem faces an explaining away situation that limits the factorization of the posterior. This makes the inference intractable. MAP inference is used to overcome that. 

## Variational Inference

-  Inference can be viewed as maximizing the ELBO with respect to $$q$$, and learning can be viewed as maximizing the ELBO with respect to parameters.

- The core idea behind variational learning is that we maximize the ELBO over a restricted family of distributions such that the expectation $$E_q[log p(x, z)]$$ is computable. A typical way to do this is to introduce assumptions about how $$q$$ factorizes (for example mean field or a structured graph e.g. LDS).

- The beauty of the variational approach is that we do not need to specify a specific parametric form for $$q$$. We specify how it should factorize, but then the optimization problem determines the optimal probability distribution within those factorization constraints. For continuous latent variables, this means that we use a branch of mathematics called calculus of variations to perform optimization over a space of functions, and actually determine which function should be used to represent $$q$$.

- 

