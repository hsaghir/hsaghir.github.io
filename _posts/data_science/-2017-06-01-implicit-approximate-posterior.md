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
 

## Variational Inference

-  Inference can be viewed as maximizing the ELBO with respect to $$q$$, and learning can be viewed as maximizing the ELBO with respect to parameters.

- The core idea behind variational learning is that we maximize the ELBO over a restricted family of distributions such that the expectation $$E_q[log p(x, z)]$$ in the ELBO is computable. This way we are basically replacing the real posterior with an approximation. A typical way to do this is to use a factorized $$q$$ (for example mean field for iid latents or a structured graph for an LDS).

- This usually severely limits the power and expressivity of approximate posteriors to factorised Gaussians. We would like to use something more powerful, for example an implicit probabilistic model instead. 

- The beauty of the variational approach is that we do not need to specify a specific parametric form for $$q$$. We specify how it should factorize, but then the optimization problem determines the optimal probability distribution within those factorization constraints. For continuous latent variables, this means that we use a branch of mathematics called calculus of variations to perform optimization over a space of functions, and actually determine which function should be used to represent $$q$$.

- Maximizing the ELBO is equivalent to minimizing the $$KL[q|p]$$. In maximum likelihood learning, we minimize KL(p_data|p_model) which is the opposite direction. KL[q|p] is chosen for computational reasons however the chosen direction has implication:

- It's useful to think about the distance measure we talked about. KL-divergence measures a sort of distance between two distributions but it's not a true distance since it's not symmetric  $$KL(P|Q) = E_P[\log\ P(x) − \log\ Q(x)]$$. So which distance direction we choose to minimize has consequences. For example, in minimizing $$KL(p|q)$$, we select a $$q$$ that has high probability where $$p$$ has high probability so when $$p$$ has multiple modes, $$q$$ chooses to blur the modes together, in order to put high probability mass on all of them. 

On the other hand, in minimizing $$KL(q \vert p)$$, we select a $$q$$ that has low probability where $$p$$ has low probability. When $$p$$ has multiple modes that are sufficiently widely separated, the KL divergence is minimized by choosing a single mode (mode collapsing), in order to avoid putting probability mass in the low-probability areas between modes of $$p$$. In VAEs we actually minimize $$KL(q \vert p)$$ so mode collapsing is a common problem. Additionally, due to the complexity of true distributions we also see blurring problem. These two cases are illustrated in the following figure from the [deep learning book](http://www.deeplearningbook.org/).

### Discrete Latent Variables



### Calculus of Variation


### Continuous Latent Variables

### Interactions between Learning and Inference

## Learned approximate inference

### Wake-Sleep

### Other forms of learned inference

