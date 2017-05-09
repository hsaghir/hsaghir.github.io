---
layout: article
title: Approximate Inference - Ch19
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---


- A model usually consists of two sets of visible and hidden variables. Inference is the problem of computing $$p(h|v)$$ or taking an expectation on it. This is usually necessary for maximum likelihood learning. 

- In graphical models like RBM and pPCA, where there is only a single hidden layer, inference is easy. 

- Intractable inference (too computationally hard to calculate, omits the gains of having a probabilistic graphical model) in DL usually arises from interaction between latent variables in a structured graphical model. It may be due to direct interactions in an undirected graph (produces large cliques of latent vars) or the "explaining away" effect in directed graphs. 

- We can solve inference by bypassing the intractable exact inference in favor of an approximate that is tractable (i.e. variational inference). We convert the exact inference problem of calculating an intractable expectation to an optimization problem of maximizing a lower bound $$q$$ to get as close as possible to a distribution through the definition of the ELBO. 

- Different forms of approximate inference use different approximate optimization methods to find the best $$q$$. We can make the optimization procedure less expensive but approximate by restricting the family of distributions $$q$$ or by using an imperfect optimization procedure that may not completely maximize the ELBO but merely increase it by a significant amount. Also the divergence metric we use to measure the distance between an initial distribution and the desired distribution is important. For example, KL divergence connects to variational inference while other divergences connect to other approximate inference techniques like belief propagation, expectation propagation, etc. 

- The ELBO is determined from introducing a variational distribution $$q$$, on lower bound on the marginal log likelihood, i.e. $$\log \ p(x)=\log \int_z p(x,z) * \frac{q(z|x)}{q(z|x)}$$. We use the log-likelihood to be able to use the concavity of the $$\log$$ function and employ Jensen's inequality to move the $$\log$$ inside the integral i.e. $$\log \ p(x) > \int_z \log\ (p(x,z) * \frac{q(z|x)}{q(z|x)})$$ and then use the definition of expectation on $$q$$ (the nominator $$q$$ goes into the definition of the expectation on $$q$$ to write that as the ELBO) $$\log \ p(x) > ELBO(z) = E_q [- \log\ q(z|x) + \log \ p(x,z)]$$. The difference between the ELBO and the marginal $$p(x)$$ which converts the inequality to an equality is the distance between the real posterior and the approximate posterior i.e. $$KL[q(z|x)\ | \ p(z|x)]$$. Or alternatively, the distance between the ELBO and the KL term is the log normalizer $$p(x)$$. Replace the $$p(z|x)$$ with Bayesian formula to see how. 

- Note that in the above derivation of the ELBO, the first term is the entropy of the variational posterior and second term is log of joint distribution. However we usually write joint distribution as $$p(x,z)=p(x|z)p(z)$$ to rewrite the ELBO as $$ E_q[\log\ p(x|z)+KL(q(z|x)\ | \ p(z))]$$. This derivation is much closer to the typical machine learning literature in deep networks. The first term is log likelihood (i.e. reconstruction cost) while the second term is KL divergence between the prior and the posterior (i.e a regularization term that won't allow posterior to deviate much from the prior). Also note that if we only use the first term as our cost function, the learning with correspond to maximum likelihood learning that does not include regularization and might over fit.

- Now that we have a defined a loss function (ELBO), we need the gradient of the loss function, $$\delta E_q[-\log q(z \vert x)+p(x,z)]$$ to be able to use it for optimization with SGD. The gradient isn't easy to derive analytically but we can estimate it by using MCMC to directly sample from $$q(z \vert x)$$ and estimate the gradient. This approach generally exhibits large variance since MCMC might sample from rare values. This is where the re-parameterization trick comes in and reduces the variance by decoupling the random and deterministic parts of the model and making it differentiable. 

## Expectation Maximization (EM)

- Algorithmicly, EM is very similar to k-means. going back and forth between a soft clustering and computing the mean from clusters. 
















