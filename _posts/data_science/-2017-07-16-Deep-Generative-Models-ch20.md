---
layout: article
title: Deep Generative Models Ch. 20
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

# Variational Autoencoder:

- Directed latent variable models can represent complex distributions over data while deep neural nets can represent arbitarily complex functions. We can use DNNs to parameterize and represent conditional distributions and that's exactly what we do in a VAE.

![alt text](/images/VAE_intuitions/latent_var_model.png "a latent variable model")

## Problem setup:
-  Let's assume our data as $$X = {x_1,x_2,...,x_n}$$. Let's also assume a set of latent variables $$Z = {z_1, z_2,...,z_n}$$ and a simple latent variable generative model for the data. $$ Z ~ p(Z) = N(0,I) ;  X ~ p(X|Z) = N(\mu, \sigma^2)$$

![alt text](/images/VAE_intuitions/vae_latent_var_model.png "simple vae latent variable model")

- we represent the conditional distribution with a deep neural network $$p_\theta (x|z) = N(DNN_\theta (x))$$ to enable arbitarily complex distribution on $$X$$. 

- Since the posterior $$p(z|x)$$ is intractable in this model, we need to use approximate inference for latent variable inference and parameter learning. Three inference methods:
    + MAP inference: simple and fast but too biased and approximate
    + MCMC inference: assymptotically unbiased but it's expensive, hard to assess it's convergence and large variance in gradient estimation
    + Variational inference: fast, and low variance in gradient with reparameterization trick. well suited to the use of deep neural nets for parameterizing conditional distributions. Also well-suited to using deep neural networks to amortize the inference of local latent variables $${z_1,..., z_n}$$ with a single deep neural network. 

### Variational inference for above model:
- Variational Inference turns inference into optimization. Given a model of latent and observed variables $$p(X, Z)$$, variational inference posits a family of distributions over its latent variables and then finds the member of that family closest to the posterior, $$p(Z|X)$$. This is typically formalized as minimizing the Kullback-Leibler (KL) divergence from the approximating family $$q(·)$$ to the posterior $$p(·)$$.

- Stochastic variational inference (SVI) scales VI to massive data. Additionally, SVI enables VI on a wide class of difficult models and enable VI with elaborate and flexible families of approximations. Stochastic Optimization replaces the gradient with cheaper noisy estimates and is guaranteed to converge to a local optimum. Example is SGD where the gradient is replaced with the gradient of a stochastic sample batch. The variational inferene recipe is:
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

#### ELBO

- The ELBO is determined from introducing a variational distribution, $$q$$, to the marginal log likelihood, i.e. $$\log \ p(x)=\log \int_z p(x,z) * \frac{q(z|x)}{q(z|x)}$$. We use the log likelihood to be able to use the concavity of the $$\log$$ function and employ Jensen's equation to move the $$\log$$ inside the integral i.e. $$\log \ p(x) > \int_z \log\ (p(x,z) * \frac{q(z|x)}{q(z|x)})$$ and then use the definition of expectation on $$q$$ (the nominator $$q$$ goes into the definition of the expectation on $$q$$ to write that as the ELBO) $$\log \ p(x) > ELBO(z) = E_q [- \log\ q(z|x) + \log \ p(x,z)]$$. The difference between the ELBO and the marginal $$p(x)$$ which converts the inequality to an equality is the distance between the real posterior and the approximate posterior i.e. $$KL[q(z|x)\ | \ p(z|x)]$$. Alternatively, the distance between the ELBO and the KL term is the log-normalizer $$p(x)$$. Replace the $$p(z|x)$$ with Bayesian formula to see how. 

Now that we have a defined a loss function, we need the gradient of the loss function, $$\delta E_q[-\log q(z \vert x)+p(x,z)]$$ to be able to use it for optimization with SGD. The gradient isn't easy to derive analytically but we can estimate it using MCMC to directly sample from $$q(z \vert x)$$ and estimate the gradient. This approach generally exhibits large variance since MCMC might sample from rare values.

This is where the re-parameterization trick we discussed above comes in. We assume that the random variable $$z$$ is a deterministic function of $$x$$ and a known $$\epsilon$$ ($$\epsilon$$ are iid samples) that injects randomness $$z=g(x,\epsilon)$$. This re-parameterization converts the undifferentiable random variable $$z$$, to a differentiable function of $$x$$ and a decoupled source of randomness. Therefore, using this re-parameterization, we can estimate the gradient of the ELBO as $$\delta E_\epsilon [\delta -\log\ q(g(x,\epsilon) \vert x) + \delta p(x,g(x,\epsilon))]$$. This estimate to the gradient has been empirically shown to have much less variance and is called "Stochastic Gradient Variational Bayes (SGVB)". The SGVB is also called a black-box inference method (similar to MCMC estimate of the gradient) which simply means it doesn't care what functions we use in the generative and inference network as long as we can calculate the gradient at samples of $$\epsilon$$. We can use SGVB with a separate set of parameters for each observation however that's costly and inefficient. We usually choose to "amortize" the inference with deep networks (to learn a single complex function for all observation to latent mappings). All the terms of the ELBO are differentiable now if we choose deep networks as our likelihood and approximate posterior functions. Therefore, we have an end-to-end differentiable model. Following depiction shows amortized SGVB re-parameterization in a VAE.

<img src="/images/VAE_intuitions/vae_structure.jpg" alt="Simple VAE structure with reparameterization" width="350" height="350">

#### VAE Amortized inference
- To optimize the ELBO, Traditional VI uses coordinate ascent which iteratively update each parameter, holding others fixed. Classical VI is inefficient since they do some local computation for each data point. Aggregate these computations to re-estimate global structure and Repeat. In particular, variational inference in a typical model where a local latent variable is introduced for every observation (z->x) would involve introducing variational distributions for each observation, but doing so would require a lot of parameters, growing linearly with observations. Furthermore, we could not quickly infer x given a previously unseen observation. We will therefore perform amortised inference where we introduce an inference network for all observations instead.


#### VAE connection to autoencoders 

- In a VAE both the inference and generative networks are deep neural networks which is similar to a regular autoencoder. 
- Note that in the above derivation of the ELBO, the first term is the entropy of the variational posterior and second term is log of joint distribution. However we usually write joint distribution as $$p(x,z)=p(x|z)p(z)$$ to rewrite the ELBO as $$ E_q[\log\ p(x|z)+KL(q(z|x)\ | \ p(z))]$$. This derivation is much closer to the typical machine learning literature in deep networks. The first term is log likelihood (i.e. reconstruction cost) while the second term is KL divergence between the prior and the posterior (i.e a regularization term that won't allow posterior to deviate much from the prior). Also note that if we only use the first term as our cost function, the learning with correspond to maximum likelihood learning that does not include regularization and might overfit.


## Semi-Supervised learning with deep generative models
- Three approaches for semi-supervised learning with VAE models are proposed in the paper, i.e. M1, M2, (M1 + M2)
    + M1 is a simple VAE model for unsupervised feature learning. The learned features are used for training a separate classifier. Approximate samples from the posterior distribution over the latent variables p(z|x) are used as features to train a classifier that predicts class labels y from data in a lower dimensional space. 

<img src="/images/VAE_intuitions/vae_semi_M1.png" alt="semi-supervised model inference" width="350" height="350">

    + M2 is a probabilistic model that describes the data as being generated by a latent categorical class variable $$y$$ in addition to a continuous latent variable $$z$$. In this model, the VAE acts as a regularizer for the classifier. Therefore, The total ELBO will be the sum of the ELBO of the classifier and the ELBO of the VAE regularizer. Depending on whether the label is present or not for an instance, the total ELBO will have two forms which are also summed to form the semi-supervised total ELBO.

<img src="/images/VAE_intuitions/vae_semi_M2.png" alt="semi-supervised model inference" width="350" height="350">


    + (M1+M2) This model stacks M1 and M2 models. First we learn features  unsupervised in M1 and then we feed M1 learned features to M2 and learn semi-supervised.

<img src="/images/VAE_intuitions/vae_semi_M1_M2.png" alt="semi-supervised model inference" width="350" height="350">

- In a traditional classifier we predict labels, $$y$$, from data, $$x$$, i.e. $$p(y|x)$$. In M2 semi-supervised model, the label $$y$$ is considered to be a categorical latent variable since for some data points the label is known and for other points it's unknown. The model has other latent variables $$Z$$. If we can calculate the joint distribution $$p(X,Y,Z)$$, we can then calculate the conditional of label given data i.e. $$p(y|x)$$ as a semi-supervised classifier. 

<img src="/images/VAE_intuitions/semi_sup_classifier.png" alt="traditional vs semi-supervised classifier" width="350" height="350">

- The inference model for the semi-supervised model is as follows: $$q(Y|X) = categorical(\lambda = DNN_\phi(X)), q(Z|X,Y)= N(\mu = DNN_\phi(X), diag(\sigma^2 = DNN_\phi(X)))$$ where the parameters $$\lambda, \mu, \sigma$$ are parameterized by the encoder deep neural network. Softmax function can be used as the categorical distribution. 

- The priors for the generative model for the semi-supervised model are: $$Z ~ p(Z) = N(0,I) ; Y ~ \frac{1}{number of labels} ; X ~ p(X|Y,Z) = DNN(X; Y,Z,\theta)$$

- So there are actually three neural nets on the encoder side. One DNN for the $$\lambda$$ parameter of the categorical latent variable (label). One for $$\mu$$ and one for $$\sigma$$.
    + First we use X to determine the $$\lambda$$ of categorical variable using first encoder network.
    + second we sample the label categorical $$Y$$
    + third we use both $$X, Y$$ to determine the $$\mu, \sigma$$ of $$z$$ using the second and third encoder networks.
    + fourth we sample $$Z$$ from the normal distribution
    + fifth we generate $$X$$ from $$Y, Z$$ using the decoder network.


