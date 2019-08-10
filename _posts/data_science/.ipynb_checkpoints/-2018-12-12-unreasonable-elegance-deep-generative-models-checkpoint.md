---
layout: article
title: The unreasonable elegance of deep generative models
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

If you've been tracking the news on machine learnign and "AI", chances are that you've heard of deep generative models. Models that have gained notable popularity recently for their ability to generate realistic images. Generative models are statistical models that try to explicitly explain the process of generating a set of data points. The ability to model the data generation process without direct supervision (unsupervised learning) has been a general goal of machine learning for a long time. Although there is [some debate](https://twitter.com/HrSaghir/status/1069631380496224257) regarding whether unsupervised learning should aim to learn everything about the data as opposed to learning specific things, so far learning everything about data (density estimation) has been on a roll and holds [promise](https://twitter.com/dpkingma/status/1070856301868068864). Generative models have been around for the good part of the past couple of decades, however, the advent of deep learning has revived interest in such probabilistic models due to the extraordinary power of deep neural networks as function approximators.



## What is a generative model?
- A generative model is a probablistic model that in some sense approximates the natural distribution of the data.
    + an approach is to fit a latent variable model of the form p(x, z|θ) = p(z|θ)p(x|z, θ) to the data, where x are the observed variables, z are the parameterized latent variables. In maximum likelihood learning, we usually fit such models by minimizing a distance/divergence metric between the true data likelihood p(x) and the marginalized model likelihood p(x|θ) (e.g. $$KL[\frac{p(x|\theta)}{p(x)}]$$). 

<img src="/images/Generative_models/what_is_GM.png" alt="a depiction of generative models to provide intuition" width="350" height="350">


If you imagine the red dots in this figure as being the observed data points in our dataset, they basically are samples from the natural distribution of the data that we do not have access to. What we desire the in generative modelling is to fit a probablistic model similar to the green blob at the top to the surface of the natural distribution of the data as closely as we can. You can imagine that if we can approximate the surface of the natural distribution of the data with a probabilistic model, then randomly sampling our model should provide data samples that look like the observed data points. 


-  Examples of some efforts in the literature that have had some relative success in : Generate bedrooms, faces, etc (GANs, GLOW, VAEs, etc), image algebra, generate image from text and image caption, visual question answering, art applications, etc.

## why?

I am specificly interested in generative models due to their unreasonably elegant and principled approach in solving machine learning problems at a time that coming up with models is still more of a creative curosity than precise engineering. The basic goal of generative models is to perform density estimation, meaning that we want to take in a bunch of training data and fit a probability density function to them. Once learned, we can then use the model probability density function to generate new data similar to the training data.

## Problem setup: the scientific method

Almost everything that we can do with data involves finding the probability distribution underlying our data P(x). This includes finding insights in data, prediction, modeling the world, etc.  Therefore, We are interested in finding the true probability distribution of our data P(x) which is unknown. We usually use the scientific method in a probabilistic pipeline to solve this problem i.e.:

1. We determine the knowledge and questions we want answered
2. We make assumptions and formulate a model (using probabilistic graphical models, deep nets as functions, etc)
3. We fit the model with data, find the parameters of the model, find patterns in data (Inference)
4. We use the model for prediction, inference and exploration
5. And we finally criticize and revise the model


<img src="/images/Generative_models/scientific_method.png" alt="the scientific method as described by [Box, 1980; Rubin, 1984; Gelman et al., 1996; Blei, 2014]" width="350" height="350">

## Statistical modelling
In the statistical modelling pipeline, the first assumption that we make is to assume that each data point is a random variable (i.e. a distribution).

$X =\{ x_1,x_2,…,x_n \}$

$\{p(x_1), p(x_2), ..., p(x_n)\}$

The goal is usually to find the joint probability distribution of all the random variables (density estimation) so that we can obtain the unerlying distribution of the data:

$P(x_1,x_2, .., x_n) = \prod_i p(x_i)p(x_i|x_m)$

<img src="/images/Generative_models/joint_distribution.png" alt="the joint distribution of two random variables" width="350" height="350">


The statistical modeling procedure usually involves the introduction of some hidden variables, $$z_i$$, as hidden causes for the observed variables $$x_i$$, and a mixing procedure that we believe will lead to generation of the data as a model for the unknown true probability distribution of the data $$P(X)$$. The collection of observed variables $$X$$, and hidden variables, $$Z$$, form a joint probability distribution  $$P(X, Z)$$ that constructs our model. 

The joint distribution $$P(X, Z)$$ can be thought of as a combination of other simpler probability distributions through a hierarchy of component distributions i.e. $$P(X, Z)=P(X|Z)P(Z)$$. This means that we first sample the top distribution over hidden variables $$P(Z)$$ to choose a component that should produce data, then the corresponding component $$P(X|Z)$$ produces a sample x. This makes it easier to express the complex probability distribution of the observed data P(x) using a model P(x,z)=P(x|z)P(z). It is important to note that the real procedure for producing a sample x is unknown to us and the model is merely an attempt to find an estimation for the true distribution.

The task is then to fit the model by "infering" the latent variables from the data i.e. $$P(Z|X)$$. 

One of the most interesting ideas in statistics is the idea of the [Bayes' theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem). Reverend Thomas Bayes of England first thought of a way to formulate the integration of evidence with prior beliefs so that one can update beliefs. The famous Pierre-Simon Laplace then formulated Bayes' ideas in the modern form of $$P(Z|X) = \frac{P(X|Z) P(Z)}{P(X)}$$. 



At the beginning, the probability of choosing a component in the mixture is based on a very crude assumption for the general shape of such a distribution (i.e. prior P(z)) and we don't know the specific value of the parameters in the assumed structure for the probability model. We would like to find these unknowns based on the data. Finding the parameters to form a posterior belief is called inference and is the key algorithmic problem which tells us what model says about the data. In the probabilistic pipeline, we will then criticize the model we have built and might modify it and solve the problem again until we arrive at a model that satisfies our needs. 

## Problem setup (Latent variable generative model):
-  Let's assume our data as $$X = {x_1,x_2,...,x_n}$$. We are interested in performing inference, two-sample tests, prediction, generate more data like it, understand it, etc. Similar to what we do in all areas of science, we try to model the data to be able to answer such questions.

-  We use probability theory as a tool to start the modeling process. Let's assume each variable, $$x_i$$, as a random variable (meaning that it is actually a distribution and the possible values are random draws from the said distribution). In this setup, the model is the joint probability distribution of all random variables, i.e. $$p(x_1,x_2, .., x_n) = \prod_i p(x_i)p(x_i|x_m)$$. 

-  Finding the true joint distribution is no easy task. We appeal to modeling to find a good approximation for this joint distribution. There are two approaches we can take to solving this. 
    + non-parametric modeling
    + parametric modeling

## Non-parametric modeling

- Won't be focusing on non-parametric methods in this post but here is an example non-parametric approach to finding the joint distribution.

- Kernel Density Estimation:  Let's assume that variables are all independent and identically distributed (iid), and let's further assume a form of a kernel density function as the distribution of a random variable. The joint distribution will simply be the product of all kernel density functions. This is called non-parametric since the method can grow in complexity with data and not because it doesn't have parameters 

## Parametric modeling 
-  The first prior we encode into our modeling process is that we assume the data is coming from a generative process that we define. For example, if observations are $$X = {x_1, x_2, .., x_n}$$, we might assume that each data point is the outcome of a hidden local variable variable. This is a  simple generative model with a set of latent variables $$Z = {z_1, z_2,...,z_n}$$.

-  Now with this assumed generative model, we want to fit the model to data and infer the latent variables, $$z_i$$, from the visible variables $$x_i$$. The Bayes rule shows a way for performing inference through the notion of posterior. The idea is that we have some initial belief, we see data, then our beliefs evolve to posterior beliefs based on our observations. This is formalized as $$p(z|x) = \frac{p(x|z)p(z)}{p(x)}$$ where $$p(z)$$ is the prior on latent variables, $$p(z|x)$$ is the posterior belief on latent variables after observations, $$p(x|z)$$ is the likelihood of observations under the model, and $$p(x)$$ is the marginalized likelihood of observations under the model alternatively called evidence (when we ingrate out all latent variables).

-  In order to be able to use the Bayesian formula for inference and learning, we usually assume a functional form for the beliefs mentioned above (probability distributions). This functional form is called density function since it shows how dense the probability of the domain of the random variable of interest is. The density functions usually come with a set of sufficient statistics that while known, would completely define the density function on the variable domain. For example, the sufficient statistics for the Normal distribution density function are mean and variance parameters. We can choose to further parameterize these sufficient statistics parameters to give more expressibility to our model.

-  Since in the Bayesian world every parameter is interpreted as a belief (probability distribution), we can also treat the parameters in a Bayesian way, express them with another level of a density functions and parameterize them and then again treat the parameters of the parameters as belief and so on in an endless loop of Bayesian utopia! However, like all other good things, this endless loop has to end which means that we have to be practical Bayesian. 


### Types of models
The discipline of generative modeling has experienced enormous leaps in capabilities in recent years, mostly with likelihood-based methods (Graves, 2013; Kingma and Welling, 2013, 2018; Dinh et al., 2014; van den Oord et al., 2016a) and generative adversarial networks (GANs) (Goodfellow et al., 2014). Likelihood-based methods can be divided into three categories:
    - Autoregressive models (Hochreiter and Schmidhuber, 1997; Graves, 2013; van den Oord et al., 2016a,b; Van Den Oord et al., 2016). Those have the advantage of simplicity, but have as disadvantage that synthesis has limited parallelizability, since the computational length of synthesis is proportional to the dimensionality of the data; this is especially troublesome for large images or video.
    - Variational autoencoders (VAEs) (Kingma and Welling, 2013, 2018), which optimize a lower bound on the log-likelihood of the data. Variational autoencoders have the advantage of parallelizability of training and synthesis, but can be comparatively challenging to optimize (Kingma et al., 2016).
    - Flow-based generative models, first described in NICE (Dinh et al., 2014) and extended in RealNVP (Dinh et al., 2016), Glow (Kingma 2018). 
        + advantage: Exact latent-variable inference and log-likelihood evaluation. In VAEs, one is able to infer only approximately the value of the latent variables that correspond to a datapoint. GAN’s have no encoder at all to infer the latents.
        + Efficient inference and efficient synthesis, compared to autoregressive models.
        + Useful latent space for downstream tasks. The hidden layers of autoregressive models have unknown marginal distributions, making it much more difficult to perform valid manipulation of data. In GANs, datapoints can usually not be directly represented in a latent space.
        + Significant potential for memory savings. Computing gradients in reversible neural networks requires an amount of memory that is constant instead of linear in their depth.
        + In most flow-based generative models (Dinh et al., 2014, 2016), the generative process is defined as an invertible transformation (g) of a latent variable (z) (usually a simple and tractable distribution e.g. mutlivariate Gaussian) to data points (x). The invertability of the transformation makes inference of latent z from data x easy. 

#### 1. Fully-observed models: 
These Models observe data directly without introducing any new local latent variables but have a deterministic internal hidden representation of the data. Let observations be $$X = {x_1, x_2, .., x_n}$$ as random variables. The joint distribution is described as $$p(X) = p(x_1)p(x_2|x_1)...p(x_n|(x_1,...,x_n))$$. This will be a directed model of only observed variables.
    + We can assume a prior for each variable e.g. $$x_i ~ Cat(\pi|\pi(x_1,...,x_i))$$. 
    + all conditional probabilities (e.g. $$p(x_i|(x_1,...,x_{i-1}))$$) may be described using deep neural nets (e.g. an LSTM).  
    + They can work with any data type.
    + likelihood function is explicit in terms of observed variables. Therefore, the log-likelihood is directly computable without any approximations. 
    + They are easy to scale up.
    + limited by the dimension of their hidden representations (assumed degree of conditional dependencies).
    + generation can be slow due to sequential assumption.
    + for undirected models parameter learning is difficult due to the need for calculation of the normalizing constant. 

![alt text](/images/Generative_models/fully_observed_models.png "map of instances of fully observed models")

- For example, in the case of char-RNN, the number of RNN unrolling steps is the degree of conditional probabilities. If we put a soft-max layer on the output, the decision of the RNN will be a probability distribution on possible outputs and the rest of the RNN can be deterministic. 

#####  examples:
-  MADE: Masked Autoencoder for Distribution Estimation: Autoregressive models are used a lot in time series modelling and language modelling: hidden Markov models or recurrent neural networks are examples. There, autoregressive models are a very natural way to model data because the data comes ordered (in time).

What's weird about using autoregressive models in this context is that it is sensitive to ordering of dimensions, even though that ordering might not mean anything. If xx encodes an image, you can think about multiple orders in which pixel values can be serialised: sweeping left-to-right, top-to-bottom, inside-out etc. For images, neither of these orderings is particularly natural, yet all of these different ordering specifies a different model above.

But it turns out, you don't have to choose one ordering, you can choose all of them at the same time. The neat trick in the masking autoencoder paper is to train multiple autoregressive models all at the same time, all of them sharing (a subset of) parameters θθ, but defined over different ordering of coordinates. This can be achieved by thinking of deep autoregressive models as a special cases of an autoencoder, only with a few edges missing.

#### 2. Latent variable models (Explicit Probabilistic graphical models): 
These models introduce an unobserved local random variable for every observed data point. This is in contrast with fully-observed models that do not impose such explicit assumptions on the data. They are easy to sample from, include hierarchy of causes believed. The latent variable structure encode our assumptions about the generative process of the data. Let data be $$X = {x_1, x_2, ..., x_n}$$. We assume a generative process for example a latent Gaussian model $$z ~ N(0,I); x|z = N(\mu(z), \Sigma(z)); p(X,Z) = p(Z)p(X|Z) $$. 
    + Conditional distributions are usually represented using deep neural nets
    + easy to include hierarchy, depth, and the believed generating structure 
    + don't assume an order of conditional independence unlike fully-observed models. If we marginalize latent variables, we induce dependencies between observed variables similar to fully-observed models.
    + Latent variables can act as a new representation for the data.
    + Directed models are easy to sample from
    + Difficult to calculate marginalized log-likelihood and involves approximations
    + Not easy to specify rich approximate posterior for latent variables.
    + inference of latents from observed data is difficult in general


![alt text](/images/VAE_intuitions/latent_var_model.png "a latent variable model")

- In latent variable models, the probabilistic nature of the model is evident in both stochastic latent variables and stochastic observation variables. This means that we assume a latent node is a probability distribution and an observation node is also a probability distribution. The probability distribution function of the latent before inference is called prior and after inference posterior. The probability distribution of the observed nodes is called likelihood probability density function. Therefore, in such models, there is an explicit likelihood probability distribution function for observable nodes. However, this likelihood function is intractable in deep latent variable models. If we marginalize the latent nodes, we get a probability distribution which we sample to get observations. This is in contrast with implicit models. In implicit models, observations are not random variables. Observations are deterministic nodes, and therefore, there is no likelihood explicit probability density function. The likelihood is implicit in a deterministic function mapping a sample from a random variable (noise source) to an observation. 

![alt text](/images/Generative_models/Latent_variable_models.png "map of latent variable models")


##### examples:
- VAE

- Another example is LDA (latent Drichlet allocation), which is basically a mixture of multinomial distributions (topics) in a document. Its difference from a typical mixture is that each document has its own mixture proportions of the topics but the topics (multinomials) are shared across the whole collection (i.e. a mixed membership model). 


#### 3. Transformation models (Implicit generative models): 
- These models Model data as a transformation of an unobserved noise source using a deterministic function. Their main properties are that we can sample from them very easily, and that we can take the derivative of samples with respect to parameters. Let data samples be $$X = {x_1, x_2, ...,x_n}$$. We model the data as a deterministic transformation of a noise source i.e. $$ Z ~ N(0,I); X = f(Z; \theta)$$
    + The transformation is usually a deep neural network. 
    + It's easy to compute expectations without knowing final distribution due to the easy sampling and Monte Carlo averages.
    + It's difficult to maintain invertability. 
    + Challenging optimization.
    + They don't have an explicit likelihood function. Therefore, difficult to calculate marginal log-likelihood for model comparison. 
    + difficult to extend to generic data types.

![alt text](/images/Generative_models/transformation_models.png "map of instances of transformation models")

These models are usually used as generative models to model distributions of observed data. They can also be used to model distributions over latent variables as well in approximate inference (e.g. adversarial autoencoder).


##### examples:
- Flow-based generative models (NICE, RealNVP, GLOW, Normalizing flows): In most flow-based generative models (Dinh et al., 2014, 2016), the generative process is defined as an invertible transformation (g) of a latent variable (z) (usually a simple and tractable distribution e.g. mutlivariate Gaussian) to data points (x). The invertability of the transformation makes inference of latent z from data x easy. 
    - Real-valued non-volume preserving transformation (Real NVP): is an invertable transformation model. The generative procedure from the model is very similar to the one used in Variational Auto-Encoders and Generative Adversarial Networks: sample a vector z from a simple distribution (here a Gaussian) and pass it through the generator network g to obtain a sample x=g(z). 
        - The idea is to split the parameters of a NN layer to two segments, then transform one group of params by a sequence of an element-wise scaling and an offset. These two element-wise transformations are done by two matrices that are a function of the other group of parameters. this function can be a NN. We interleave the transformation of the two groups of parameters. 
        - This constructs a reversible function that has a diagonal jacobian. Therefore calculating the determinant of the jacobian is as easy of the product of the diagonal of the jacobian. 
        - The generator network g has been built in the paper according to a convolutional architecture, making it relatively easy to reuse the model to generate bigger images. As the model is convolutional, the model is trying to generate a “texture” of the dataset rather than an upsampled version of the images it was trained on. This explains why the model is most successful when trained on background datasets like LSUN different subcategories. This sort of behaviour can also be observed in other models like Deep Convolutional Generative Adversarial Networks.
        - From the generated samples, it seems the model was able to capture the statistics from the original data distribution. For example, the samples are in general relatively sharp and coherent and therefore suggest that the models understands something more than mere correlation between neighboring pixels. This is due to not relying on fixed form reconstruction cost like squared loss on the data level. The models seems also to understand to some degree the notion of foreground/background, and volume, lighting and shadows. 

    - Glow: it's similar to realNVP with the difference that the structure of the reversible function is different and uses 1x1 convolutions as the scaling transformation. 


- Another example, Deep Unsupervised Learning using Nonequilibrium Thermodynamics. What we typically try to do in representation learning is to map data to a latent representation. While the Data can have arbitrarily complex distribution along some complicated nonlinear manifold, we want the computed latent representations to have a nice distribution, like a multivariate Gaussian. This paper takes this idea very explicitly using a stochastic mapping to turn data into a representation: a random diffusion process. If you take any data, and apply Brownian motion-like stochastic process to this, you will end up with a standard Gaussian distributed variable due to the stationarity of the Brownian motion. Now the trick the authors used is to train a dynamical system (a Markov chain) to inverts this random walk, to be able to reconstruct the original data distribution from the random Gaussian noise. Amazingly, this works, and the traninig objective becomes very similar to variational autoencoders. 

- GANs are also of this type of (transformation) generative models where a random Gaussian noise is transformed into data (e.g.an image) using a deep neural network. These models also assume a noise model on the latent cause. The good thing about such models is that it's easy to sample and compute expectation from these models without knowing the final distribution. Since classifiers are well-develped, we can use our knowledge there for density ratio estimation in these models. However, these models lack noise model and likelyhood. It's also difficult to optimize them.  Any implicit model can be easily turned into a prescribed model by adding a simple likelihood function (noise model) on the generated outputs but models with likelihood functions also regularly face the problem of intractable marginal likelihoods.But the specification of a likelihood function provides knowledge of data marginal p(x) that leads to different algorithms by exploiting this knowledge, e.g., NCE resulting from class-probability based testing in un-normalised models, or variational lower bounds for directed graphical models.

- Think of GANs as a density estimation problem. There are two high dimensional surfaces. One of them is P(x) which is unknown, we want to estimate, and we only have some examples of. The other is a maleable goo from a family of surfaces that we want to match onto the unknown density to estimate it. 

- Variational inference tries to solve this by assuming a goo Q(z|x) defined by a parametric model from the exponential family, a fixed distance metric (i.e. KL divergence) to make a force field, and letting the physics takes its course (running SGD on parameters). MCMC methods don't use any model and instead try to just throw random points onto the surface and see where it ends up and use samples to estimate the shape of the surface P(x). Variational inference can't match all the intricacies of the high dimensional surface while MCMC is very costly since using samples to estimate a high dimensional surface is a fools errand!

- GANs have a more elegant approach, they define the goo to be a transformation model Q(x) that doesn't have a tractable likelihood function (can get very complex) but is very easy to sample from instead. Instead of using a rigid distance metric, GANs actively define an adaptive distance metric that tries to capture the intricacies of the unknown surface in every iteration. They do this by using the insight that classifiers can actually find a surface between two data sources easily and we have two data sources from the unknown surface and the goo. Therefore, the surface that the classifier finds to distinguish the data sources, captures the intricacies of the unknown surface P(x). So, if after each iteration that the classifier finds the optimum surface between the unknown and the goo, we reshape the goo to beat the classifier surface, the goo will very well take the shape of the surface at the end. This is a very clever way of matching the goo to the unknown surface P(x) that basically uses an adaptive distance metric using the distance to the classifer surface instead of the unknown surface. the distance to the classifier surface sort of hand holds the goo optimization step by step until it gets as close as possible to the unkown surface. Another point is that we only have a limited number of examples from the unknown surface, so we actually do the above process stocastically in batches of examples from unknown and the goo surfaces.

- Another type of implicit models, are simulators that transform a noise source into an output. Sampling from such simulators is easy but explicitly calculating a likelihood distribution function is usually no possible if the simulator is not invertable and mostly intractable. An example is a physical simulator based on differential euqations derived from eqautions of motions.

- Deep Implicit Models (DIM): 
    - These are stacked transformation models in a graph. For example, if a GAN is creating the noise code for the next GAN, it's a deep implicit model since likelihoods are not explicitly defined. This enables building complex densities in a hierarchical way similar to probabilistic graphical models. 

    - Inference in these models encounters two problems. First, like other inference problems marginal probability is intractable. Second, in transformation models likelihood is also intractable. We thus turn to variational inference with density ratio estimation instead of density estimation.

    - A DIM is simply a deep neural network with random noise injected at certain layers. An additional reason for deep latent structure appears from this perspective: training may be easier if we inject randomness at various layers of a neural net, rather than simply at the input. This relates to noise robustness.


### Bayesian rule and inference
- we assume a generative model, fit the model by "inferring" the latent variables from the data
- The Bayes rule shows a way. 
- to use the Bayesian formula for inference and learning, we assume a functional form for the beliefs (density function)
- We can choose to parameterize these density functions to give more expressibility to our model.
- Putting parameterized density functions into Bayesian formula, we end up with an equation of unknowns. 


#### Inference problems:

1. Evidence estimation
- marginal likelihood of evidence: Write the log density as the marginalization of the joint. We introduce a variational approximate q, into our marginal integral of joint p(x,z), to get p(x). By taking the log from both sides, and using Jensen's inequality we get the ELBO. Maximizing the ELBO is equivalent to minimizing the KL divergence of the real and variational posterior. 

2. Density ratio estimation (Density estimation by comparison)
The main idea is to estimate a ratio of real data distribution and model data distribution p(x)/q(x) instead of computing two densities that are hard. The ELBO in variational inference can be written in terms of the ratio. Introducing the variational posterior into the marginal integral of the joint results in the ELBO being $E[log p(x,z)- log q(z/x)]$. By subtracting emprical distribution on the observations, q(x) which is a constant and doesn't change optimization we have the ELBO using ratio as $E[log p(x,z)/q(x,z)]$. 

- Probabilistic classification: We can frame the ratio estimation as as the problem of classifying the real data (p(x)) from the data produced from model (q(x)). This is what happens in GANs.

- moment matching (log density difference): if all the infinite statistical moments of two distributions are the same the distributions are the same. So the idea is to set the moments of the numenator distribution (p(x)) equal to the moments of a transformed version of the denumerator (r(x)q(x)). This makes it possible to calculate the ratio r(x).

- Ratio matching: basic idea is to directly match a density ratio model r(x) to the true density ratio under some divergence. A kernel is usually used for this density estimation problem plus a distance measure (e.g. KL divergence) to measure how close the estimation of r(x) is to the true estimation. So it's variational in some sense. Loosely speaking, this is what happens in variational Autoencoders!

- Divergence minimization: Another approach to two sample testing and density ratio estimation is to use the divergence (f-divergence, Bergman divergence) between the true density p and the model q, and use this as an objective to drive learning of the generative model. f-GANs use the KL divergence as a special case and are equipped with an exploitable variational formulation (i.e. the variational lower bound). There is no discriminator in this formulation, and this role is taken by the ratio function. We minimise the ratio loss, since we wish to minimise the negative of the variational lower bound; we minimise the generative loss since we wish to drive the ratio to one.

- Maximum mean discrepancy(MMD): is a nonparametric way to measure dissimilarity between two probability distributions. Just like any metric of dissimilarity between distributions, MMD can be used as an objective function for generative modelling.  The MMD criterion also uses the concept of an 'adversarial' function f that discriminates between samples from Q and P. However, instead of it being a binary classifier constrained to predict 0 or 1, here f can be any function chosen from some function class. The idea is: if P and Q are exactly the same, there should be no function whose expectations differ under Q and P. In GAN, the maximisation over f is carried out via stochastic gradient descent, here it can be done analytically. One could design a kernel which has a deep neural network in it, and use the MMD objective!?

- Instead of estimating ratios, we estimate gradients of log densities. For this, we can use[ denoising as a surrogate task](http://www.inference.vc/variational-inference-using-implicit-models-part-iv-denoisers-instead-of-discriminators/).denoisers estimate gradients directly, and therefore we might get better estimates than first estimating likelihood ratios and then taking the derivative of those


3. Moment computation

$E[f(z)|x] =\int f(z)p(z|x)dz$

4. Prediction

$p(xt+1) =\int p(xt+1|xt)p(xt)dxt$

5. Hypothesis Testing

$B = log p(x|H1) - log p(x|H2)$

### Parameter Learning
- Let's get back to the Bayesian formula. If we write the probability distributions as parameterized density functions, we will end up with an equation with unknown parameters. So the inference task is now transformed into the problem of finding parameter values from observations. 

- how to learn a parametric model based on training data? There are three main approaches to this problem. The first approach, called maximum likelihood, says that one should choose parameter values that maximize the components that directly sample data (i.e. the likelihood or the probability of the data under the model, this corresponding to the first term in the ELBO, or an error function). In a sense, they provide the best possible fit to the data. As a result, there is a tendency towards over fitting. If we have a small amount of data, we can come to some quick conclusions due to small sample size.

The second is the Bayesian approach which is a belief system that suggests every variable including parameters should be beliefs (probability dist) not a single value. It provides an alternative approach to maximum likelihood, that does not make such a heavy commitment towards a single set of parameter values, and incorporates prior knowledge. In the Bayesian approach, all parameter values are considered possible, even after learning. No single set of parameter values is selected. Instead of choosing the set of parameters that maximize the likelihood, we maintain a probability distribution over the set of parameter values. The ELBO represents Bayesian approach where the second term represents balancing a prior distribution with the model's explanation of the data. The third approach, maximum a-posteriori (MAP), is a compromise between the maximum likelihood and the Bayesian belief system. The MAP estimate is the single set of parameters that maximize the probability under the posterior and is found by solving a penalized likelihood problem. However, it remedies the over fitting problem a bit by veering away from the maximum likelihood estimate (MLE), if MLE has low probability under the prior. As more data are seen, the prior term will be swallowed by the likelihood term, and the estimate will look more and more like the MLE.

- So the three schools of thoughts for how the parameter set should be chosen:
    + The simplest and most obvious school of thought is that parameters are not distributions and should be chosen in a way that will maximize the likelihood of observations under the model. This gives rise to the maximum likelihood parameter learning concept. 

    + The Bayesian school of thought obviously believes that the parameters are distributions themselves and thus we need to infer distributions for parameters too and not just learn single values.

    + Another school of thought tries to balance the above two ideas by professing that although the parameter maybe distributions but we want to be practical so instead of inferring a distribution for each parameter we choose the single parameter values that will maximize the posterior belief of latent values. This gives rise to the Maximum a Posteriori (MAP) inference concept. 

Note that the loss function (i.e. the distance measure) in optimization (i.e. maximum likelihood) is the root of the all problems since optimization's only objective is to reduce the loss. If the loss is not properly defined, the model can't learn well and if the loss function does not consider the inherent noise of the data (i.e. regularization or MAP), the model will eventually over fit to noise in the data and reduce generalization. Therefore, the loss function (distance measure) is very important and the reason why GANs work so well is that they don't explicitly define a loss function and learn it instead! The reason that Bayesian approach prevents over fitting is because they don't optimize anything, but instead marginalize (integrate) over all possible choices. The problem then lies in the choice of proper prior beliefs regarding the model.

- Vanilla maximum likelihood learning and MAP inference are simple and fast but too biased and approximate. So we usually would like to have one more level of full Bayesian goodness and obtain distributions. If we take the Bayesian school of thought and try to infer the posterior belief of latent variables, we need to be able to calculate the marginal likelihood term $$p(x)$$. Unfortunately, this term involves an integration which is intractable for most interesting models. 

- Since the posterior $$p(z|x)$$ is intractable in this model, we need to use approximate inference for latent variable inference and parameter learning. Two common approaches are:
    + MCMC inference: asymptotically unbiased but it's expensive, hard to assess the Markov chain convergence and manifests large variance in gradient estimation.
    + Variational inference: fast, and low variance in gradient with re parameterization trick. well suited to the use of deep neural nets for parameterizing conditional distributions. Also well-suited to using deep neural networks to amortize the inference of local latent variables $${z_1,..., z_n}$$ with a single deep neural network. 


## Bayesian network vs Parameterization network
As can be seen, the probabilistic graphical model (Bayesian network), the generative model factorization, the inference model factorization, the ELBO objective, and even the variational inference procedure, don't include any information about how the conditional distributions in the graph are parameterized. Therefore, in implementation we can separate these two parts and have two parts in the deep generative model implementation code:
1. The graphical model, inference and generative network factorizations, the ELBO, and even the variational inference procedures. 
2. The parameterizations using neural network

## Model vs Action
- We can build a generative model by pairing a model, inference scheme and build an algorithms for density estimation of the model. There are therefore, two distinct things we might be interested in, first is building a model. Second is what to do with this model. If we want to make decisions and act upon the model then we face the problem of reinforcement learning. We build a model first and then put it in an environment to take actions and get rewards in order to evolve. 



We Write down a parameterized density function (a function that assigns a probability to all data points) and find parameters that maximize the likelihood that the density function assigns to data. Most of the time though we can't write the likelihood explicitly so we end up maximizing other functions that approximate likelihood (e.g. ELBO).
    + Explicit density: 
        * tractable density:
            - Autoregressive models/Fully visible belief nets (MADE/NADE/PixelRNN/PixelCNN/Wavenet )
                + [Frey 1996] used logistic regression for conditionals
                + [Bengios 2000] used neural nets for conditionals
                + [Larochelle 2011] used neural nets with weight sharing (NADE) for images
                + We need to pick an ordering of dimensions to use such models but since density is explicit they are easy to train. 
                    * PixelCNN: uses dilated convolutions to model conditionals in the chain rule to p(X). The conditional is also assumed over color channels. Each conditional outputs a multinomial distribution for the value of each pixel. Uses a stack of dilated convolutions which can be parallelized in training. Generation is still slow though.
            - Deterministic trasformations/Change of variable models (nonlinear ICA/real NVP/normalizing flows)
        * approximate density:
            - variational inference (e.g. VAE)
            - MCMC variations (e.g. RBM)
    
    + Implicit density:
        * Direct:
            - Generative Adversarial Nets (GAN)
                + In GAN, the latent variable Z is drawn at random during training, and a discriminator is trained to tell if the generated output looks like it's been drawn from the same distribution as the training set.
    
            - Generative Latent Optimization(GLO): 
                + In GLO, the latent variable Z is optimized during training so as to minimize some distance measure between the generated sample and the training sample Z* = min_z = Distance(Y,G(Z)). The parameters of the generator are adjusted after this minimization. The learning process is really a joint optimization of the distance with respect to Z and to the parameters of G, averaged on a training set of samples. After training, Z can be sampled from their allowed set to produce new samples. Nice examples are shown in the paper.
        * Markov chain:
            - Generative Stochastic Networks(GSN)
                + In GSN, a Markov chain is built to sample data and is modified incrementally as it produces samples from the model using MCMC variations.




Generative models can be used for:
    + Data augmentation and generating simulated training data to improve classification (Apple did this for eye gaze data [Shrivastava 2016])

    + Missing data imputation/Semi-supervised learning: [Yeh et al 2016](https://arxiv.org/abs/1607.07539) used generative models to do image in painting

    + Train model to learn multiple correct answers: [Lotter et al 2016] use generative models to predict next frame video prediction. based on the rotation of the face, multiple answers might be correct. If we train using MSE loss, multiple futures are combined to a single answer and result in a blurry average of all. Using adversarial loss they were able to capture the distribution over all possible future images and generate a sharp prediction.

    + Realistic generation tasks: generate media and special effects (art applications). [Zhu et al 2016] created iGAN where the model will find the nearest neighbor picture from a simple drawing. CycleGAN translates horses to zebras in an unsupervised fashion and Image synthesis creates a picture conditioned on a sentence.

    + simulation by prediction: instead of using a computational model for simulating a phenomenon, [de Olivera et al 2017] use generative models to predict the output of the simulator conditioned on simulation parameters.

    + Learn useful embeddings: for example word embeddings. [Radford et al 2015] use DCGAN to find embedding for images that enables linear algebra on images. This can be useful for image search, information retrieval etc. [Chen et al 2016] introduce infoGAN to learn controllable and interpretable latent codes. 
        * [Nguyen et al 2016 ] introduce PPGN and combine adversarial training, moment matching, denoising autoencoders (DAE) and Monte Carlo sampling (Langevin sampling) to generate high quality high-res images conditioned on category or caption on ImageNet.
            - Basic idea is that they have a variation of a Markov chain that follows the gradient of the log density (Langevin sampling). It uses a DAE to estimate the gradient instead of explicitly computing the gradient (synthetic gradient?). And finally use a special DAE that has been trained with multiple losses including a GAN loss to obtain best results. 



explore variations in data, to reason about the structure and behavior of the world, and ultimately, for decision-making. Deep generative models have widespread applications including those in image denoising and in-painting, data compression, scene understanding, representation learning, 3D scene construction, semi-supervised classification, and hierarchical control, amongst many others.




## Evaluating generative models

- an approach to representation learning is to fit a latent variable model of the form p(x, z|θ) = p(z|θ)p(x|z, θ) to the data, where x are the observed variables, z are the parameterized latent variables. In maximum likelihood learning, we usually fit such models by minimizing a distance/divergence metric between the true data likelihood p(x) and the marginalized model likelihood p(x|θ) (e.g. $$KL[\frac{p(x|\theta)}{p(x)}]$$). However, the fundamental problem is that these loss functions only depend on p(x|θ), and not on p(x, z|θ). Thus they do not measure or optimize for the quality of the representation at all. In particular, if we have a powerful stochastic decoder p(x|z, θ), such as an RNN or PixelCNN, a VAE can easily ignore z and still obtain high marginal likelihood p(x|θ). Thus obtaining a good ELBO (and more generally, a good marginal likelihood) is not enough for good representation learning.

- Two identical values of ELBO, can have different quantitative and qualitative characteristics due to the fact that ELBO consists of two terms (i.e. reconstruction cost plus the KL between prior and variational posterior). 

- Rate-Distortion theory addresses the problem of determining the minimal number of bits per symbol, rate R, that should be communicated over a channel, so that the source can be reconstructed at the output without exceeding a given distortion D. rate-distortion curve that characterizes the tradeoff between compression and reconstruction accuracy. 

- The main idea of the paper is that a better way to assess the value of representation learning is to measure the mutual information I between the observed X and the latent Z. This is intractable but a variational lower and upper bounds on the mutual information between the input and the latent variable can be obtained ($$H − D ≤ I_e(X;Z) ≤ R$$ where H is dataset entropy, D is distortion reconstruction cost as measured through our encoder,
decoder channel, and rate R is KL between prior and variational posterior and depends only
on the encoder and variational marginal). By varying I, we can tradeoff between how much the data has been compressed vs how much information we retain represented as the rate-distortion curve.  information constraints provide an interesting alternative way to regularize the learning of latent variable models.

- Having defined a joint density, a symmetric, non-negative, reparameterization-independent measure of how much information one random variable contains about the other is given by the mutual information. There are two natural limits the mutual information can take. In one extreme, X and Z are independent random variables, so the mutual information vanishes: our representation contains no information about the data whatsoever. In the other extreme, our encoding might just be an identity map, in which Z = X and the mutual information becomes the entropy in the data H(X).

- Alternatively, instead of considering the rate as fixed, and tracing out the optimal distortion as a function of the rate D(R), we can perform a Legendre transformation and can find the optimal rate and distortion for a fixed $$\beta$$ in a beta-VAE setup.


- For decoder-based model that do not have a tractable likelihood function, likelihood of a sample data point can be evaluated using sampling and Monte Carlo. 
    + In case of a VAE, the liklihood density function is assumed to be a Gaussian with known mean and variance. 
    + However, in case of GANs or GMMN, the likelihood is implicit and you don't have access to it, therefore, we appeal to sampling to find it. We simulate data from model (sample) and then assume a Gaussian ball around the sample as the density function to get the likelihood of that data point. 


### rate-distortion vs. information bottleneck

- While the rate-distortion theory allowed the consideration of trade-offs between communication rate and the amount of data that would survive it, Tishbi et al noticed that it doesn't consider the relevance of information.

    + Firstly, a distortion measure has to be defined before a rate-distortion problem can be defined. This poses a big problem, since the answer to the question ‘how much relevant information survives the communication?’ becomes dependent upon the choice of that distortion measure. 

    + Secondly, rate-distortion theory is especially powerful in cases where the distortion measure is assumed to be additive: $$d(\bold{x}, \bold{\hat{x}}) = \sum\limits_{x} d_i(x_i, \hat{x}_i)$$, where $$\bold{x}$$ and $$\bold{\hat{x}}$$ are vectors whose components are $$x_i$$ and $$\hat{x}_i$$, respectively. This of course excludes many possible cases, such as when relevant information exists only in one part of the vector.


- idea: The main premise of information bottleneck theory is as follows: Consider a random variable X, that has some hidden characteristic Y. Transmitting X to a different location, we are not interested in recreating X, but only in the characteristic Y. It would be very hard to find an additive distortion measure that would indicate good performance if we try to analyze this approach through rate-distortion theory.

- Since the total correlation depends on the joint distribution p(X,Y) and by extension on P(X). If we have $$n$$ binary $${0,1}$$ variables, then the search over all P(Y|X) involves $$2^n$$ variables which is intractable. $$\min \limits_{P_{T|X}}\{I(X;Y) - \beta I(Y;T)\}$$

- The coefficient \beta controls the trade-off between the importance of “forgetting” irrelevant parts of X while creating T and the importance of maintaining relevant parts for Y (aka bottleneck). When \beta = 0 only compression is important and the minimum can easily attain the value 0 by choosing T = 0. When \beta is very large, compression is no longer important and the minimum is achieved by choosing T = X. Thus trying to maximize I(T;Y) to relay the information contained in relavant part of data while minimizing I(T;X) to find the minimal set of relavant information at the same time creates a bottleneck effect.

- In any neural network, the output of each layer can be viewed as a random variable T. This random variable is fabricated from the entry to the network X, and thus the Markov chain $$T \leftrightarrow X \leftrightarrow Y$$ is respected. As this can be claimed for any layer in the network, each layer can be treated simultaneously as the exit of an encoder with entry X, and the entry to a decoder with the output being the prediction Y. Denoting the outputs of each of the layers as$$ T_1, T_2, \ldots$$, we can deduct from the data processing theorem that:

$$I(X;Y) \leq I(T_1; Y) \leq I(T_2; Y) \leq \cdots \leq I(\hat{Y}; Y)$$

Each of the representations T can thus be seen in the information-bottleneck perspective resulting in the view of a DNN as a chain of information bottlenecks.

- Experimenting with a synthetic dataset of (X,Y) that have a high mutual information, the following observation is made: Looking at any layer, its outputs T can be drawn on the ‘information plain’ defined by the mutual information with both the input X and the expected label Y. Doing so, the observation is that the training process can clearly be divided into two. 
    + In the first part, called the ‘drift’, the network learns to extract the correct label Y from the information X relatively well. It does so by increasing the mutual information of each of the layers T both with the input X and with the label Y. Intuitively, it is easy to understand why the network would seek to increase the mutual information with the labels Y.
    + In the second, slower, part of the training process, 'compression/diffusion', the network maintains the mutual information with Y while decreasing the mutual information between X and T. Intuitively, what can be gained by decreasing the information of each layer T about the inputs, when we keep in mind that there are no communication constraints? Maybe it helps generalization by forgetting non-relavant parts of the input that don't help with knowing Y better. But why wouldn't we always get better generalizatio by letting more training happen? Maybe the network forgets some relavant parts of X that need to be there for generalization but since the training objective is only looking at the observed Y, it can't generalize to unseen Ys?
        - This also explains why this phase of training is slower than the first – it happens randomly as the network sees more and more examples that may or may not carry these “unimportant” characteristics. 

- an ICLR18 paper reproduced the Tishby's results and noted that: changing the activation function from a tanh nonlinearity (except for the final layer) to a ReLU nonlinearity changed the observed results drastically. 
    + It turns out that only with the tanh nonlinearity a decrease in the mutual information with X can be observed in later parts of the training process. ReLU nonlinearity does not exhibit this kind of behaviour.
    + Andrew Saxe et al. argues that the compression phase observed in the original paper is an artifact of the double-sided saturating nonlinearity used in that study. Saxe et al. observe no compression with the more widely used relu nonlinearity (nor in the linear case). Moreover, they show that even in cases where compression is observed, there is no causal relationship between compression and generalization (which seems to be consistent with the results from the revNet paper claiming that loss of information is not a necessary condition of learning for networks that generalize) and that in cases where compression is observed, it is not caused by the stochasticity of SGD, thus pretty much refuting all claims of the original paper. 


- The SVCCA paper, introduces the SVCCA metric as a similarity metric between learned representations of the layers. It is obtained by calculating the sum of the principle eigenvalues of the matrix formed by the outputs of each neuron for every datapoint in the dataset.
    + It plots the SVCCA similarity ρ between all pairs of layers in the network and notes that learning broadly happens ‘bottom up’, meaning that layers closer to the input seem to solidify into their final representations first with the exception of the very top layers. And the later layers converge to their final representation much later in the training compared to the top layers.
    +  Other patterns are also visible – batch norm layers maintain nearly perfect similarity to the layer preceding them due to scaling invariance.
    + In resnet, we see a stripe like pattern due to skip connections inducing high similarities to previous layers.
    +  the “lower layers learn first” behavior was also observed for recurrent neural networks.



### VAEs vs GANs

- VAEs use maximum likelihood and variational posterior inference for learning and require generative models to have explicit densities and noise terms in the sample space to make inference possible. 
    + a combination of model mismatch and poor estimation of the posterior due to approximation/amortization gaps results in systematic biases in the learned distribution, e.g. undesirable averaging effects. 

- GANs are the antithesis of maximum likelihood since they require neither an inference mechanism, nor a generative distribution that admits an explicit density. Estimation of the discrepancy between the data distribution and the generative model is accomplished through a divergence approximator that is learned adversarially using independent samples from both distributions.
    + This estimation is sound only at the non-parametric limit and the gap between the actual discrepancy and the estimated lower bound needs to be reduced through careful guidance of the functions selected to represent the divergence.

- [Primal-Dual Wasserstein GAN]() synthesizes the desirable qualities of VAEs and GANs using Optimal Transport and divergences (Wasserstein distance). 


#### Unifying deep generative models 
VAEs and GANs can be formulated as instances or approximations of a loss-augmented variational posterior inference problem of latent variable graphical models. Both VAEs and GANs have a generator that we can sample to generate data x. The difference is that while the GAN generator doesn't admit evaluating the likelihood of the data, a VAE's generator admits an explicit liklihood $$p(x|z)$$. 

- VAEs additionally learn a variational distribution (a.k.a. inference model) 𝚚(z|x; η), which approximates the true posterior p(z|x; θ) that is proportional to p(x|z; θ)p(z). And, using the classic framework of variational EM algorithm, the model is trained to minimize the KL divergence between the variational distribution and the true posterior: 

$$KL( 𝚚(z|x; η) || p(z|x; θ) ) $$

- In contrast, GANs accompany the generator with a discriminator, 𝚚_φ, by setting up an adversarial game in which the discriminator is trained to distinguish between real data and generated (fake) samples, while the generator is trained to produce samples that are good enough to confuse the discriminator.



### Integrating prior structures into models 


#### Posterior regularization
Posterior regularization is a principled framework to impose known fixed knowledge constraints on posterior distributions of probabilistic models. PR augments the objective (typically log-likelihood) by adding a constraint term $$L(\theta, q)$$ encoding the domain knowledge. 


For efficient optimization, instead of directly doing constraint optimization using the constraint $$f(x)$$ (i.e. adding $$f(x)$$ to objective), we add a term that minimizes the distance between the constraint version and the unconstraint version. We do this by imposing the constraint $$f(x)$$ on an auxiliary distribution q, which is encouraged to stay close to the posterior of the model $$p_\theta$$ through a KL divergence term. 

$$L(\theta, q) = KL(q(x)||p_\theta (x)) − \alpha E_q [f_\phi (x)] ,$$

This objective trades off likelihood and distance to the desired posterior subspace. The problem is solved using an EM-style algorithm. Specifically, 

- the E-step optimizes above equation w.r.t q, which is convex and has a closed-form solution at each iteration given θ:

$$q^∗(x) = p_θ(x) exp {αf(x)}/ Z$$

$$q^*$$ can be seen as an energy-based distribution with the negative energy defined by $$ \alpha f(x) + \log p_\theta(x)$$.
- Given q from the E-step, the M-step optimizes the loss w.r.t θ with:

$$min_θ KL(q(x)||p_θ(x)) = min_θ −E_q [log p_θ(x)] + constraint given by q.$$

This minimizes the KL divergence between the two so that the unconstrained version gets as close as possible to the constrained version.

In practice, this means that we have an additive objective with two terms, the original unconstrained objective (e.g. log-likelihood) and a constraint term that depends on (q). In each forward pass, we first calculate (p) and then project it into a  constrained subspace using the closed form solution above ($$q^∗(x) = p_θ(x) exp {αf(x)}/ Z$$) to get (q). Second, in the backward pass, we minimize the network w.r.t. the additive objective. 

A similar imitation procedure has been used in other settings called distillation. Following them we call pθ(y|x) the “student” and q(y|x) the “teacher”, which can be intuitively explained in analogous to human education where a teacher is aware of systematic general rules and she instructs students by providing her solutions to particular questions.  An important difference from previous distillation work, where the teacher is obtained beforehand and the student is trained thereafter, is that our teacher and student are learned simultaneously during training.

Though it is possible to combine a neural network with rule constraints by projecting the network to the rule-regularized subspace after it is fully trained as before with only data-label instances, or by optimizing projected network directly, we found our iterative teacher-student distillation approach provides a much superior performance.  Moreover, since pθ distills the rule information into the weights θ instead of relying on explicit rule representations, we can use pθ for predicting new examples at test time when the rule assessment is expensive or even unavailable.

#### learning the constraints

However, many deep generative models (e.g. GANs, autoregressive NN) do not possess a straight-forward probabilistic formulation or even meaningful latent variables. Besides, the constraints need to be known beforehand to use constraint optimization of PR framework. The papers ["Deep Generative Models with Learnable Knowledge Constraints"](https://arxiv.org/pdf/1806.09764.pdf) establishes formal mathematical correspondence between the model and constraints in PR and the policy and reward in entropy-regularized policy optimization. Then it uses inverse RL to learn the constraints (i.e. reward) and then uses PR to do constraint optimization on any type of model. 

- a line of research is trying to formalizing RL as a probabilistic inference problem

Inverse reinforcement learning (IRL) seeks to learn a reward function from expert demonstrations. The paper uses maximum-entropy IRL to derive the constraint learning objective, and leverage the unique structure of PR for efficient importance sampling estimation.

Algo:
    - In short, we break an input into the constraint region and other info. For example, in sentence generation, constraint region is the templated part of the sentence, while the rest are content information like entities. 
    - we have a GAN-like setup with two interactive models, that are learned in stages.
        + First model is an energy-based model that assigns an affinity score (similarity score) to a template and the sentence. It somehow implements the constraint by assigning a high score to a template and its corresponding sentence. 
            * Stage1 learning: we use inverse RL gradient to train this part of the model. 
        + Second model is the generator that takes in entities and reconstructs the entire sentences from entities information and affinity score from first model.
            * This part can of model can be trained using maximum likelihood with reconstruction cost of adversarially using an additional discriminator. 




## Integrating domain knowledge into DGM

- difficult to exploit rich problem structures and domain knowledge in various deep generative models

- Posterior Regularization provides a framework for imposing knowledge constraints on generative models

---

## learning the constraints
["Deep Generative Models with Learnable Knowledge Constraints"](https://arxiv.org/pdf/1806.09764.pdf)

- many deep generative models (e.g. GANs, autoregressive NN) do not possess a straight-forward probabilistic formulation (i.e. a posterior)

- the constraints need to be known beforehand to use PR. 

- contribution: establishes mathematical relationship between the model and constraints in PR and the policy and reward in entropy-regularized policy optimization.


- contribution: Uses max entropy IRL to learn the constraints (i.e. reward) and then PR to do constraint optimization on any type of model. 

---
# main ideas 

1. Imposing constraints on generative models

- ["Posterior Regularization for Structured Latent Variable Models"](http://www.jmlr.org/papers/volume11/ganchev10a/ganchev10a.pdf)

2. Connection between RL and Generative models 

- ["A Connection Between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models"](https://arxiv.org/abs/1611.03852)

- ["Trust Region Policy Optimization"](https://arxiv.org/pdf/1502.05477)


---

## Posterior regularization

- Problem: impose known knowledge constraints $f_\phi (x)$ on the posterior distribution of a probabilistic model $p_\theta (x)$. 

- For efficient optimization: impose the constraint on an auxiliary distribution $q$, which is encouraged to stay close to the posterior $p_\theta$ through a $\mathcal{KL}$ divergence term. 

$$L(\theta, q)= \mathcal{KL}(q(x)||p_\theta (x)) - \alpha E_q[f_\phi(x)]$$

- new objective:

$$
\min_{\theta, q} L(\theta) + \lambda L(\theta,q)
$$

solved using an EM-style algorithm.

---
- the E-step optimizes above equation w.r.t. $q$, which is convex and has a closed-form solution at each iteration given $\theta$:

    $$q^*(x) =  p_\theta(x) \frac{\exp(\alpha f(x))}{Z} $$
    
    
- Given q from the E-step, the M-step optimizes the loss w.r.t $\theta$:
$$
\begin{aligned}
\min_\theta L(\theta) + \lambda L(\theta,q) = \\ \min_\theta L(\theta)-E_q[\log p_\theta(x)] + const
\end{aligned}
$$

---

## Entropy regularized policy optimization

-  find the policy that maximizes the expected reward
$$
\min_q - E_q[R(x)]
$$

- regularize objective with $\mathcal{KL}$ between the new and old policy to stablize learning as in TRPO.

$$
\min_q \mathcal{KL}(q(x)||p(x)) - \alpha E_q[R(x)]
$$

- resemblence between this objective and PR objective where generative model corresponds to policy and constraint to reward. 


- Therefore,learning **the constraints** is equivalent to learning the **reward function**. 


---
## Inverse Reinforcement Learning

learning the reward function from expert demonstrations. 

### Maximum Entropy IRL

Maximum Entropy RL models the demonstrations using a Boltzmann distribution, where the energy is given by the cost fuction:


$$
q_\phi(\tau) = \frac{1}{Z} \exp(-c_\phi(\tau))
$$


- The optimal trajectories have the highest likelihood, and the expert can generate suboptimal paths with a probability that decreases exponentially. 

- Learning the reward function is cast as maximizing the liklihood of above distribution:

$$
\phi^* = argmax_\phi E_{\tau\sim p} [\log q_\phi(\tau)]
$$

---

## Algorithm

- Objective 

$$
\min_{q, \phi, \theta} L(\theta) +\mathcal{KL}(q(x)||p_\theta (x)) - \alpha E_q[f_\phi(x)]
$$

1. optimize objective w.r.t. $q$, to impose constraints (closed-form solution):

$$
q^*(x) =  p_\theta(x) \frac{\exp(\alpha f_\phi(x))}{Z} 
$$
    
2. optimize objective w.r.t. $\phi$, to learn constraints (using MaxEnt IRL):
   
$$
\phi^* = argmax_\phi E_{\tau\sim p} [\log q_\phi(\tau)]
$$
   
    
3. Given $q, \phi$ optimize objective w.r.t. $\theta$, to learn generative model:

$$
\min_\theta L(\theta)-E_q[\log p_\theta(x)] + const
$$

---
## Connections to GANs

- for implicit models, evaluating density is not possible. The paper proposes to minimize the reverse $\mathcal{KL}$ instead. 

$$
\begin{aligned}
\min_\theta L(\theta)- \mathcal{KL}(p_\theta (x)||q(x)) = \\
\min_\theta L(\theta) - E_{p_\theta}[\alpha f_\phi(x)] + \mathcal{KL}[p_\theta||p_\theta] + const
\end{aligned}
$$

- the two objectives w.r.t $\phi$ and $\theta$ are similar to a discriminator and a generator in a GAN. 
    + the constraints (discriminator) assign low energy to real samples from the data distribution $p_d(x)$ and high energy to samples from the constrained posterior $q(x)$. 
    + the generator $p_\theta(x)$ is optimized to generate samples that confuse the constraints $f_\phi(x)$
    + The generator uses information from the discriminator in generating a constrained fake data.


---


## Guided Cost Learning in MaxEnt IRL

 
- The algorithm estimates $Z$ by training a new sampling distribution $q(\tau)$ and using importance sampling. 

$$
\begin{aligned}
\mathcal{L}_{cost} (\theta) = \mathbb{E}_{\tau \sim p}[-\log p_\theta(\tau)] = \mathbb{E}_{\tau \sim p}[c_\theta(\tau)] + \log Z \\ = \mathbb{E}_{\tau \sim p}[c_\theta(\tau)] + \log (\mathbb{E}_{\tau \sim q} [\frac{\exp(-c_\theta (\tau))}{q(\tau)}])
\end{aligned}
$$


- The importance sampling estimate can have very high variance if the sampling distribution $q$ fails to cover some trajectories $\tau$ with high values of $\exp (-c_\theta (\tau))$. One way to address this is to mix sampling data and demonstrations $\mu = \frac{1}{2} p + \frac{1}{2} q$.


$$
\begin{aligned}
\mathcal{L}_{cost} (\theta) = \mathbb{E}_{\tau \sim p}[-\log p_\theta(\tau)] = \mathbb{E}_{\tau \sim p}[c_\theta(\tau)] + \log Z \\ = \mathbb{E}_{\tau \sim p}[c_\theta(\tau)] + \log (\mathbb{E}_{\tau \sim \mu}[\frac{\exp(-c_\theta (\tau))}{\mu(\tau)}])
\end{aligned}
$$

---

## GAN = MaxInt IRL

["A Connection Between Generative Adversarial Networks, Inverse Reinforcement Learning, and Energy-Based Models"](https://arxiv.org/abs/1611.03852)

For GAN the log loss for discriminator is equal to:


$$
\mathcal{L} (D_\theta) = \mathbb{E}_{\tau \sim p}[-\log D_\theta (\tau)] + \mathbb{E}_{\tau \sim q} [-\log (1 - D_\theta(\tau))]
$$


where

$$ D_\theta(t) = \frac{\frac{1}{Z}\exp(-c_{\theta}(\tau))}{\frac{1}{Z}\exp(-c_{\theta}(\tau)) + q(\tau)} $$

---

There are three facts that imply that GANs optimize precisely the MaxEnt IRL problem

1. The value of $Z$ which minimizes the discriminator's loss is an importance-sampling estimator for the partition function.n (Compute $\frac{\partial \mathcal{L}(D_\theta)}{\partial Z}$)
2. For this value of $Z$, the derivative of the discriminator's loss wrt. $\theta$ is equal to the derivative for the MaxEnt IRL objective. (Compute $\frac{\partial \mathcal{L}(D_\theta)}{\partial \theta}$ and $\frac{\partial \mathcal{L}_{cost}(\theta)}{\partial \theta}$)
3. The generator's loss is exactly equal to the cost $c_\theta$ minus the entropy of $q(\tau)$.

