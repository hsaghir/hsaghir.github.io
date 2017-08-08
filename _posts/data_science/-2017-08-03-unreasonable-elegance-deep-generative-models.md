---
layout: article
title: The unreasonable elegance of deep generative models
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

"What I cannot create, I do not understand" ~ Richard Feynman

If generative models had a motto, it has to be this quote from Richard Feynman. Generative models try to solve unsupervised learning. 

Generative models can be used to explore variations in data, to reason about the structure and behaviour of the world, and ultimately, for decision-making. Deep generative models have widespread applications including those in density estimation, image denoising and in-painting, data compression, scene understanding, representation learning, 3D scene construction, semisupervised classification, and hierarchical control, amongst many others.

![alt text](/images/VAE_intuitions/latent_var_model.png "a latent variable model")

## Statistical modelling

Finding insights in data, prediction, modeling the world and everything that we can do with data involves finding the probability distribution underlying our data P(x). Therfore, We are interested in finding the true probability distribution of our data P(x) which is unknown. We usually use the scientific method of probabilistic pipline to solve this problem i.e.:

1. Knowledge and questions we want answered
2. make assumptions and formulate a model (using probabilitic graphical models, deep nets as functions, etc)
3. Fit the model with data, find the parameters of the model, find patterns in data (Inference)
4. predict and explore
5. criticize and revise the model

The statistical modelling procedure involves introduction of some hidden variables, z, and a mixing procedure for them in a way which we believe will lead to an estimation for the true probability distribution of the data P(x). These hidden variables and their structure construct our model. Therefore, the new problem under our model is the joint distribution P(x,z) of the hidden variables z, and the observed variables x. 

The joint distribution P(x,z) can be thought of as a combination of other simpler probability distributions P(x,z)=P(x|z)P(z). The way they are combined is through a hierarchy of component distributions. So first a sample from the top distribution over hidden variables P(z) (i.e. prior) chooses the component that should produce data, then the corresponding component P(x|z) (i.e. likelihood) produces a sample x. This makes it easier to express the complex probability distribution of the observed data P(x) using a model P(x,z)=P(x|z)P(z). It is important to note that the real procedure for producing a sample x is unknown to us and the model is merely an attempt to find an estimation for the true distribution.

At the beginning, the probability of choosing a component in the mixture is based on a very crude assumption for the general shape of such a distribution (i.e. prior P(z)) and we don't know the specific value of the parameters in the assumed structure for the probability model. We would like to find these unknowns based on the data. Finding the parameters to form a posterior belief is called inference and is the key algorithmic problem which tells us what model says about the data. In the probabilistic pipleline, we will then criticize the model we have built and might modify it and solve the problem again until we arrive at a model that satisfies our needs. 

## Problem setup (Latent variable generative model):
-  Let's assume our data as $$X = {x_1,x_2,...,x_n}$$. We are interested in performing inference, two-sample tests, prediction, generate more data like it, understand it, etc. Similar to what we do in all areas of science, we try to model the data to be able to answer such questions.  

-  We use probability theory as a tool to start the modelling process. Let's assume each variable, $$x_i$$, as a random variable (meaning that it is actually a distribution and the possible values are random draws from the said distribution). In this setup, the model is the joint probability distribution of all random variables, i.e. $$p(x_1,x_2, .., x_n) = \prod_i p(x_i)p(x_i|x_m)$$ where $$m$$ means all variables except for $$i$$. 

- Finding the true joint distribution is no easy task. We appeal to modelling to find a good approximation for this joint distribution. There are two approaches we can take to solving this. 
    + non-parametric modelling
    + parametric modelling

## Non-parametric modelling

- Won't be focusing on non-parametric methods in this post but here is an example non-parametric approach to finding the joint distribution.

- Kernel Density Estimation:  Let's assume that variables are all independent and identically distributed (iid), and let's further assume a form of a kernel density function as the distribution of a random variable. The joint distribution will simply be the product of all kernel density functions. This is called non-parametric since the method can grow in complexity with data and not because it doesn't have parameters 

## Parametric modelling 
-  The first prior we encode into our modelling process is that we assume the data is coming from a generative process that we define. 

- Second, the way parametric modelling works is that we usually assume a functional form for the beliefs or probability distributions. This functional form is called density function that is usually parameterized using a few parameters (e.g. Gaussian distributions is parameterized using sufficient statistics parameters, mean and variance).  

- We can also treat the parameters in a Bayesian way and assume a density and parameterize the parameters and again treat the parameters of the parameters as belief and parameterize them and so on in an endless loop of Bayesian utopia! However, like all other good things, this endless loop has to stop somewhere as well which means that we have to be practical Bayesians. 


### Types of models

#### 1. Fully-observed models: 
These Models observe data directly. Char-RNNs, PixelRNN, RBMs and autoregressive models are of this type. These models don't introduce any new unobserved local variables for the observations but do have a deterministic internal hidden representation of the data which acts like a memory of previous data in a sequence based on the degree of conditionals assumed. For example, in the case of char-RNN, the number of unrolling the RNN is the number of conditional probabilities. If we put a softmax layer on the output, the decision of the RNN will be a probability distribution on possible outputs and the rest of the RNN can be deterministic. We sample from the output softmax distribution based on deterministic hidden variable and input sequence. They can directly model relationships between data points in a sequence due to their memory. But, they are limited by the dimension of their hidden representations or put another way, limited by the assumed conditional dependencies between a point and the number of its previous points. 

#### 2. Latent variable models (Explicit Probabilistic graphical models): 
These models introduce an unobserved random variable for every observed data point. Gaussian mixture models (GMM) are of this type. The latent variables can be used to explain hidden causes and factors of variation in the data. Therefore, such models incorporate our assumptions about the hidden causes of the data in the form of their graphical models. This is in contrast with fully-observed models that do not impose such explicit assumptions on the data. They are easy to sample from, include hierarchy of causes believed.

- In latent variable models, the probabilistic nature of the model is evident in both stochastic latent variables and stochastic observation variables. This means that we assume a latent node is a probability distribution and an observation node is also a probability distribution. The probability distribution function of the latent before inference is called prior and after inference posterior. The probability distribution of the observation node is called likelihood probability density function. Therefore, in such models, there is an explicit likelihood probability distribution function for observable nodes. If we marginalize the latent nodes, we get a probability distribution which we sample to get observations. This is in contrast with implicit models explained below. In implicit models, observations are not random variables. Observations are deterministic nodes, and therefore, there is no likelihood explicit probability density function. The likelihood is implicit in a deterministic function mapping a sample from a random variable (noise source) to an observation. 

#### 3. Transformation models (Implicit generative models): 
These models Model data as a transformation of an unobserved noise source using a deterministic function. Their main properties are that we can sample from them very easily, and that we can take the derivative of samples with respect to parameters. They are used as generative models  to model distributions of observed data but can be used for approximate inference as well where one uses them to model distributions over latent variables.

### Parameter Learning
- Let's get back to the Bayesian formula. If we write the probability distributions as parameterized density functions, we will end up with an equation with unknown parameters. So the inference task is now transformed into the problem of finding parameter values from observations. 

- how to learn a parametric model based on training data? There are three main approaches to this problem. The first approach, called maximum likelihood, says that one should choose parameter values that maximize the components that directly sample data (i.e. the likelihood or the probability of the data under the model, this corresponding to the first term in the ELBO, or an error function). In a sense, they provide the best possible fit to the data. As a result, there is a tendency towards overfitting. If we have a small amount of data, we can come to some quick conclusions due to small sample size.

Baysian approach is a belief system that suggests every variable including parameters should be beliefs (probability dist) not a single value. It provides an alternative approach to maximum likelihood, that does not make such a heavy commitment towards a single set of parameter values, and incorporates prior knowledge. In the Bayesian approach, all parameter values are considered possible, even after learning. No single set of parameter values is selected. Instead of choosing the set of parameters that maximize the likelihood, we maintain a probability distribution over the set of parameter values. The ELBO represents Bayesian approach where the second term represents balancing a prior distribution with the model's explanation of the data. The third approach, maximum a-posteriori (MAP), is a compromise between the maximum likelihood and the Bayesian belief system. The MAP estimate is the single set of parameters that maximize the probability under the posterior and is found by solving a penalized likelihood problem. However, it remedies the overfitting problem a bit by veering away from the maximum likelihood estimate (MLE), if MLE has low probability under the prior. As more data are seen, the prior term will be swallowed by the likelihood term, and the estimate will look more and more like the MLE.

- So the three schools of thoughts for how the parameter set should be chosen:
    + The simplest and most obvious school of thought is that parameters are not distributions and should be chosen in a way that will maximize the likelihood of observations under the model. This gives rise to the maximum likelihood parameter learning concept. 

    + The Bayesian school of thought obviously believes that the parameters are distributions themselves and thus we need to infer distributions for parameters too and not just learn single values.

    +  Another school of thought tries to balance the above two ideas by professing that although the parameter maybe distributions but we want to be practical so instead of infering a distribution for each parameter we choose the single parameter values that will maximize the posterior belief of latent values. This gives rise to the Maximum a Poseriori (MAP) inference concept. 

Note that the loss function (i.e. the distance measure) in optimization is the root of the all problems since optimization's only objective is to reduce the loss. If the loss is not properly defined, the model can't learn well and if the loss function does not consider the inherent noise of the data (i.e. regularization), the model will eventually overfit to noise in the data and reduce generalization. Therefore, the loss function (distance measure) is very important and the reason why GANs work so well is that they don't explicitly define a loss function and learn it instead! The reason that Bayesian approach prevents overfitting is because they don't optimize anything, but instead marginalise (integrate) over all possible choices. The problem then lies in the choice of proper prior beliefs regarding the model.

- Vanilla maximum likelihood learning and MAP inference are simple and fast but too biased and approximate. So we usually would like to have one more level of full Bayesian goodness and obtain distributions. If we take the Bayesian school of thought and try to infer the posterior belief of latent variables, we need to be able to calculate the marginal likelihood term $$p(x)$$. Unfortunately, this term involves an integration which is intractable for most interesting models. 

- Since the posterior $$p(z|x)$$ is intractable in this model, we need to use approximate inference for latent variable inference and parameter learning. Two common approaches are:
    + MCMC inference: assymptotically unbiased but it's expensive, hard to assess the Markov chain convergence and manifests large variance in gradient estimation.
    + Variational inference: fast, and low variance in gradient with reparameterization trick. well suited to the use of deep neural nets for parameterizing conditional distributions. Also well-suited to using deep neural networks to amortize the inference of local latent variables $${z_1,..., z_n}$$ with a single deep neural network. 

## Model vs Action
- We can build a generative model by pairing a model, inference scheme and build an algorithms for density estimation of the model. There are therefore, two distinct things we might be interested in, first is building a model. Second is what to do with this model. If we want to make decisions and act upon the model then we face the problem of reinforcement learning. We build a model first and then put it in an environment to take actions and get rewards in order to evolve. 
