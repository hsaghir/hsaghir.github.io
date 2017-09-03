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

The basic goal of generative models is to perform density estimation meaning that we want to take in a bunch of training data and fit a probability density function to it. We Write down a parameterized density function (a function that assigns a probability to all data points) and find parameters that maximize the likelihood that the density function assigns to data. Most of the time though we can't write the likelihood explicitly so we end up maximizing other functions that approximate likelihood (e.g. ELBO).
    + Explicit density: 
        * tractable denisty:
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
    + Missing data imputation/Semi-supervised learning: [Yeh et al 2016](https://arxiv.org/abs/1607.07539) used generative models to do image inpainting
    + Train model to learn multiple correct answers: [Lotter et al 2016] use generative models to predict next frame video prediction. based on the rotation of the face, multiple answers might be correct. If we train using MSE loss, multiple futures are combined to a single answer and result in a blurry average of all. Using adversarial loss they were able to capture the distribution over all possible future images and generate a sharp prediction.
    + Realistic generation tasks: generate media and special effects (art applications). [Zhu et al 2016] created iGAN where the model will find the nearest neighbor picture from a simple drawing. CycleGAN translates horses to zebras in an unsupervised fashion and Image synthesis creates a picture conditioned on a sentence.
    + simulation by prediction: instead of using a computational model for simulating a phenomenon, [de Olivera et al 2017] use generative models to predict the output of the simulator conditioned on simulation parameters.
    + Learn useful embeddings: for example word embeddings. [Radford et al 2015] use DCGAN to find embedding for images that enables linear algebra on images. This can be useful for image search, information retrieval etc. [Chen et al 2016] introduce infoGAN to learn controlable and interpretable latent codes. 
        * [Nguyen et al 2016 ] introduce PPGN and combine adversarial training, moment matching, denoising autoencoders (DAE) and Monte Carlo sampling (Langevin sampling) to generate high quality high-res images conditioned on category or caption on ImageNet.
            - Basic idea is that they have a variation of a Markov chain that follows the gradient of the log density (Langevin sampling). It uses a DAE to estimate the gradient instead of explicitly computing the gradient (synthetic gradient?). And finally use a special DAE that has been trained with multiple losses including a GAN loss to obtain best results. 



explore variations in data, to reason about the structure and behaviour of the world, and ultimately, for decision-making. Deep generative models have widespread applications including those in image denoising and in-painting, data compression, scene understanding, representation learning, 3D scene construction, semisupervised classification, and hierarchical control, amongst many others.

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

-  We use probability theory as a tool to start the modelling process. Let's assume each variable, $$x_i$$, as a random variable (meaning that it is actually a distribution and the possible values are random draws from the said distribution). In this setup, the model is the joint probability distribution of all random variables, i.e. $$p(x_1,x_2, .., x_n) = \prod_i p(x_i)p(x_i|x_m)$$. 

- Finding the true joint distribution is no easy task. We appeal to modelling to find a good approximation for this joint distribution. There are two approaches we can take to solving this. 
    + non-parametric modelling
    + parametric modelling

## Non-parametric modelling

- Won't be focusing on non-parametric methods in this post but here is an example non-parametric approach to finding the joint distribution.

- Kernel Density Estimation:  Let's assume that variables are all independent and identically distributed (iid), and let's further assume a form of a kernel density function as the distribution of a random variable. The joint distribution will simply be the product of all kernel density functions. This is called non-parametric since the method can grow in complexity with data and not because it doesn't have parameters 

## Parametric modelling 
-  The first prior we encode into our modelling process is that we assume the data is coming from a generative process that we define. For example, if observations are $$X = {x_1, x_2, .., x_n}$$, we might assume that each data point is the outcome of a hidden local variable variable. This is a  simple gaenerative model with a set of latent variables $$Z = {z_1, z_2,...,z_n}$$.

- Now with this assumed generative model, we want to fit the model to data and infer the latent variables, $$z_i$$, from the visible variables $$x_i$$. The Bayes rule shows a way for performing inference through the notion of posterior. The idea is that we have some initial belief, we see data, then our beliefs evolve to posterior beliefs based on our observations. This is formulized as $$p(z|x) = \frac{p(x|z)p(z)}{p(x)}$$ where $$p(z)$$ is the prior on latent variables, $$p(z|x)$$ is the posterior belief on latent variables after observations, $$p(x|z)$$ is the likelihood of observations under the model, and $$p(x)$$ is the marginalized likelihood of observations under the model alternatively called evidence (when we inegrate out all latent variables).

- In order to be able to use the Bayesian formula for inference and learning, we usually assume a functional form for the beliefs mentioned above (probability distributions). This functional form is called density function since it shows how dense the probability of the domain of the random variable of interest is. The density functions usually come with a set of sufficient statistics that while known, would completely define the density function on the variable domain. For example, the sufficient statistics for the Normal distribution density function are mean and variance parameters. We can choose to further parameterize these sufficient statistics parameters to give more expressability to our model.

- Since in the Bayesian world every parameter is interpreted as a belief (probability distribution), we can also treat the parameters in a Bayesian way, express them with another level of a density functions and parameterize them and then again treat the parameters of the parameters as belief and so on in an endless loop of Bayesian utopia! However, like all other good things, this endless loop has to end which means that we have to be practical Bayesians. 


### Types of models

#### 1. Fully-observed models: 
These Models observe data directly without introducing any new local latent variables but have a deterministic internal hidden representation of the data. Let observations be $$X = {x_1, x_2, .., x_n}$$ as random variables. We assume a prior for each variable e.g. $$x_i ~ Cat(\pi|\pi(x_1,...,x_i))$$. The joint distribution is described as $$p(X) = p(x_1)p(x_2|x_1)...p(x_n|(x_1,...,x_n))$$. This will be a directed model of only observed variables.
    + all conditional probabilities (e.g. $$p(x_i|(x_1,...,x_{i-1}))$$) may be described using deep neural nets (e.g. an LSTM).  
    + They can work with any data type.
    + likelihood function is explicit in terms of observed variables. Therefore, the log-likelihood is directly computable without any approximations. 
    + They are easy to scale up.
    + limited by the dimension of their hidden representations (assumed degree of conditional dependencies).
    + generation can be slow due to sequential assumption.
    + for undirected models parameter learning is difficult due to the need for calculation of the normalizing constant. 

![alt text](/images/Generative_models/fully_observed_models.png "map of instances of fully observed models")

- For example, in the case of char-RNN, the number of RNN unrolling steps is the degree of conoditional probabilities. If we put a softmax layer on the output, the decision of the RNN will be a probability distribution on possible outputs and the rest of the RNN can be deterministic. 

#### 2. Transformation models (Implicit generative models): 
These models Model data as a transformation of an unobserved noise source using a deterministic function. Their main properties are that we can sample from them very easily, and that we can take the derivative of samples with respect to parameters. Let data samples be $$X = {x_1, x_2, ...,x_n}$$. We model the data as a deterministic transformation of a noise source i.e. $$ Z ~ N(0,I); X = f(Z; \theta)$$
    + The transformation is usually a deep neural network. 
    + It's easy to compute expectations without knowing final distribution due to the easy sampling and Monte Carlo averages.
    + It's difficult to maintain invertability. 
    + Challenging optimization.
    + They don't have an explicit likelihood function. Therefore, difficult to calculate marginal log-likelihood for model comparison. 
    + difficult to extend to generic data types.

![alt text](/images/Generative_models/transformation_models.png "map of instances of transformation models")

These models are usually used as generative models to model distributions of observed data. They can also be used to model distributions over latent variables as well in approximate inference (e.g. adversarial autoencoder).
 
#### 3. Latent variable models (Explicit Probabilistic graphical models): 
These models introduce an unobserved local random variable for every observed data point. This is in contrast with fully-observed models that do not impose such explicit assumptions on the data. They are easy to sample from, include hierarchy of causes believed. The latent variable structure encode our assumptions about the generative process of the data. Let data be $$X = {x_1, x_2, ..., x_n}$$. We assume a generative process for example a latent Gaussian model $$z ~ N(0,I); x|z = N(\mu(z), \Sigma(z)); p(X,Z) = p(Z)p(X|Z) $$. 
    + Conditional distributions are usually represented using deep neural nets
    + easy to include hierarchy, depth, and the believed generating structure 
    + don't assume an order of coniditional independance unlike fully-observed models. If we marginalize latent variables, we induce dependancies between observed variables similar to fully-observed models.
    + Latent variables can act as a new representation for the data.
    + Directed models are easy to sample from
    + Difficult to calculate marginalized log-likelihood and involves approximations
    + Not easy to specify rich approximate posterior for latent variables.
    + inference of latents from observed data is difficult in general
    

![alt text](/images/VAE_intuitions/latent_var_model.png "a latent variable model")

- In latent variable models, the probabilistic nature of the model is evident in both stochastic latent variables and stochastic observation variables. This means that we assume a latent node is a probability distribution and an observation node is also a probability distribution. The probability distribution function of the latent before inference is called prior and after inference posterior. The probability distribution of the observed nodes is called likelihood probability density function. Therefore, in such models, there is an explicit likelihood probability distribution function for observable nodes. However, this likelihood function is intractable in deep latent variable models. If we marginalize the latent nodes, we get a probability distribution which we sample to get observations. This is in contrast with implicit models. In implicit models, observations are not random variables. Observations are deterministic nodes, and therefore, there is no likelihood explicit probability density function. The likelihood is implicit in a deterministic function mapping a sample from a random variable (noise source) to an observation. 

![alt text](/images/Generative_models/Latent_variable_models.png "map of latent variable models")

### Parameter Learning
- Let's get back to the Bayesian formula. If we write the probability distributions as parameterized density functions, we will end up with an equation with unknown parameters. So the inference task is now transformed into the problem of finding parameter values from observations. 

- how to learn a parametric model based on training data? There are three main approaches to this problem. The first approach, called maximum likelihood, says that one should choose parameter values that maximize the components that directly sample data (i.e. the likelihood or the probability of the data under the model, this corresponding to the first term in the ELBO, or an error function). In a sense, they provide the best possible fit to the data. As a result, there is a tendency towards overfitting. If we have a small amount of data, we can come to some quick conclusions due to small sample size.

The second is the Baysian approach which is a belief system that suggests every variable including parameters should be beliefs (probability dist) not a single value. It provides an alternative approach to maximum likelihood, that does not make such a heavy commitment towards a single set of parameter values, and incorporates prior knowledge. In the Bayesian approach, all parameter values are considered possible, even after learning. No single set of parameter values is selected. Instead of choosing the set of parameters that maximize the likelihood, we maintain a probability distribution over the set of parameter values. The ELBO represents Bayesian approach where the second term represents balancing a prior distribution with the model's explanation of the data. The third approach, maximum a-posteriori (MAP), is a compromise between the maximum likelihood and the Bayesian belief system. The MAP estimate is the single set of parameters that maximize the probability under the posterior and is found by solving a penalized likelihood problem. However, it remedies the overfitting problem a bit by veering away from the maximum likelihood estimate (MLE), if MLE has low probability under the prior. As more data are seen, the prior term will be swallowed by the likelihood term, and the estimate will look more and more like the MLE.

- So the three schools of thoughts for how the parameter set should be chosen:
    + The simplest and most obvious school of thought is that parameters are not distributions and should be chosen in a way that will maximize the likelihood of observations under the model. This gives rise to the maximum likelihood parameter learning concept. 

    + The Bayesian school of thought obviously believes that the parameters are distributions themselves and thus we need to infer distributions for parameters too and not just learn single values.

    +  Another school of thought tries to balance the above two ideas by professing that although the parameter maybe distributions but we want to be practical so instead of infering a distribution for each parameter we choose the single parameter values that will maximize the posterior belief of latent values. This gives rise to the Maximum a Poseriori (MAP) inference concept. 

Note that the loss function (i.e. the distance measure) in optimization (i.e. maximum likelihood) is the root of the all problems since optimization's only objective is to reduce the loss. If the loss is not properly defined, the model can't learn well and if the loss function does not consider the inherent noise of the data (i.e. regularization or MAP), the model will eventually overfit to noise in the data and reduce generalization. Therefore, the loss function (distance measure) is very important and the reason why GANs work so well is that they don't explicitly define a loss function and learn it instead! The reason that Bayesian approach prevents overfitting is because they don't optimize anything, but instead marginalise (integrate) over all possible choices. The problem then lies in the choice of proper prior beliefs regarding the model.

- Vanilla maximum likelihood learning and MAP inference are simple and fast but too biased and approximate. So we usually would like to have one more level of full Bayesian goodness and obtain distributions. If we take the Bayesian school of thought and try to infer the posterior belief of latent variables, we need to be able to calculate the marginal likelihood term $$p(x)$$. Unfortunately, this term involves an integration which is intractable for most interesting models. 

- Since the posterior $$p(z|x)$$ is intractable in this model, we need to use approximate inference for latent variable inference and parameter learning. Two common approaches are:
    + MCMC inference: assymptotically unbiased but it's expensive, hard to assess the Markov chain convergence and manifests large variance in gradient estimation.
    + Variational inference: fast, and low variance in gradient with reparameterization trick. well suited to the use of deep neural nets for parameterizing conditional distributions. Also well-suited to using deep neural networks to amortize the inference of local latent variables $${z_1,..., z_n}$$ with a single deep neural network. 


## Bayesian network vs Parameterization network
As can be seen, the probabilistic graphical model (Bayesian network), the generative model factorization, the inference model factorization, the ELBO objective, and even the variational inference procedure, don't include any information about how the conditional distributions in the graph are parameterized. Therefore, in implementation we can seperate these two parts and have two parts in the deep generative model implementation code:
1. The graphical model, inference and generative network factorizations, the ELBO, and even the variational inference procedures. 
2. The parameterizations using neural network

## Model vs Action
- We can build a generative model by pairing a model, inference scheme and build an algorithms for density estimation of the model. There are therefore, two distinct things we might be interested in, first is building a model. Second is what to do with this model. If we want to make decisions and act upon the model then we face the problem of reinforcement learning. We build a model first and then put it in an environment to take actions and get rewards in order to evolve. 
