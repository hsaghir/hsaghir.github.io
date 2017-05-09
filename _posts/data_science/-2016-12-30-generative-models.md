---
layout: article
title: An intuitive primer on deep generetive models
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

"What I cannot create, I do not understand" ~ Richard Feynman

If generative models had a motto, it has to be this quote from Richard Feynman. Generative models try to solve unsupervised learning. 


They can be used to explore variations in data, to reason about the structure and behaviour of the world, and ultimately, for decision-making. Deep generative models have widespread applications including those in density estimation, image denoising and in-painting, data compression, scene understanding, representation learning, 3D scene construction, semisupervised classification, and hierarchical control, amongst many others.

The objective function we use for training a probabilistic model should match the way we ultimately want to use the model. Very often people don't make this distinction clear when talking about generative models which is one of the reasons why there is still no clarity about what different objective functions do. In "How (not) to Train your Generative Model: Scheduled Sampling, Likelihood, Adversary?", Ferenc argues that when the goal is to train a model that can generate natural-looking samples, maximum likelihood is not a desirable training objective. Maximum likelihood is consistent so it can learn any distribution if it is given infinite data and a perfect model class. However, under model misspecification and finite data (almost always), it has a tendency to produce models that overgeneralise.

Generative modelling is about finding a probabilistic model $$Q$$ that in some sense approximates the natural distribution of data $$P$$. So if we want to train models that produce nice samples, my recommendation is to try to use $$KL[Q∥P]$$ as an objective function or something that behaves like it since perceived quality of each sample is related to the $$ −\log P(x)$$ under the human observers' subjective prior of stimuli. On the other hand, Maximum likelihood is roughly the same as minimising $$KL[P∥Q]$$. Both divergences ensure consistency, However, they differ fundamentally in the way they deal with finite data and model misspecification. $$KL[P∥Q]$$ tends to favour approximations $$Q$$ that overgeneralise $$P$$. Practically this means that the model will occasionally sample unplausible samples that don't look anything like samples from $$P$$. $$KL[Q‖P]$$ tends to favour under-generalisation. Practically this means that $$KL[Q‖P]$$ will try to avoid introducing unplausible samples, sometimes at the cost of missing the majority of plausible samples under $$P$$. In other words: In yet other words: $$KL[P‖Q]$$ is an optimist, $$KL[Q‖P]$$ is a pessimist.

In unsupervised learning we are given indepenent data samples from some underlying data distribution P which we don't know, and our goal is to come up with an approximate distribution Q that is as close to P as possible, only using the dataset. Often, Q is chosen from a parametric family of distributions, and our goal is to find the optimal parameters so that the distribution P is best approximated.

The central issue of unsupervised learning is choosing an appropriate objective function, that appropriately measures the quality of our approximation, and which is tractable to compute and optimise when we are working with complicated, deep models. Typically, a probability model is specified as q(x;θ)=f(x;θ)/Zθ, where f(x;θ) is some positive function parametrised by θ. The denominator Zθ is called the normalisation constant which makes sure that q is a valid probability model: it has to sum up to one over all possible configurations of x. The central problem in unsupervised learning is that for the most interesting models in high dimensional spaces, calculating ZθZθ is intractable. So The most straightforward choice of objective functions, i.e. marginal likelihood (log likelihood/evidence), is intractable in most cases.Therefore people come up with:

1-Alternative objective functions and fancy methods to deal with intractable models such as:
- density ratio estimation (like  adversarial training, maximum mean discrepancy, discussed later), 
- pseudolikelihood: The twist is this: instead of thinking about fitting a probabilistic model p(x;θ) to data x, you learn a joint probability distribution p(x,z;θ) of the data x and it's noise-corrupted version z. The noise corruption is artificially introduced by us, following a corruption distribution. The point is, if you learn the joint model p(x,z;θ), that also implies a generative model p(x;θ)=∫p(x,z;θ). To fit the joint model to observations (xi,zi), we use score matching with the pseudolikelihood scoring rule as objective function.

- MSE or maximum spacing estimation: if you squash a random variable X through its cumulative distribution function F, the transformed variable F(X) follows a standard uniform distribution between 0 and 1. Furthermore, we can say that for any random variable X, its CDF (F) is the only monotonic, left-continuous function with this property. Using this observation one can come up with objective functions for univariate density estimation. Let's assume we have a parametric family of distributions described by their CDFs Fθ, and we want to find the θ that best fits our observations Xi. If Fθ is close to the true CDF of the data distribution, we would expect Fθ(xi) to follow a uniform distribution. Therefore, we only need a way to measure the uniformity of Fθ(xi) across samples. Maximum Spacing Estimation (MSE) uses the geometric mean of spacing between ordered samples as an objective function. 

- Alternative optimisation methods or approximate inference methods such as contrastive divergence or variational Bayes.

2- Interesting models that have tractable normalisation constants:
-  MADE: Masked Autoencoder for Distribution Estimation: This paper sidesteps the high dimensional normalisation problem by restricting the class of probability distributions to autoregressive models! In a model like this, we only need to compute the normalisation of each q(xd|x1:d−1;θ) term, and we can be sure that the resulting model is a valid model over the whole vector x. But as normalising these one-dimensional probability distributions is a lot easier, we have a whole range of interesting tractable distributions at our disposal.

- Another idea in representation learning is to map data to a latent representation. While the Data can have arbitrarily complex distribution along some complicated nonlinear manifold, we want the computed latent representations to have a nice distribution, like a multivariate Gaussian. "Deep Unsupervised Learning using Nonequilibrium Thermodynamics" takes this idea very explicitly. If you take any data point, and apply Brownian motion-like stochastic process to this, you will end up with a standard Gaussian distributed variable, due to the stationarity of the Brownian motion. This is a diffusion process. Now the trick is to train a dynamical system to inverts this random walk, to be able to reconstruct the original data distribution from the random Gaussian noise. Amazingly, this works, and the traninig objective becomes very similar to variational autoencoders. Therefore, The information about the data distribution is encoded in the approximate inverse dynamical system.


#
The main ideas that repeatedly come up as one explores deep generative models are  Bayesian deep learning, variational approximations, memoryless and amortised inference, and stochastic gradient estimation. So the hope is that by having an intuition of these ideas before the mathematics, one can understand deep generative models. 

## treat deep networks as functions, Convolutional, recurrent, LSTM, GRU, fully-connected, etc

## design of probabilistic models
Three class of probabilistic generetive models exist. 

1. Fully-observed models: These Models observe data directly. Char-RNNs, PixelRNN, RBMs and autoregressive models are of this type. These models don't introduce any new unobserved local variables for the observations but do have a deterministic internal hidden representation of the data which acts like a memory of previous data in a sequence based on the degree of conditionals assumed. For example, in the case of char-RNN, the number of unrolling the RNN is the number of conditional probabilities. If we put a softmax layer on the output, the decision of the RNN will be a probability distribution on possible outputs and the rest of the RNN can be deterministic. We sample from the output softmax distribution based on deterministic hidden variable and input sequence. They can directly model relationships between data points in a sequence due to their memory. But, they are limited by the dimension of their hidden representations or put another way, limited by the assumed conditional dependencies between a point and the number of its previous points. 

-  MADE: Masked Autoencoder for Distribution Estimation: Autoregressive models are used a lot in time series modelling and language modelling: hidden Markov models or recurrent neural networks are examples. There, autoregressive models are a very natural way to model data because the data comes ordered (in time).

What's weird about using autoregressive models in this context is that it is sensitive to ordering of dimensions, even though that ordering might not mean anything. If xx encodes an image, you can think about multiple orders in which pixel values can be serialised: sweeping left-to-right, top-to-bottom, inside-out etc. For images, neither of these orderings is particularly natural, yet all of these different ordering specifies a different model above.

But it turns out, you don't have to choose one ordering, you can choose all of them at the same time. The neat trick in the masking autoencoder paper is to train multiple autoregressive models all at the same time, all of them sharing (a subset of) parameters θθ, but defined over different ordering of coordinates. This can be achieved by thinking of deep autoregressive models as a special cases of an autoencoder, only with a few edges missing.

2. Latent variable models (Explicit Probabilistic graphical models): These models introduce an unobserved random variable for every observed data point. Gaussian mixture models (GMM) are of this type. The latent variables can be used to explain hidden causes and factors of variation in the data. Therefore, such models incorporate our assumptions about the hidden causes of the data in the form of their graphical models. This is in contrast with fully-observed models that do not impose such explicit assumptions on the data. They are easy to sample from, include hierarchy of causes believed.

- In latent variable models, the probabilistic nature of the model is evident in both stochastic latent variables and stochastic observation variables. This means that we assume a latent node is a probability distribution and an observation node is also a probability distribution. The probability distribution function of the latent before inference is called prior and after inference posterior. The probability distribution of the observation node is called likelihood probability density function. Therefore, in such models, there is an explicit likelihood probability distribution function for observable nodes. If we marginalize the latent nodes, we get a probability distribution which we sample to get observations. This is in contrast with implicit models explained below. In implicit models, observations are not random variables. Observations are deterministic nodes, and therefore, there is no likelihood explicit probability density function. The likelihood is implicit in a deterministic function mapping a sample from a random variable (noise source) to an observation. 

- An example is LDA (latent Drichlet allocation), which is basically a mixture of multinomial distributions (topics) in a document. Its difference from a typical mixture is that each document has its own mixture proportions of the topics but the topics (multinomials) are shared across the whole collection (i.e. a mixed membership model). 


3. Transformation models (Implicit generative models): These models Model data as a transformation of an unobserved noise source using a deterministic function. Their main properties are that we can sample from them very easily, and that we can take the derivative of samples with respect to parameters. They are used as generative models  to model distributions of observed data but can be used for approximate inference as well where one uses them to model distributions over latent variables.

- Real-valued non-volume preserving transformation (Real NVP) is an invertable transformation model. The generative procedure from the model is very similar to the one used in Variational Auto-Encoders and Generative Adversarial Networks: sample a vector z from a simple distribution (here a Gaussian) and pass it through the generator network g to obtain a sample x=g(z). From the generated samples, it seems the model was able to capture the statistics from the original data distribution. For example, the samples are in general relatively sharp and coherent and therefore suggest that the models understands something more than mere correlation between neighboring pixels. This is due to not relying on fixed form reconstruction cost like squared loss on the data level. The models seems also to understand to some degree the notion of foreground/background, and volume, lighting and shadows. The generator network g has been built in the paper according to a convolutional architecture, making it relatively easy to reuse the model to generate bigger images. As the model is convolutional, the model is trying to generate a “texture” of the dataset rather than an upsampled version of the images it was trained on. This explains why the model is most successful when trained on background datasets like LSUN different subcategories. This sort of behaviour can also be observed in other models like Deep Convolutional Generative Adversarial Networks.

- Another example, Deep Unsupervised Learning using Nonequilibrium Thermodynamics. What we typically try to do in representation learning is to map data to a latent representation. While the Data can have arbitrarily complex distribution along some complicated nonlinear manifold, we want the computed latent representations to have a nice distribution, like a multivariate Gaussian. This paper takes this idea very explicitly using a stochastic mapping to turn data into a representation: a random diffusion process. If you take any data, and apply Brownian motion-like stochastic process to this, you will end up with a standard Gaussian distributed variable due to the stationarity of the Brownian motion. Now the trick the authors used is to train a dynamical system (a Markov chain) to inverts this random walk, to be able to reconstruct the original data distribution from the random Gaussian noise. Amazingly, this works, and the traninig objective becomes very similar to variational autoencoders. 

- GANs are also of this type of (transformation) generative models where a random Gaussian noise is transformed into data (e.g.an image) using a deep neural network. These models also assume a noise model on the latent cause. The good thing about such models is that it's easy to sample and compute expectation from these models without knowing the final distribution. Since classifiers are well-develped, we can use our knowledge there for density ratio estimation in these models. However, these models lack noise model and likelyhood. It's also difficult to optimize them.  Any implicit model can be easily turned into a prescribed model by adding a simple likelihood function (noise model) on the generated outputs but models with likelihood functions also regularly face the problem of intractable marginal likelihoods.But the specification of a likelihood function provides knowledge of data marginal p(x) that leads to different algorithms by exploiting this knowledge, e.g., NCE resulting from class-probability based testing in un-normalised models, or variational lower bounds for directed graphical models.

- Think of GANs as a density estimation problem. There are two high dimensional surfaces. One of them is P(x) which is unknown, we want to estimate, and we only have some examples of. The other is a maleable goo from a family of surfaces that we want to match onto the unknown density to estimate it. 

- Variational inference tries to solve this by assuming a goo Q(z|x) defined by a parametric model from the exponential family, a fixed distance metric (i.e. KL divergence) to make a force field, and letting the physics takes its course (running SGD on parameters). MCMC methods don't use any model and instead try to just throw random points onto the surface and see where it ends up and use samples to estimate the shape of the surface P(x). Variational inference can't match all the intricacies of the high dimensional surface while MCMC is very costly since using samples to estimate a high dimensional surface is a fools errand!

- GANs have a more elegant approach, they define the goo to be a transformation model Q(x) that doesn't have a tractable likelihood function (can get very complex) but is very easy to sample from instead. Instead of using a rigid distance metric, GANs actively define an adaptive distance metric that tries to capture the intricacies of the unknown surface in every iteration. They do this by using the insight that classifiers can actually find a surface between two data sources easily and we have two data sources from the unknown surface and the goo. Therefore, the surface that the classifier finds to distinguish the data sources, captures the intricacies of the unknown surface P(x). So, if after each iteration that the classifier finds the optimum surface between the unknown and the goo, we reshape the goo to beat the classifier surface, the goo will very well take the shape of the surface at the end. This is a very clever way of matching the goo to the unknown surface P(x) that basically uses an adaptive distance metric using the distance to the classifer surface instead of the unknown surface. the distance to the classifier surface sort of hand holds the goo optimization step by step until it gets as close as possible to the unkown surface. Another point is that we only have a limited number of examples from the unknown surface, so we actually do the above process stocastically in batches of examples from unknown and the goo surfaces.



- Another type of implicit models, are simulators that transform a noise source into an output. Sampling from such simulators is easy but explicitly calculating a likelihood distribution function is usually no possible if the simulator is not invertable and mostly intractable. An example is a physical simulator based on differential euqations derived from eqautions of motions.

4. Deep Implicit Models (DIM): These are stacked transformation models in a graph. For example, if a GAN is creating the noise code for the next GAN, it's a deep implicit model since likelihoods are not explicitly defined. This enables building complex densities in a hierarchical way similar to probabilistic graphical models. 

- Inference in these models encounters two problems. First, like other inference problems marginal probability is intractable. Second, in transformation models likelihood is also intractable. We thus turn to variational inference with density ratio estimation instead of density estimation.

- A DIM is simply a deep neural network with random noise injected at certain layers. An additional reason for deep latent structure appears from this perspective: training may be easier if we inject randomness at various layers of a neural net, rather than simply at the input. This relates to noise robustness.

## Inference problems:
There is a clear distinction between the choice of model, choice of inference, and the resulting algorithm. A variety of models can be used and a variety of learning principles and inference algorithms (e.g. ML, MAP, EM, MCMC, Variational, etc) are available for performing inference in these models. Combining models with different inference schemes leads to different algorithms e.g. GAN, VAE, Regularization, Optimization methods like SGD, etc. 


- Evidence estimation
1- marginal likelihood of evidence: Write the log density as the marginalization of the joint. We introduce a variational approximate q, into our marginal integral of joint p(x,z), to get p(x). By taking the log from both sides, and using Jensen's inequality we get the ELBO. Maximizing the ELBO is equivalent to minimizing the KL divergence of the real and variational posterior. 


- Density ratio estimation (Density estimation by comparison)
The main idea is to estimate a ratio of real data distribution and model data distribution p(x)/q(x) instead of computing two densities that are hard. The ELBO in variational inference can be written in terms of the ratio. Introducing the variational posterior into the marginal integral of the joint results in the ELBO being $E[log p(x,z)- log q(z/x)]$. By subtracting emprical distribution on the observations, q(x) which is a constant and doesn't change optimization we have the ELBO using ratio as $E[log p(x,z)/q(x,z)]$. 


1- Probabilistic classification: We can frame the ratio estimation as as the problem of classifying the real data (p(x)) from the data produced from model (q(x)). This is what happens in GANs.

2- moment matching (log density difference): if all the infinite statistical moments of two distributions are the same the distributions are the same. So the idea is to set the moments of the numenator distribution (p(x)) equal to the moments of a transformed version of the denumerator (r(x)q(x)). This makes it possible to calculate the ratio r(x).

3- Ratio matching: basic idea is to directly match a density ratio model r(x) to the true density ratio under some divergence. A kernel is usually used for this density estimation problem plus a distance measure (e.g. KL divergence) to measure how close the estimation of r(x) is to the true estimation. So it's variational in some sense. Loosely speaking, this is what happens in variational Autoencoders!

4- Divergence minimization: Another approach to two sample testing and density ratio estimation is to use the divergence (f-divergence, Bergman divergence) between the true density p and the model q, and use this as an objective to drive learning of the generative model. f-GANs use the KL divergence as a special case and are equipped with an exploitable variational formulation (i.e. the variational lower bound). There is no discriminator in this formulation, and this role is taken by the ratio function. We minimise the ratio loss, since we wish to minimise the negative of the variational lower bound; we minimise the generative loss since we wish to drive the ratio to one.

5- Maximum mean discrepancy(MMD): is a nonparametric way to measure dissimilarity between two probability distributions. Just like any metric of dissimilarity between distributions, MMD can be used as an objective function for generative modelling.  The MMD criterion also uses the concept of an 'adversarial' function f that discriminates between samples from Q and P. However, instead of it being a binary classifier constrained to predict 0 or 1, here f can be any function chosen from some function class. The idea is: if P and Q are exactly the same, there should be no function whose expectations differ under Q and P. In GAN, the maximisation over f is carried out via stochastic gradient descent, here it can be done analytically. One could design a kernel which has a deep neural network in it, and use the MMD objective!?

6- Instead of estimating ratios, we estimate gradients of log densities. For this, we can use[ denoising as a surrogate task](http://www.inference.vc/variational-inference-using-implicit-models-part-iv-denoisers-instead-of-discriminators/).denoisers estimate gradients directly, and therefore we might get better estimates than first estimating likelihood ratios and then taking the derivative of those


- Moment computation

$E[f(z)|x] =\int f(z)p(z|x)dz$

- Prediction

$p(xt+1) =\int p(xt+1|xt)p(xt)dxt$

- Hypothesis Testing

$B = log p(x|H1) - log p(x|H2)$


## Bayesian deep learning

## Variational inference:
For most interesting models, the denominator of posterior is not tractable. We appeal to approximate posterior inference.

Traditional solution to the inference problem involves Gibbs sampling which is a variant of MCMC based on sampling from the posterior. Gibbs sampling is based on randomness and sampling, has strict conjugacy requirements, require many iterations and sampling, and it's not very easy to gauge whether the Markov Chain has converged to start collecting samples from the posterior. Variational Inference solves these problems by turning the inference into optimization. We posit a variational family of distributions over the latent variables,  Fit the variational parameters to be close (in KL sense) to the exact posterior. KL is intractable (only possible exactly if q is simple enough and compatible with the prior), so VI optimizes the evidence lower bound (ELBO) instead which is a lower bound on log p(x). Maximizing the ELBO is equivalent to minimizing the KL, note that the ELBO is not convex. The ELBO trades off two terms, the first term prefers q(.) to place its mass on the MAP estimate. The second term encourages q(.) to be diffuse.

To optimize the ELBO, Traditional VI uses coordinate ascent which iteratively update each parameter, holding others fixed. Classical VI is inefficient since they do some local computation for each data point. We will therefore perform amortised inference where we introduce an inference network for all observations instead.

Stochastic variational inference (SVI) scales VI to massive data. Additionally, SVI enables VI on a wide class of difficult models and enable VI with elaborate and flexible families of approximations. Stochastic Optimization replaces the gradient with cheaper noisy estimates and is guaranteed to converge to a local optimum. Example is SGD where the gradient is replaced with the gradient of a stochastic sample batch. The variational inferene recipe is:
- Start with a model
- Choose a variational approximation (variational family)
- Write down the ELBO and compute the expectation.  
- Take ELBO derivative (use stohcastic estimate instead)
- Optimize using the GD/SGD update rule


## Inference vs Action
- We can build a generative model by pairing a model, inference scheme and build an algorithms for density estimation of the model. There are therefore, two distinct things we might be interested in, first is building a model. Second is what to do with this model. If we want to make decisions and act upon the model then we face the problem of reinforcement learning. We build a model first and then put it in an environment to take actions and get rewards in order to evolve. 


# Other Generative modeling ideas (inspired by GANs)

[Unsupervised learning of visual representations by solving jigsaw puzzles](http://arxiv.org/abs/1603.09246) is a clever trick. The author break the image into a puzzle and train a deep neural network to solve the puzzle. The resulting network has one of the highest performance of pre-trained networks.

[Unsupervised learning of visual representations from image patches and locality](https://arxiv.org/abs/1511.06811) is also a clever trick. Here they take two patches of the same image closely located. These patches are statistically of the same object. A third patch is taken from a random picture and location, statistically not of the same object as the other 2 patches. Then a deep neural network is trained to discriminate between 2 patches of same object or different objects. The resulting network has one of the highest performance of pre-trained networks.

[Unsupervised learning of visual representations from stereo image reconstructions](http://arxiv.org/abs/1604.03650) takes a stereo image, say the left frame, and reconstruct the right frame. Albeit this work was not aimed at unsupervised learning, it can be! This method also generates interesting [3D movies form stills](https://github.com/piiswrong/deep3d).

[Unsupervised Learning of Visual Representations using surrogate categories](http://arxiv.org/abs/1406.6909) uses patches of images to create a very large number of surrogate classes. These image patches are then augmented, and then used to train a supervised network based on the augmented surrogate classes. This gives one of the best results in unsupervised feature learning.

[Unsupervised Learning of Visual Representations using Videos](http://arxiv.org/abs/1505.00687) uses an encoder-decoder LSTM pair. The encoder LSTM runs through a sequence of video frames to generate an internal representation. This representation is then decoded through another LSTM to produce a target sequence. To make this unsupervised, one way is to predict the same sequence as the input. Another way is to predict future frames.

Another paper (MIT: Vondrick, Torralba) using videos with very compelling results is [here](http://arxiv.org/abs/1504.08023). This work is from April 2015! The great idea behind this work is to predict the representation of future frames from a video input. This is an elegant approach.

[PredNet is a network designed to predict future frames in video](https://coxlab.github.io/prednet/)
PredNet is a very clever neural network model that in our opinion will have a major role in the future of neural networks. PredNet learns a neural representation that extends beyond the single frames of supervised CNN.It uses [predictive coding](http://www.nature.com/neuro/journal/v2/n1/full/nn0199_79.html) and using [feedback connections in neural models](http://arxiv.org/abs/1608.03425)

# Future

Unsupervised training is very much an open topic, where you can make a large contribution by:

- creating a new unsupervised task to train networks, e.g.: solve a puzzle, compare image patches, generate images, …)

- thinking of tasks that create great unsupervised features, e.g.: what is object and what is background, same on stereo images, same on video frames ~= similar to how our human visual system develops


## self-organizing maps (SOM)
- Self organizing maps (SOM) unsupervisedly map a vector from some input space (usually of high dimensionality) onto a vector in some output space (usually 2d) while maintaining topographic relationships (vectors that are similar in the input space correspond to vectors that are similar in the output space). Do transformation models like GANs perform the opposite!? [video](https://www.youtube.com/watch?v=b3nG4c2NECI)


- SOM is a space-filling grid (for example a 2D lattice of neurons) that provides a dimensionality reduction of the data. For intuition think of a bunch of people in a stadium, every one compares themselves to their neighbors (using a distance metric like euclidean) and moves closer to the ones that are more similar to them until convergence. Looking from above, we see a self organizing map. SOM is biologically inspired by the way that sensory information is processed in neocortex by highly ordered neural nets. Representations of the sensory modalities are organized into well ordered maps in the brain tangent to the cortical surfaces.


- You start with a high-dimensional space of data points, and an arbitrary grid that sits in that space. The grid can be of any dimension, but is usually smaller than the dimension of your dataset, and is commonly 2D, because that's easy to visualise. For each datum in your data set, you find the nearest grid point, and "pull" that grid point toward the data set. You also pull each of the neighbouring grid points toward the new position of the first grid point. At the start of the process, you pull lots of the neighbours toward the data point. Later in the process, when your grid is starting to fill the space, you move less neighbours, and this acts as a kind of fine tuning. This process results in a set of points in the data space that fit the shape of the space reasonably well, but can also be treated as a lower-dimension grid.


- SOM learning algorithm is simple. 1- initialize all weights randomly 2- feed a data point to the net and find its distance to weight vectors of all nodes, the minimum distance is best matching unit (BMU) 3- Find the set of points on the lattice within a given radius of from BMU (neighborhood radius gets smaller with time). 4- Adjust the weight vectors of nodes in neibourhood to be more like the data point (add a portion of the distance to datapoint to the weights based on their proximity to BMU) 5- Repeat unit convergence! 6- the density of nodes on the lattice (number of data points connected to each node) as a heat map is called SOM and we can run clustering algos on them.