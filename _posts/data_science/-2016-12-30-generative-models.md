---
layout: article
title: An intuitive primer on deep generetive models
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---
Deep generative models provide a solution to the problem of unsupervised learning, in which a machine learning system is required to discover the structure hidden within unlabelled data streams. They can be used to explore variations in data, to reason about the structure and behaviour of the world, and ultimately, for decision-making. Deep generative models have widespread applications including those in density estimation, image denoising and in-painting, data compression, scene understanding, representation learning, 3D scene construction, semisupervised classification, and hierarchical control, amongst many others.

The objective function we use for training a probabilistic model should match the way we ultimately want to use the model.  generative models: models we actually want to use to generate samples from which are then shown to a human user/observer. This includes use-cases such as image captioning, texture generation, machine translation, speech synthesis and dialogue systems, but excludes things like unsupervised pre-training for supervised learning, semisupervised learning, data compression, denoising and many others. Very often people don't make this distinction clear when talking about generative models which is one of the reasons why there is still no clarity about what different objective functions do.

In "How (not) to Train your Generative Model: Scheduled Sampling, Likelihood, Adversary?", Ferenc argues that when the goal is to train a model that can generate natural-looking samples, maximum likelihood is not a desirable training objective. Maximum likelihood is consistent so it can learn any distribution if it is given infinite data and a perfect model class. However, under model misspecification and finite data (that is, in pretty much every practically interesting scenario), it has a tendency to produce models that overgeneralise.

Generative modelling is about finding a probabilistic model Q that in some sense approximates the natural distribution of data P.So if we want to train models that produce nice samples, my recommendation is to try to use KL[Q∥P] as an objective function or something that behaves like it since perceived quality of each sample is related to the $ −log P(x)$ under the human observers' subjective prior of stimuli. On the other hand, Maximum likelihood is roughly the same as minimising KL[P∥Q]. Both divergences ensure consistency, However, they differ fundamentally in the way they deal with finite data and model misspecification. KL[P∥Q] tends to favour approximations Q that overgeneralise P. Practically this means that the model will occasionally sample unplausible samples that don't look anything like samples from P. KL[Q‖P]  tends to favour under-generalisation. Practically this means that KL[Q‖P] will try to avoid introducing unplausible samples, sometimes at the cost of missing the majority of plausible samples under P. In other words: KL[P‖Q] is liberal, KL[Q‖P] is conservative. In yet other words: KL[P‖Q] is an optimist, KL[Q‖P] is a pessimist.

In unsupervised learning we are given indepenent data samples from some underlying data distribution P which we don't know, and our goal is to come up with an approximate distribution Q that is as close to P as possible, only using the dataset. Often, Q is chosen from a parametric family of distributions, and our goal is to find the optimal parameters so that the distribution P is best approximated.

The central issue of unsupervised learning is choosing an appropriate objective function, that appropriately measures the quality of our approximation, and which is tractable to compute and optimise when we are working with complicated, deep models. Typically, a probability model is specified as q(x;θ)=f(x;θ)/Zθ, where f(x;θ) is some positive function parametrised by θ. The denominator Zθ is called the normalisation constant which makes sure that q is a valid probability model: it has to sum up to one over all possible configurations of x. The central problem in unsupervised learning is that for the most interesting models in high dimensional spaces, calculating ZθZθ is intractable. So The most straightforward choice of objective functions, i.e. marginal likelihood (log likelihood/evidence), is intractable in most cases.Therefore people come up with:

1-Alternative objective functions and fancy methods to deal with intractable models such as:
- density ratio estimation (like  adversarial training, maximum mean discrepancy, discussed later), 
- pseudolikelihood: The twist is this: instead of thinking about fitting a probabilistic model p(x;θ) to data x, you learn a joint probability distribution p(x,z;θ) of the data x and it's noise-corrupted version z. The noise corruption is artificially introduced by us, following a corruption distribution. The point is, if you learn the joint model p(x,z;θ), that also implies a generative model p(x;θ)=∫p(x,z;θ). To fit the joint model to observations (xi,zi), we use score matching with the pseudolikelihood scoring rule as objective function.

- MSE or maximum spacing estimation: if you squash a random variable X through its cumulative distribution function F, the transformed variable F(X) follows a standard uniform distribution between 0 and 1. Furthermore, we can say that for any random variable X, its CDF (F) is the only monotonic, left-continuous function with this property. Using this observation one can come up with objective functions for univariate density estimation. Let's assume we have a parametric family of distributions described by their CDFs Fθ, and we want to find the θ that best fits our observations Xi. If Fθ is close to the true CDF of the data distribution, we would expect Fθ(xi) to follow a uniform distribution. Therefore, we only need a way to measure the uniformity of Fθ(xi) across samples. Maximum Spacing Estimation (MSE) uses the geometric mean of spacing between ordered samples as an objective function. 


- Alternative optimisation methods or approximate inference methods such as contrastive divergence or variational Bayes.

2- Interesting models that have tractable normalisation constants:
-  MADE: Masked Autoencoder for Distribution Estimation: This paper sidesteps the high dimensional normalisation problem by restricting the class of probability distributions to autoregressive models! In a model like this, we only need to compute the normalisation of each q(xd|x1:d−1;θ) term, and we can be sure that the resulting model is a valid model over the whole vector x. But as normalising these one-dimensional probability distributions is a lot easier, we have a whole range of interesting tractable distributions at our disposal.


The main ideas that repeatedly come up as one explores deep generative models are  Bayesian deep learning, variational approximations, memoryless and amortised inference, and stochastic gradient estimation. So the hope is that by having an intuition of these ideas before the mathematics, one can understand deep generative models. 

## treat deep networks as functions, Convolutional, recurrent, LSTM, GRU, fully-connected, etc

## design of probabilistic models
Three class of probabilistic generetive models exist. 

1. Fully-observed models: These Models observe data directly. Char-RNNs, PixelRNN, RBMs and autoregressive models are of this type. These models don't introduce any new unobserved local variables for the observations but do have a deterministic internal hidden representation of the data which acts like a memory of previous data in a sequence based on the degree of conditionals assumed. For example, in the case of char-RNN, the number of unrolling the RNN is the number of conditional probabilities. If we put a softmax layer on the output, the decision of the RNN will be a probability distribution on possible outputs and the rest of the RNN can be deterministic. We sample from the output softmax distribution based on deterministic hidden variable and input sequence. They can directly model relationships between data points in a sequence due to their memory. But, they are limited by the dimension of their hidden representations or put another way, limited by the assumed conditional dependencies between a point and the number of its previous points. 

-  MADE: Masked Autoencoder for Distribution Estimation: Autoregressive models are used a lot in time series modelling and language modelling: hidden Markov models or recurrent neural networks are examples. There, autoregressive models are a very natural way to model data because the data comes ordered (in time).

What's weird about using autoregressive models in this context is that it is sensitive to ordering of dimensions, even though that ordering might not mean anything. If xx encodes an image, you can think about multiple orders in which pixel values can be serialised: sweeping left-to-right, top-to-bottom, inside-out etc. For images, neither of these orderings is particularly natural, yet all of these different ordering specifies a different model above.

But it turns out, you don't have to choose one ordering, you can choose all of them at the same time. The neat trick in the masking autoencoder paper is to train multiple autoregressive models all at the same time, all of them sharing (a subset of) parameters θθ, but defined over different ordering of coordinates. This can be achieved by thinking of deep autoregressive models as a special cases of an autoencoder, only with a few edges missing.

2. Latent variable models (Probabilistic graphical models): These models introduce an unobserved random variable for every observed data point. Gaussian mixture models (GMM) are of this type. The latent variables can be used to explain hidden causes and factors of variation in the data. Therefore, such models incorporate our assumptions about the hidden causes of the data in the form of their graphical models. This is in contrast with fully-observed models that do not impose such explicit assumptions on the data. They are easy to sample from, include hierarchy of causes believed.

3. Transformation models (Implicit generative models): These models Model data as a transformation of an unobserved noise source using a deterministic function. Their main properties are that we can sample from very easily, and that we can take the derivative of samples with respect to parameters. They are used as generative models  to model distributions of observed data but can be used for approximate inference as well where one uses them to model distributions over latent variables.

Real-valued non-volume preserving transformation (Real NVP) is an invertable transformation model. The generative procedure from the model is very similar to the one used in Variational Auto-Encoders and Generative Adversarial Networks: sample a vector z from a simple distribution (here a Gaussian) and pass it through the generator network g to obtain a sample x=g(z). From the generated samples, it seems the model was able to capture the statistics from the original data distribution. For example, the samples are in general relatively sharp and coherent and therefore suggest that the models understands something more than mere correlation between neighboring pixels. This is due to not relying on fixed form reconstruction cost like squared loss on the data level. The models seems also to understand to some degree the notion of foreground/background, and volume, lighting and shadows. The generator network g has been built in the paper according to a convolutional architecture, making it relatively easy to reuse the model to generate bigger images. As the model is convolutional, the model is trying to generate a “texture” of the dataset rather than an upsampled version of the images it was trained on. This explains why the model is most successful when trained on background datasets like LSUN different subcategories. This sort of behaviour can also be observed in other models like Deep Convolutional Generative Adversarial Networks.

Another example, Deep Unsupervised Learning using Nonequilibrium Thermodynamics. What we typically try to do in representation learning is to map data to a latent representation. While the Data can have arbitrarily complex distribution along some complicated nonlinear manifold, we want the computed latent representations to have a nice distribution, like a multivariate Gaussian. This paper takes this idea very explicitly using a stochastic mapping to turn data into a representation: a random diffusion process. If you take any data, and apply Brownian motion-like stochastic process to this, you will end up with a standard Gaussian distributed variable due to the stationarity of the Brownian motion. Now the trick the authors used is to train a dynamical system (a Markov chain) to inverts this random walk, to be able to reconstruct the original data distribution from the random Gaussian noise. Amazingly, this works, and the traninig objective becomes very similar to variational autoencoders. 

GANs are also of this type of (transformation) generative models where a random Gaussian noise is transformed into data (e.g.an image) using a deep neural network. These models also assume a noise model on the latent cause. The good thing about such models is that it's easy to sample and compute expectation from these models without knowing the final distribution. Since classifiers are well-develped, we can use our knowledge there for density ratio estimation in these models. However, these models lack noise model and likelyhood. It's also difficult to optimize them.  Any implicit model can be easily turned into a prescribed model by adding a simple likelihood function (noise model) on the generated outputs but models with likelihood functions also regularly face the problem of intractable marginal likelihoods.But the specification of a likelihood function provides knowledge of data marginal p(x) that leads to different algorithms by exploiting this knowledge, e.g., NCE resulting from class-probability based testing in un-normalised models, or variational lower bounds for directed graphical models.

There is a clear distinction between the choice of model, choice of inference, and the resulting algorithm. A variety of models can be used and a variety of learning principles and inference algorithms (e.g. ML, MAP, EM, MCMC, Variational, etc) are available for performing inference in these models. Combining models with different inference schemes leads to different algorithms e.g. GAN, VAE, Regularization, Optimization methods like SGD, etc. 

## Inference problems:
- Evidence estimation


- Density ratio estimation 
The main idea is to estimate a ratio of real data distribution and model data distribution p(x)/q(x) instead of computing two densities that are hard. 

1- Probabilistic classification: We can frame it as the problem of classifying the real data (p(x)) from the data produced from model (q(x)). 

2- moment matching: if all the infinite statistical moments of two distributions are the same the distributions are the same. So the idea is to set the moments of the numenator distribution (p(x)) equal to the moments of a transformed version of the denumerator (r(x)q(x)). This makes it possible to calculate the ratio r(x).

3- Ratio matching: basic idea is to directly match a density ratio model r(x) to the true density ratio under some divergence. A kernel is usually used for this density estimation problem plus a distance measure (e.g. KL divergence) to measure how close the estimation of r(x) is to the true estimation. So it's variational in some sense. Loosely speaking, this is what happens in variational Autoencoders!

4- Divergence minimization: Another approach to two sample testing and density ratio estimation is to use the divergence between the true density p and the model q, and use this as an objective to drive learning of the generative model. f-GANs use the KL divergence as a special case and are equipped with an exploitable variational formulation (i.e. the variational lower bound). There is no discriminator in this formulation, and this role is taken by the ratio function. We minimise the ratio loss, since we wish to minimise the negative of the variational lower bound; we minimise the generative loss since we wish to drive the ratio to one.

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