---
layout: article
title: An intuitive understanding of variational autoencoders without any formula 
comments: true
categories: data_science
image:
  teaser: VAE_intuitions/manifold.jpg
---

I love the simplicity of [autoencoders](https://en.wikipedia.org/wiki/Autoencoder) as a very intuitive [unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning) method. They are in the simplest case, a three layer [neural network](https://hsaghir.github.io/a-primer-on-neural-networks/). In the first layer the data comes in, the second layer typically has smaller number of nodes than the input and the third layer is similar to the input layer. These layers are usually fully connected with each other. Such networks are called auto-encoders since they first "encode" the input to a hidden code and then "decode" it back from the hidden representation. They can be trained by simply measuring the reconstruction error and [back-propagating](https://hsaghir.github.io/a-primer-on-neural-networks/) it to the network's parameters. If you add noise to the input before sending it through the network, they can learn better and in that case, they are called [denoising autoencoders](https://en.wikipedia.org/wiki/Autoencoder). They are useful because they help with understanding the data by trying to extract regularities in them and can compress them into a lower dimensional code.

![alt text](/images/VAE_intuitions/Autoencoder_structure.png "A simple autoencoder")

A typical autoencoder can usually encode and decode data very well with low reconstruction error, but a random latent code seems to have little to do with the training data. In other words the latent code does not learn the probability distribution of the data and therefore, if we are interested in generating more data like our dataset a typical autoencoder doesn't seem to work. There is however another version of autoencoders, called ["variational autoencoder - VAE"](https://arxiv.org/abs/1312.6114) that are able to solve this problem since they explicitly define a probability distribution on the latent code. The neural network architecture is very similar to a regular autoencoder but the difference is that the hidden code comes from a [probability distribution](https://en.wikipedia.org/wiki/Probability_distribution) that is learned during the training. Since it is not very easy to navigate through the math and equations of VAEs, I want to dedicate this post to explaining the intuition behind them. With the intuition in hand, you can understand the math and code if you want to. 

I start with a short history. Over the past two decades, researchers have been trying to develop a framework for probabilistic modelling that explicitly conveys the assumption being made about the data. The result of this effort has been probabilistic graphical models that combines ideas from graph theory and probability theory. Graphical models are a very flexible language that admit composability for building arbitrary complex models and provides a set of tools to perform reasoning (inference) about differents nodes in the graph. 

Using graphical models language we can explicitly articulate the hidden structure that we believe is generating the data. For example, if we assume that there is a hidden variable $$z_i$$ for each observable data point $$x_i$$ that is the cause for generating that point, we can make a model like this:

![alt text](/images/VAE_intuitions/vae_latent_var_model.png "a simple vae latent variable model")

Reasoning (inference) about hidden variables in probabilistic graphical models has traditionally been very hard and only limited to very restricted graph structures. This had significantly limited our ability to build very interesting models and perfrom reasoning about latent variables. However, in the past few years there have been some breakthroughs that have enabled us to perform approximate reasoning with scalable algorithms like stochastic gradient descent. This has trigerred another wave of interest in graphical models. 

A variational autoencoder is essentially a graphical model in the simplest case akin to the figrue above. We assume a local latent variable, $$z_i$$ for each data point $$x_i$$. However, the inference and data generation in a VAE benefit from the power of deep neural networks and scalable optimization i.e. SGD. In a VAE, the encoder becomes a variational inference network that maps the data to the a distribution for the hidden variable, and the decoder becomes a generative network that maps the latent variable back to the data. 

![alt text](/images/VAE_intuitions/VAE_inf_gen.jpg "A Variational Autoencoder")

We need to sample the hidden code from its distribution to be able to generate data (hidden code is a distribution not a single value anymore). Therefore, to make it differentiable, we treat the mean and variances of the distributions as traditional network parameters and multiply the variance by a sample from a normal noise generator to add randomness. By parameterizing the hidden distribution this way, we can backpropagate the gradient to the parameters of the encoder and train the whole network with stochastic gradient descent. This procedure will allow us to learn mean and variance values for the hidden code and it's called the ["re-parameterization trick"](http://stats.stackexchange.com/questions/199605/how-does-the-reparameterization-trick-for-vaes-work-and-why-is-it-important). It is important to appreciate the importance of the fact that the whole network is now differentiable. This means that optimization techniques can now be used to solve the inference problem efficiently. 

In classic version of neural networks we could simply measure the error of network outputs with desired target value using a simple [mean square error](https://en.wikipedia.org/wiki/Mean_squared_error). But now that we are dealing with distributions, MSE is no longer a good error metric. So instead, loosely speaking, we use another metric for measuring the difference between two distributions i.e. [KL-Divergence](https://www.quora.com/What-is-a-good-laymans-explanation-for-the-Kullback-Leibler-Divergence). It turns out that this distance between our variational approximate and the real posterior distribution is not very easy to minimize either [2]. However, using some simple math, we can show that this distance is always positive and that it comprises of two main parts (probability of data minus a function called ELBO). So instead of minimizing the whole thing, we can maximize the smaller term (ELBO) [4]. This term comes from marginalizing the log-probability of data and is always smaller than the log probability of the data. That's why we call it a lower bound on the evidence (data given model). From the perspective of autoencoders, the ELBO function can be seen as the sum of the reconstruction cost of the input plus the regularization terms. 

If after maximizing the ELBO, the lower bound of data is close to the data distribution, then the distance is close to zero and voila!; We have minimized the error distance indirectly. The algorithm we use to maximize the lower bound is the exact opposite of gradient descent. Instead of going in the reverse direction of the gradient to get to the minimum, we go toward the positive direction to get to the maximum, so it's now called gradient ascent! This whole algorithm is called "autoencoding variational Bayes" [1]! After we are done with the learning we can visualize the latent space of our VAE and generate samples from it. Pretty cool, eh!?

Here is a high level pseudo-code for the architecture of a VAE to put things into perspective:

```

network= {

  # encoder
  encoder_x = Input_layer(size=input_size, input=data)
  encoder_h = Dense_layer(size=hidden_size, input= encoder_x)

  # the re-parameterized distributions that are inferred from data 
  z_mean = Dense(size=number_of_distributions, input=encoder_h)
  z_variance = Dense(size=number_of_distributions, input=encoder_h)
  epsilon= random(size=number_of_distributions)

  # decoder network needs a sample from the code distribution
  z_sample= z_mean + exp(z_variance / 2) * epsilon

  #decoder
  decoder_h = Dense_layer(size=hidden_size, input=z_sample)
  decoder_output = Dense_layer(size=input_size, input=decoder_h)
}

cost={
  reconstruction_loss = input_size * crossentropy(data, decoder_output)
  kl_loss = - 0.5 * sum(1 + z_variance - square(z_mean) - exp(z_variance))
  cost_total= reconstruction_loss + kl_loss
}

stochastic_gradient_descent(data, network, cost_total)

```

Now that the intuition is clear, [here is a Jupyter notebook](https://github.com/hsaghir/VAE_intuitions/blob/master/VAE_MNIST_keras.ipynb) for playing with VAEs, if you like to learn more. The notebook is based on [this](https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py) Keras example. The resulting learned latent space of the encoder and the manifold of a simple VAE trained on the MNIST dataset are below. 


<img src="/images/VAE_intuitions/latent_space.jpg" alt="VAE Latent space" width="350" height="350"> | <img src="/images/VAE_intuitions/manifold.jpg" alt="VAE generated samples" width="350" height="350">


##### Side note 1: 
The ELBO is determined from introducing a variational distribution q, on lower bound on the marginal log likelihood, i.e. $$\log \ p(x)=\log \int_z p(x,z) * \frac{q(z|x)}{q(z|x)}$$. We use the log likelihood to be able to use the concavity of the $$\log$$ function and employ Jensen's equation to move the $$\log$$ inside the integral i.e. $$\log \ p(x) > \int_z \log\ (p(x,z) * \frac{q(z|x)}{q(z|x)})$$ and then use the definition of expectation on $$q$$ (the nominator $$q$$ goes into the definition of the expectation on $$q$$ to write that as the ELBO) $$\log \ p(x) > ELBO(z) = E_q [- \log\ q(z|x) + \log \ p(x,z)]$$. The difference between the ELBO and the marginal $$p(x)$$ which converts the inequality to an equality is the distance between the real posterior and the approximate posterior i.e. $$KL[q(z|x)\ | \ p(z|x)]$$. Alternatively, the distance between the ELBO and the KL term is the log-normalizer $$p(x)$$. Replace the $$p(z|x)$$ with Bayesian formula to see how. 

Now that we have a defined a loss function, we need the gradient of the loss function, $$\delta E_q[-\log q(z \vert x)+p(x,z)]$$ to be able to use it for optimization with SGD. The gradient isn't easy to derive analytically but we can estimate it by using MCMC to directly sample from $$q(z \vert x)$$ and estimate the gradient. This approach generally exhibits large variance since MCMC might sample from rare values.

This is where the re-parameterization trick we discussed above comes in. We assume that the random variable $$z$$ is a deterministic function of $$x$$ and a known $$\epsilon$$ ($$\epsilon$$ are iid samples) that injects randomness $$z=g(x,\epsilon)$$. This re-parameterization converts the undifferentiable random variable $$z$$, to a differentiable function of $$x$$ and a decoupled source of randomness. Therefore, using this re-parameterization, we can estimate the gradient of the ELBO as $$\delta E_\epsilon [\delta -\log\ q(g(x,\epsilon) \vert x) + \delta p(x,g(x,\epsilon))]$$. This estimate to the gradient has empirically shown to have much less variance and is called "Stochastic Gradient Variational Bayes (SGVB)". The SGVB is also called a black-box inference method (similar to MCMC estimate of the gradient) which simply means it doesn't care what functions we use in the generative and inference network. It only needs to be able the calculate the gradient at samples of $$\epsilon$$. We can use SGVB with a separate set of parameters for each observation however that's costly and inefficient. We usually choose to "amortize" the cost of inference with deep networks (to learn a single complex function for all observations instead). All the terms of the ELBO are differentiable now if we choose deep networks as our likelihood and approximate posterior functions. Therefore, we have an end-to-end differentiable model. Following depiction shows amortized SGVB re-parameterization in a VAE.

<img src="/images/VAE_intuitions/vae_structure.jpg" alt="Simple VAE structure with reparameterization" width="350" height="350">

##### side note 2:
Note that in the above derivation of the ELBO, the first term is the entropy of the variational posterior and second term is log of joint distribution. However we usually write joint distribution as $$p(x,z)=p(x|z)p(z)$$ to rewrite the ELBO as $$ E_q[\log\ p(x|z)+KL(q(z|x)\ | \ p(z))]$$. This derivation is much closer to the typical machine learning literature in deep networks. The first term is log likelihood (i.e. reconstruction cost) while the second term is KL divergence between the prior and the posterior (i.e a regularization term that won't allow posterior to deviate much from the prior). Also note that if we only use the first term as our cost function, the learning with correspond to maximum likelihood learning that does not include regularization and might overfit.


##### Side note 3: problems that we might encounter while working with VAE; 
It's useful to think about the distance measure we talked about. KL-divergence measures a sort of distance between two distributions but it's not a true distance since it's not symmetric  $$KL(P|Q) = E_P[\log\ P(x) − \log\ Q(x)]$$. So which distance direction we choose to minimize has consequences. For example, in minimizing $$KL(p|q)$$, we select a $$q$$ that has high probability where $$p$$ has high probability so when $$p$$ has multiple modes, $$q$$ chooses to blur the modes together, in order to put high probability mass on all of them. 

On the other hand, in minimizing $$KL(q \vert p)$$, we select a $$q$$ that has low probability where $$p$$ has low probability. When $$p$$ has multiple modes that are sufficiently widely separated, the KL divergence is minimized by choosing a single mode (mode collapsing), in order to avoid putting probability mass in the low-probability areas between modes of $$p$$. In VAEs we actually minimize $$KL(q \vert p)$$ so mode collapsing is a common problem. Additionally, due to the complexity of true distributions we also see blurring problem. These two cases are illustrated in the following figure from the [deep learning book](http://www.deeplearningbook.org/).

![alt text](/images/VAE_intuitions/KL_direction.png "KL Divergence directions")




references:

[1] Kingma, Diederik P., and Max Welling. "Auto-encoding variational Bayes." arXiv preprint arXiv:1312.6114 (2013).

[2] Rezende, Danilo Jimenez, Shakir Mohamed, and Daan Wierstra. "Stochastic backpropagation and approximate inference in deep generative models." arXiv preprint arXiv:1401.4082 (2014).

[3] https://hips.seas.harvard.edu/blog/2013/01/24/complexity-of-inference-in-bayes-nets/

[4] https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf