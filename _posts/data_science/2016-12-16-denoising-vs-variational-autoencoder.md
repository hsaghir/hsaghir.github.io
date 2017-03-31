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

I start with a short history. In the 90s, a few researchers suggested a [probabilistic interpretation](https://en.wikipedia.org/wiki/Probabilistic_neural_network) of neural network models that was very promising since they offered a proper [bayesian approach](https://en.wikipedia.org/wiki/Bayesian_inference), robustness to [overfitting](https://en.wikipedia.org/wiki/Overfitting), uncertainty estimates, and could easily learn from small datasets. These are great properties that all machine learning practitioners strive for, so people were excited! However, learning parameters for such models proved to be very challenging, until recently. New advancements in deep learning research has led to more efficient parameter learning methods for such probabilistic methods. Therefore the excitement is back and the bayesian approaches to probabilistic reasoning have gained popularity again.

The probabilistic interpretation relaxes the rigid constraint of a single value for each parameter in the network by assuming a probability distributions for each parameter. So for example, if in classical neural networks we calculated a weight as $$w_i=0.7$$, in the probabilistic version we calculate a Gaussian distribution around mean $$u_i=0.7$$ and some variance $$v_i=0.1$$, i.e. $$w_i=N(0.7, 0.1)$$. This is typically done for all the weights but not biases of the network. This assumption will convert the inputs, hidden representations, and the outputs of a neural network to [probabilistic random variables](https://en.wikipedia.org/wiki/Random_variable) within a directed [graphical model](https://en.wikipedia.org/wiki/Graphical_model). Such a network is called a bayesian neural network or BNN.

![alt text](/images/VAE_intuitions/weight_2_dist.jpg "parameters to distributions")


The goal of learning would now be to find the parameters of the mentioned distributions instead of single-value weights. This learning is now called ["inference"](https://en.wikipedia.org/wiki/Bayesian_inference) in probabilistic terms since we want to infer distributions for weights from our data distribution. Inference in a Bayes net corresponds to calculating the conditional probability of latent variables with respect to the data, or put simply, finding the mean and variance for Gaussian distributions over parameters. 

It has been shown that exact inference in Bayes nets is [NP-hard](https://en.wikipedia.org/wiki/NP-hardness) [3]. So such models were not used much with big and moderate size datasets until recently, where a variational approximate inference approach was introduced that transformed the problem into an optimization problem, which could in turn be solved using [stochastic gradient descent](https://hsaghir.github.io/a-primer-on-neural-networks/). “variational” is an umbrella term for optimization-based formulation of problems. It has historical roots in the calculus of variations and thus the name. A variational representation of a problem is its re-formulation in the form of a cost function that is ideally [convex](https://en.wikipedia.org/wiki/Convex_function) and can be optimized. For example, solving a linear matrix equation involves a matrix inversion which is hard, so we can solve this problem in the variational sense by reformulating it as an optimization problem that can be solved computationally using a method like gradient descent.

Let's get back to the bayesian net, since parameters now have distributions, the network can be re-parameterized based on the parameters of the distributions instead of single weight values. In a variational autoencoder, these distributions are only assumed on the hidden code not all parameters of the network. So the encoder becomes a variational inference network that maps the data to the distributions for the hidden code, and the decoder becomes a generative network that maps the hidden code back to distribution of the data. 

![alt text](/images/VAE_intuitions/VAE_inf_gen.jpg "A Variational Autoencoder")

We need to sample the hidden code from its distribution to be able to generate data (hidden code is a distribution not a single value anymore). Therefore to make it differentiable, we treat the mean and variances of the distributions as traditional network parameters and multiply the variance by a sample from a normal noise generator to add randomness. By parameterizing the hidden distribution this way, we can backpropagate the gradient to the parameters of the encoder and train the whole network with stochastic gradient descent. This procedure will allow us to learn mean and variance values for the hidden code and it's called the ["re-parameterization trick"](http://stats.stackexchange.com/questions/199605/how-does-the-reparameterization-trick-for-vaes-work-and-why-is-it-important). It is important to appreciate the importance of the fact that the whole network is now differentiable. This means that optimization techniques can now be used to solve the inference problem efficiently. 

In classic version of neural networks we could simply measure the error of network outputs with desired target value using a simple [mean square error](https://en.wikipedia.org/wiki/Mean_squared_error). But now that we are dealing with distributions, MSE is no longer a good error metric. So instead, loosely speaking, we use another metric for measuring the difference between two distributions i.e. [KL-Divergence](https://www.quora.com/What-is-a-good-laymans-explanation-for-the-Kullback-Leibler-Divergence). It turns out that this distance between our variational approximate and the real posterior distribution is not very easy to minimize either [2]. However, using some simple math, we can show that this distance is always positive and that it comprises of two main parts (probability of data minus a function called ELBO). So instead of minimizing the whole thing, we can maximize the smaller term (ELBO) [4]. This term comes from marginalizing the log-probability of data and is always smaller than the log probability of the data. That's why we call it a lower bound on the evidence (data given model). From the perspective of autoencoders, the ELBO function can be seen as the sum of the reconstruction cost of the input plus the regularization terms. 

If after maximizing the ELBO, the lower bound of data is close to the data distribution, then the distance is close to zero and voila! we have minimized the error distance indirectly. The algorithm we use to maximize the lower bound is the exact opposite of gradient descent. Instead of going in the reverse direction of the gradient to get to the minimum, we go toward the positive direction to get to the maximum, so it's now called gradient ascent! This whole algorithm is called "autoencoding variational bayes" [1]! After we are done with the learning we can visualize the latent space of our VAE and generate samples from it. Pretty cool, eh!?

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

Now that the intuition is clear, [here is a jupyter notebook](https://github.com/hsaghir/VAE_intuitions/blob/master/VAE_MNIST_keras.ipynb) for playing with VAEs, if you like to learn more. The notebook is based on [this](https://github.com/fchollet/keras/blob/master/examples/variational_autoencoder.py) keras example. The resulting learned latent space of the encoder and the manifold of a simple VAE trained on the MNIST dataset are below. 


<img src="/images/VAE_intuitions/latent_space.jpg" alt="VAE Latent space" width="250" height="250"> | <img src="/images/VAE_intuitions/manifold.jpg" alt="VAE generated samples" width="250" height="250">


##### Side note 1: 
The ELBO is determined from introducing a variational distribution q, on lower bound on the marginal log likelihood, i.e. $$\log \ p(x)=\log \int_z p(x,z) * \frac{q(z|x)}{q(z|x)}$$. We use the log likelihood to be able to use the concavity of the $$\log$$ function and employ Jensen's equation to move the $$\log$$ inside the integral i.e. $$\log \ p(x) > \int_z \log\ (p(x,z) * \frac{q(z|x)}{q(z|x)})$$ and then use the definition of expectation on $$q$$ (the nominator $$q$$ goes into the definition of the expectation on $$q$$ to write that as the ELBO) $$\log \ p(x) > ELBO(z) = E_q [- \log\ q(z|x) + \log \ p(x,z)]$$. The difference between the ELBO and the marginal $$p(x)$$ which converts the inequality to an equality is the distane between the real posterior and the approximate posterior i.e. $$KL[q(z|x)\ | \ p(z|x)]$$. Or alternatively, the distance between the ELBO and the KL term is the log normalizer $$p(x)$$. Replace the $$p(z|x)$$ with Bayesian formula to see how. 

Now that we have a defined a loss function, we need the gradient of the loss function, $$\delta E_q[-\log q(z \vert x)+p(x,z)]$$ to be able to use it for optimization with SGD. The gradient isn't easy to derive analytically but we can estimate it by using MCMC to directly sample from $$q(z \vert x)$$ and estimate the gradient. This approach generally exihibits large variance since MCMC might sample from rare values.

This is where the reparameterization trick we discussed above comes in. We assume that the random variable $$z$$ is a deterministic function of $$x$$ and a known $$\epsilon$$ ($$\epsilon$$ are iid samples) that injects randomness $$z=g(x,\epsilon)$$. This reparameterization converts the undifferentiable random variable $$z$$, to a differentiable function of $$x$$ and a decoupled source of randomness. Therefore, using this reparameterization, we can estimate the gradient of the ELBO as $$\delta E_\epsilon [\delta -\log\ q(g(x,\epsilon) \vert x) + \delta p(x,g(x,\epsilon))]$$. This estimate to the gradient has empirically shown to have much less variance and is called "Stochastic Gradient Variational Bayes (SGVB)". The SGVB is also called a black-box inference method (similar to MCMC estimate of the gradient) which simply means it doesn't care what functions we use in the generative and inference network. It only needs to be able the calculate the gradient at samples of $$\epsilon$$. We can use SGVB with a seperate set of parameters for each observation however that's costly and inefficient. We usually choose to "amortize" the cost of inference with deep networks (to learn a single complex funtion for all observations instead). All the terms of the ELBO are differentiable now if we choose deep networks as our likelihood and approximate posterior functions. Therefore, we have an end-to-end differentiable model. Following depiction shows amortized SGVB reparameterization in a VAE.

![alt text](/images/VAE_intuitions/vae_structure.jpg "Simple VAE structure with reparameterization")

##### side note 2:
Note that in the above derivation of the ELBO, the first term is the entropy of the variational posterior and second term is log of joint distribution. However we usually write joint distribution as $$p(x,z)=p(x|z)p(z)$$ to rewrite the ELBO as $$ E_q[\log\ p(x|z)+KL(q(z|x)\ | \ p(z))]$$. This derivation is much closer to the typical machine learning literature in deep networks. The first term is log likelihood (i.e. reconstruction cost) while the second term is KL divergence between the prior and the posterior (i.e a regularization term that won't allow posterior to deviate much from the prior). Also note that if we only use the first term as our cost function, the learning with correspond to maximum likelihood learning that does not include regularization and might overfit.


##### Side note 3: problems that we might encounter while working with VAE; 
It's useful to think about the distance measure we talked about. KL-divergence meausres a sort of distance between two distributions but it's not a true distance since it's not symmetric  $$KL(P|Q) = E_P[\log\ P(x) − \log\ Q(x)]$$. So which distance direction we choose to minimize has consequences. For example, in minimizing $$KL(p|q)$$, we select a $$q$$ that has high probability where $$p$$ has high probability so when $$p$$ has multiple modes, $$q$$ chooses to blur the modes together, in order to put high probability mass on all of them. 

On the other hand, in minimizing $$KL(q \vert p)$$, we select a $$q$$ that has low probability where $$p$$ has low probability. When $$p$$ has multiple modes that are sufficiently widely separated, the KL divergence is minimized by choosing a single mode (mode collapsing), in order to avoid putting probability mass in the low-probability areas between modes of $$p$$. In VAEs we actually minimize $$KL(q \vert p)$$ so mode collapsing is a common problem. Additionally, due to the complexity of true distributions we also see blurring problem. These two cases are illustrated in the following figure from the [deep learning book](http://www.deeplearningbook.org/).

![alt text](/images/VAE_intuitions/KL_direction.png "KL Divergence directions")




references:

[1] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).

[2] Rezende, Danilo Jimenez, Shakir Mohamed, and Daan Wierstra. "Stochastic backpropagation and approximate inference in deep generative models." arXiv preprint arXiv:1401.4082 (2014).

[3] https://hips.seas.harvard.edu/blog/2013/01/24/complexity-of-inference-in-bayes-nets/

[4] https://www.cs.princeton.edu/courses/archive/fall11/cos597C/lectures/variational-inference-i.pdf