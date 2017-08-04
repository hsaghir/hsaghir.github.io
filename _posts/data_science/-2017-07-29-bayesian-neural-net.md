---
layout: article
title: Bayesian Neural Networks and uncertainty
comments: true
categories: data_science
image:
  teaser: VAE_intuitions/uncertainty.jpg
---

I start with a short history. In the 90s, a few researchers suggested a [probabilistic interpretation](https://en.wikipedia.org/wiki/Probabilistic_neural_network) of neural network models that was very promising since they offered a proper [Bayesian approach](https://en.wikipedia.org/wiki/Bayesian_inference), robustness to [overfitting](https://en.wikipedia.org/wiki/Overfitting), uncertainty estimates, and could easily learn from small datasets. These are great properties that all machine learning practitioners strive for, so people were excited! However, learning parameters for such models proved to be very challenging, until recently. New advancements in deep learning research has led to more efficient parameter learning methods for such probabilistic methods. Therefore, the excitement is back and the Bayesian approaches to probabilistic reasoning have gained popularity again.

The probabilistic interpretation relaxes the rigid constraint of a single value for each parameter in the network by assuming a probability distributions for each parameter. So for example, if in classical neural networks we calculated a weight as $$w_i=0.7$$, in the probabilistic version we calculate a Gaussian distribution around mean $$u_i=0.7$$ and some variance $$v_i=0.1$$, i.e. $$w_i=N(0.7, 0.1)$$. This is typically done for all the weights but not biases of the network. This assumption will convert the inputs, hidden representations, and the outputs of a neural network to [probabilistic random variables](https://en.wikipedia.org/wiki/Random_variable) within a directed [graphical model](https://en.wikipedia.org/wiki/Graphical_model). Such a network is called a Bayesian neural network or BNN.

![alt text](/images/VAE_intuitions/weight_2_dist.jpg "parameters to distributions")


The goal of learning would now be to find the parameters of the mentioned distributions instead of single-value weights. This learning is now called ["inference"](https://en.wikipedia.org/wiki/Bayesian_inference) in probabilistic terms since we want to infer distributions for weights from our data distribution. Inference in a Bayes net corresponds to calculating the conditional probability of latent variables with respect to the data, or put simply, finding the mean and variance for Gaussian distributions over parameters. 

It has been shown that exact inference in Bayes nets is [NP-hard](https://en.wikipedia.org/wiki/NP-hardness) [3]. So such models were not used much with big and moderate size datasets until recently, where a variational approximate inference approach was introduced that transformed the problem into an optimization problem, which could in turn be solved using [stochastic gradient descent](https://hsaghir.github.io/a-primer-on-neural-networks/). “variational” is an umbrella term for optimization-based formulation of problems. It has historical roots in the calculus of variations and thus the name. A variational representation of a problem is its re-formulation in the form of a cost function that is ideally [convex](https://en.wikipedia.org/wiki/Convex_function) and can be optimized. For example, solving a linear matrix equation involves a matrix inversion which is hard, so we can solve this problem in the variational sense by reformulating it as an optimization problem that can be solved computationally using a method like gradient descent.

Let's get back to the Bayesian net, since parameters now have distributions, the network can be re-parameterized based on the parameters of the distributions instead of single weight values. In a variational autoencoder, these distributions are only assumed on the hidden code not all parameters of the network. So the encoder becomes a variational inference network that maps the data to the distributions for the hidden code, and the decoder becomes a generative network that maps the hidden code back to distribution of the data. 


references:

[1] Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, Daan Wierstra. "Weight Uncertainty in Neural Networks" https://arxiv.org/abs/1505.05424

[2] Yarin Gal, Uncertainty in Deep Learning (PhD Thesis),  http://mlg.eng.cam.ac.uk/yarin/blog_2248.html

[3] Yarin Gal, Zoubin Ghahramani. "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning", http://proceedings.mlr.press/v48/gal16.html

[4] Yarin Gal, Jiri Hron, Alex Kendall. "Concrete Dropout" https://arxiv.org/abs/1705.07832. Jupyter notebook: https://github.com/yaringal/ConcreteDropout/blob/master/concrete-dropout.ipynb

[5] Yarin Gal, Riashat Islam, Zoubin Ghahramani. "Deep Bayesian Active Learning with Image Data" https://arxiv.org/abs/1703.02910

[6] https://hips.seas.harvard.edu/blog/2013/01/24/complexity-of-inference-in-bayes-nets/

