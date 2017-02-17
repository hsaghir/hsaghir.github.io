---
layout: article
title: Deep Feedforward nets Deep Learning book Ch6
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

- Think of feedforward nets as function approximation machines designed to achieve statistical generalization. Think of a layer as a set of units doing vector-to-scaler mappings in parallel.

- Starting with a linear model $wx+b$. We can transform the x into a new represenation before passing into the model. Three approaches are possible, first to use a generic kernel method; generalization is poor since it uses smoothness assumption. Second is to do feature engineering and hand-design features which requires very expert knowledge of data. Third is to do feature learning which is basically, $\phi(x;\theta)w+b$. We use a broad class of mappings $\phi(x;\theta)$ where $\theta$ are parameters that we learn. 

- In a neural network, if nonlinear activation functions are not used, the whole network would only be able to represent a linear function. Therefore, a nonlinear activation function is necessary to increase the representing power of a network. In modern neural nets, this is done using ReLU (rectified linear) function. The nonlinearity of NNs cause most interesting loss functions to be non-convex which has no convergence gaurantees and may depend on initialization.

- Most modern neural nets are trained using maximum likelihood so the cost function becomes negative log likelihood or cross-entropy loss between training data emprical distribution and model distribution (E_p_em[log p_model(y/x)]) plus some regularization. We want cost functions that don't satuarate or become flat since those would cause zero gradients. If we use satuarating activation functions or exp output units, we might encounter saturaing cost. In addition, we don't want operations like division in the loss function since those functions and their gradients would have numerical difficulties like underflow or overflow. Maximum likelihood learning helps with exp outputs; for example sigmoid units for bernoulli output distribution (binary output) or softmax unit for multinoulli (discrete variable with n possible values) to exponentiate and normalize the output as a probability between [0,1]. We should always think about when the gradients of the loss become very small which will result in no more learning so choice of output units an loss function go hand in hand! 

- We can learn probability distributions by parameterizing them (for example, mean and variance) and have a neural net function output the parameters. We must note that the conditions of parameters need to be met at the output of the neural network. For example, a covariance matrix is positive definite so a neural net producing covariance matrix parameters has to produce a positive definite matrix (e.g. by having the network learn its Cholesky decomposition) that we can use as covariance matrix in our distribution. Note that computing the determinant and inverse of a full rank covariance matrix for the likelihood function is expensive O(d^3). We can use maximum likelihood or MAP learning for establishing a cost function and learn the neural net function for mapping from x data to parameters of the distribution (mean/variance). 

- Combine this idea with graphical models; now you have structured probabilistic models and you can use NN function to learn their parameters! We can also use another NN function for the generative part of the graphical model as well. If we use reparameterization trick to make the probability distribution differentiable we can train the model end to end and we get a VAE! For example, We might want to use a NN as an inference function for parameters of a Gaussian mixture model $p(y/x)=\sum (p(c/x)p(y/c))$. The network needs to have three sets of outputs. A vector defining p(c/x) that chooses which Gaussian should produce y defining a multinoulli distribution (i.e. softmax output); A matrix providing means for multi-dimensional Gaussian components (a linear output with no nonlinearity); And a tensor defining covariance matrices for multi-dimensional Gaussian components. Maximum likelihood is a little bit more complicated to write down for this model! 

- ReLU activation function has derivative of 1 in all places that the neuron is active and 0 everywhere else. This means that when the neuron is active, the gradients can pass through very well but on the other hand, if a neuron gets unactive, no gradients can pass through and no learning can happen on that neuron! Fixes include leaky ReLU where the unactivated neuron passes a small value, a parametric ReLU where the small value is learned, or absolute value where -1 is passed which effectively passes absolute value and not polarity.







- PyTorch has four main features:
1. it provides a multi-dimensional array like Numpy array that can be processed on GPU when it's datatype is cast as (torch.cuda.TensorFloat). This array and it's associated functions are general scientific computing tool and don't know anything about gradients, optimization or neural nets in general. 

2. The second feature is the autograd package which provides the ability to define a computational graph so that we can automatically compute gradients. In the computational graph, a node is an array and an edge is an operation on arrays. To make a computational graph, we make a node by wrapping an array inside (torch.aurograd.Variable()) function. Then all operations that we do on this node will be defined as edges and their results will be new nodes in the computational graph. Each node in the graph (i.e. Variable()), has a (node.data) property which is a multi-dimensional array and a (node.grad) property which is it's gradient with respect to some scalar value (node.grad is also a Variable()). After defining the computational graph this way, with a single command (loss.backward()) we can calculate gradients of loss with respect to all nodes in the graph. 

3. The third feature is a high-level neural networks library (torch.nn) that abstracts away all parameter handling in layers of neural networks to make it easy to define a NN in a few commands (e.g. torch.nn.conv). This pachage also comes with popular loss functions (e.g. torch.nn.MSEloss). We start with defining a model container, for example a model with a sequence of layers using (torch.nn.Sequential) and then list the layers that we desire in a sequence. The library handles every thing else; we can access the parameters (Variable()) using (model.parameters()) 

4. The forth feature is an optimization package (torch.optim) that works in tandem with the NN library. This library contains more sophisticated optimzers like Adam, RMSprop, etc. We define an optimzer and pass network parameters and learning rate to it (opt=torch.optim.Adam(model.parameters(), lr=learning_rate)) and then we just call (opt.step()) to do a one step update on our parameters. 

We can also define custom layers by subclassing (torch.nn.Module) and implementing a (forward()) function that accepts a (Variable()) as input and produces a (Variable()) as output. We can also do a dynamic network by defining a layer that morphs in time!