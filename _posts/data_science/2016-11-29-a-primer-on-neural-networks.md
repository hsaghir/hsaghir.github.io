---
layout: article
title: A introduction to Neural Networks without any formula
comments: true
image:
  feature: Aneuron.jpg
---

What is a neural network? To get started, it's beneficial to keep in mind that modern neural network started as an attempt to model the way that brain performs computations. We have billions of [neuron](https://en.wikipedia.org/wiki/Neuron) cells in our brains that are connected to each other and pass information around. 

![alt text](/images/neuron.jpg "A Neuron Cell")


An artificial neuron is a simplified model of the neuron cell. It takes several inputs x1,x2, ... and produces a [linear combination](https://en.wikipedia.org/wiki/Linear_combination) of the inputs and then applies some sort of a [nonlinear function](http://www.glencoe.com/sec/math/prealg/prealg05/study_guide/pdfs/prealg_pssg_G112.pdf) to them. The parameters for the linear combination part are called weights and biases and the nonlinear function is usually a [sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function), [tanh](https://reference.wolfram.com/language/ref/Tanh.html) or a simple line that is clipped at one end (rectified linear unit- [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))). 

![alt text](/images/Aneuron.jpg "An artificial Neuron ")


You can make a neural network by connecting several neurons with each other. A common way of doing this is by arranging them in layers so that a layer of neurons get their inputs from the previous layer and give their outputs to the next. So a neural network simply gets some inputs, performs a succession of linear combinations and nonlinearities and produces some output. The important point is that such a network has parameters, i.e. weights and biases, that in some sense linearly scale the outputs of one layer before it goes to the next layer. You can set these parameters as whatever you want and as a result, the output of the network will change. So it is easy to imagine that if you set them in some clever way, you might be able to make the network produce outputs like another function. In fact, it has been shown that given enough layers and neurons, a neural network can estimate any function. 

![alt text](/images/neuralNet.jpg "A Neural Network ")


So the question is how to set such clever weights and biases for different layers in the network to make it do useful things. For example, if I want my neural network to estimate a function for house prices based on size and neighborhood informations, how can I find the proper parameters? Well, the cool thing about neural networks is that they can "learn" their parameters by themselves without the need for me to set them by hand. So what does that learning mean? It means that if I have enough examples of (size, neighborhood) vs. (house price), a clever but simple [optimization](https://en.wikipedia.org/wiki/Mathematical_optimization) technique can iteratively find the best parameters for the network for this data. 

How does this clever optimization work, you ask? In the beginning, it uses some random numbers for weights and biases and then calculates an output based on your (size, neighborhood) as the input to the network. Such an estimated (house price) information is obviously not that good since the parameters were random. Then you calculate how far it is from the real (house price) value for the (size, neighborhood) information you gave it as the error. If we can reduce this error for all examples we are getting closer to good parameter values for our network. So if you imagine all error values as being plotted on a curve, we can remember from high school calculus that the derivative (gradient) of this error is equal to the slope of the tangent line at that point. So if we want to get to the minimum of the error curve, we simply have to move in reverse direction of gradient (we'll get to maximum if we move in the direction of gradient). This is the clever optimization technique that we talked about earlier and is called a fancy name i.e. "[stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)".

![alt text](/images/sgd.jpg "stochastic gradient descent")


How do we move toward this minimum, you might ask? We are interested in network weights and biases that produce the minimum error. What good is the gradient of error, if we only care about finding the right parameters? Well, if you remember the [chain rule](https://en.wikipedia.org/wiki/Chain_rule) for calculating derivatives, you know that you can calculate the derivative for the previous layers based on the error value at the output layer. This is called another fancy name i.e. "[back-propagation](https://en.wikipedia.org/wiki/Backpropagation)". We now have the gradient at each layer, we can just deduct a percentage ([learning rate](http://datascience.stackexchange.com/questions/410/choosing-a-learning-rate)) of the derivative from each parameter according to our clever optimization technique. Each time we do this procedure, we can reduce the error and get closer to good parameters for the network. 

![alt text](/images/chainrule.jpg "The chain Rule")


This is the gist of neural networks but of course there are some details that I skipped here. The core idea is simple and elegant. You might have heard of deep learning. It is interesting to know that deep learning means just more layers between input and output. Before 2006, people thought that calculating the weights and biases would not be possible for deep network but with today's fast GPUs and the vast amount of labeled data that we have today, it is now possible and it does wonders!