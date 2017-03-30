---
layout: article
title: ConvNets Deep Learning book Ch9
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

- DL practitioners need to decide the right course of action on whether to gather more data, increase or decrease model capacity, add or remove regularizing features, improve the optimization of a model, improve approximate inference in a model, or debug the software implementation of the model.

- Design process (1)Determine what error metric to use, and your target value for this error metric. These goals and error metrics should be driven by the problem that the application is intended to solve. (2)Establish a working end-to-end pipeline as soon as possible, including the estimation of the appropriate performance metrics. (3) Instrument the system well to determine bottlenecks in performance. Diagnose which components are performing worse than expected and whether it is due to overfitting, underfitting, or a defect in the data or software. (4) Repeatedly make incremental changes such as gathering new data, adjusting hyperparameters, or changing algorithms, based on specific findings from your instrumentation.

- 













































## [From other places on web!](https://nmarkou.blogspot.ca/2017/02/the-black-magic-of-deep-learning-tips.html?utm_campaign=Revue+newsletter&utm_medium=Newsletter&utm_source=revue)

-   Always shuffle. Never allow your network to go through exactly the same minibatch. If your framework allows it shuffle at every epoch. 
-   Expand your dataset. DNN's need a lot of data and the models can easily overfit a small dataset. I strongly suggest expanding your original dataset. If it is a vision task, add noise, whitening, drop pixels, rotate and color shift, blur and everything in between. There is a catch though if the expansion is too big you will be training mostly with the same data. I solved this by creating a layer that applies random transformations so no sample is ever the same. If you are going through voice data shift it and distort it
-   This tip is from Karpathy, before training on the whole dataset try to overfit on a very small subset of it, that way you know your network can converge.
-   Always use dropout to minimize the chance of overfitting. Use it after large > 256 (fully connected layers or convolutional layers). There is an excellent thesis about that ([Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning](https://arxiv.org/abs/1506.02142))
-   Avoid LRN pooling, prefer the much faster MAX pooling.
-   Avoid Sigmoid's , TanH's gates they are expensive and get saturated and may stop back propagation. In fact the deeper your network the less attractive Sigmoid's and TanH's are. Use the much cheaper and effective ReLU's and PreLU's instead. As mentioned in [Deep Sparse Rectifier Neural Networks](http://jmlr.org/proceedings/papers/v15/glorot11a/glorot11a.pdf) they promote sparsity and their back propagation is much more robust.
-   Don't use ReLU or PreLU's gates before max pooling, instead apply it after to save computation
-   Don't use ReLU's they are so 2012. Yes they are a very useful non-linearity that solved a lot of problems. However try fine-tuning a new model and watch nothing happen because of bad initialization with ReLU's blocking backpropagation. Instead use PreLU's with a very small multiplier usually 0.1. Using PreLU's converges faster and will not get stuck like ReLU's during the initial stages. [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852). ELU's are still good but expensive.
-   Use Batch Normalization (check paper[ Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)) ALWAYS. It works and it is great. It allows faster convergence ( much faster) and smaller datasets. You will save time and resources.
-   I don't like removing the mean as many do, I prefer squeezing the input data to [-1, +1]. This is more of  a training and deployment trick rather a performance trick.
-   Always go for the smaller models, if you are working and deploying deep learning models like me, you quickly understand the pain of pushing gigabytes of models to your users or to a server in the other side of the world. Go for the smaller models even if you lose some accuracy.
-   If you use the smaller models try ensembles. You can usually boost your accuracy by ~3% with an enseble of 5 networks. 
-   Use xavier initialization as much as possible. Use it only on large Fully Connected layers and avoid them on the CNN layers. [An-explanation-of-xavier-initialization](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization)
-   If your input data has a spatial parameter try to go for CNN's end to end. Read and understand [SqueezeNet ](https://arxiv.org/abs/1602.07360), it is a new approach and works wonders, try applying the tips above. 
-   Modify your models to use 1x1 CNN's layers where it is possible, the locality is great for performance. 
-   Don't even try to train anything without a high end GPU.
-   If you are making templates out of models or your own layers, parameterize everything otherwise you will be rebuilding your binaries all the time. You know you will
-   And last but not least understand what you are doing, Deep Learning is the Neutron Bomb of Machine Learning. It is not to be used everywhere and always. Understand the architecture you are using and what you are trying to achieve don't mindlessly copy models.