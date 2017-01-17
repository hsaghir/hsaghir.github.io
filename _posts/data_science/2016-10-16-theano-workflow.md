---
layout: article
title: Theano workflow
comments: true
categories: data_science
image:
  teaser: practical\Deep_RBM.png
---

Theano might look intimidating, but there are a few concepts that if understood, would make the engineering involved in deep learning more tangible. The first is the concept of tensors and operations on them. A tensor is simply a multi-dimensional array. For example a 1d tensor is a vector and a 2d tensor is a matrix. We also want to do efficient operations on tensors such as addition, multiplication, etc . Tensors are useful in deep learning since we usually want to work with multi-dimensional data (e.g. images are 2d plus color channels i.e. 3d) and we also want to send a number of data points into the model at a time (batch which adds another dimension). 

Making these operations on a tensor object efficient needs careful use of computational resources. This is usually done by hardware manufacturers that write highly efficient code libraries (e.g. BLAS, cuBLAS, cuDNN) to provide the capability to perform such operations on their chips. The architecture of GPUs, in particular, allow them to perform multiple operations in parallel and thus are highly in demand for deep learning applications. theano explicitly distinguishes between CPU and GPU operations and thus we need to know which one we want to use. 

Another important concept to keep in mind is the idea of computational graphs. theano makes a computational graph of symbolic variables to make the automatic differentiation possible. For example consider $ c=a+b $, this computation can be represented as a->c<-b. Such representation can modularize the computations so that the GPU and the CPU can handle different parts in which they are more adept at. Additionally the computational graph can make automatic differentiation very easy. The gradient of any function represented as a computational graph can be computed using a few general rules and thus we get automatic differentiations for free! Therefore, defining a model in theano involves specifying the structure of the symbolic computational graph and then compiling the graph into a model using the **theano.function()** method. Here is a theano pseudo-code for a simple sigmoid neural net layer without the details to  layout the work flow in defining a model in theano.

```
import theano 
import theano.tensor as T

X=x #x is the numpy input tensor defining the shape
W=theano.shared(w, theano.config.floatx) # w is a numpy tensor defining the shape
B=theano.shared(b, theano.config.floatx) # b is a numpy tensor defining the shape

y_out=T.nnet.softmax(T.dot(W,X)+B) #defining the softmax nonlinearity layer

loss=y*T.log(y_out)+(1-y)*T.log(1-y_out)  # y is the target value

# gradients
g_w=T.grad(loss,W)
g_b=T.grad(loss,B)

# gradient descent updates
updt=[(W, W - lr*g_w), (B, B - lr*g_b)]

model_train=theano.function(input=X, output=loss, updates=updt) # training model

model_pred=theano.funtion(input=X, output=y_out) # prediction model

for i in range(iter): # number of iterations
  cost=model_train(X)

predictions=model_pred(test_X) # testing the model!

```

This is the meat of a making a model with theano! There are of course details in the actual implementations that I skipped in the interest of understanding but you can easily figure them out by looking at [this](http://deeplearning.net/tutorial/logreg.html) working theano example!