---
layout: article
title: Theano workflow
comments: true
categories: data_science
image:
  teaser: Deep_RBM.png
---

Theano might look intimidating, so here is a theano pseudo-code for a simple sigmoid neural net layer without the details to show how theano works. The important thing to keep in mind is that theano makes a computational graph of symbolic variables to make the automatic differentiation possible. Therefore, defining a model in theano involves specifying the structure of the symbolic graph and then compiling the graph using the **theano.function()** method. 

```
import theano 
import theano.tensor as T

X=x #x is the desired input tensor defining the shape
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