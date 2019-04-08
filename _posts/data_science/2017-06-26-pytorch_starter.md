---
layout: article
title:  Understand PyTorch code in 10 minutes
comments: true
categories: data_science
image:
  teaser: practical/Pytorch_logo_new.png
---

So PyTorch is the new popular framework for deep learners and many new papers release code in PyTorch that one might want to inspect. Here is my understanding of it narrowed down to the most basics to help read PyTorch code. This is based on Justin Johnson's [great tutorial](https://github.com/jcjohnson/pytorch-examples). If you want to learn more or have more than 10 minutes for a PyTorch starter go read that!

<!-- 
```
[Edit: updating this post in honor of the release of version 1.0]
There has been significant changes and improvements to both pytorch front and backend since I originally wrote this post about one and half ago. Since I aim for this post to still be a quick 10-minute introduction to pytorch, I am not going to add many things, but rather update it to reflect the newest of changes in pytorch.
``` -->


PyTorch consists of 4 main packages:
1. torch: a general purpose array library similar to Numpy that can do computations on GPU when the tensor type is cast to (torch.cuda.TensorFloat)
2. torch.autograd: a package for building a computational graph and automatically obtaining gradients 
3. torch.nn: a neural net library with common layers and cost functions
4. torch.optim: an optimization package with common optimization algorithms like SGD,Adam, etc
5. torch.jit: a just-in-time (JIT) compiler that at runtime takes your PyTorch models and rewrites them to run at production-efficiency. The JIT compiler can also export your model to run in a C++-only runtime based on Caffe2 bits.

##### 0. import stuff
You can import PyTorch stuff like this:

```python
import torch # arrays on GPU
import torch.autograd as autograd #build a computational graph
import torch.nn as nn ## neural net library
import torch.nn.functional as F ## most non-linearities are here
import torch.optim as optim # optimization package

```


##### 1.  the torch array replaces numpy ndarray ->provides linear algebra on GPU support

PyTorch provides a multi-dimensional array like Numpy array that can be processed on GPU when it's data type is cast as (torch.cuda.TensorFloat). This array and it's associated functions are general scientific computing tool. 


confer [Torch for numpy users](https://github.com/torch/torch7/wiki/Torch-for-Numpy-users) for how it relates to numpy.


```python
# 2 matrices of size 2x3 into a 3d tensor 2x2x3
d = [ [[1., 2.,3.], [4.,5.,6.]], [[7.,8.,9.], [11.,12.,13.]] ]
d = torch.Tensor(d) # array from python list
print "shape of the tensor:", d.size()

# the first index is the depth
z = d[0] + d[1]
print "adding up the two matrices of the 3d tensor:",z
```

    shape of the tensor: torch.Size([2, 2, 3])
    adding up the two matrices of the 3d tensor: 
      8  10  12
     15  17  19
    [torch.FloatTensor of size 2x3]
    



```python
# a heavily used operation is reshaping of tensors using .view()
print d.view(2,-1) #-1 makes torch infer the second dim
```

    
      1   2   3   4   5   6
      7   8   9  11  12  13
    [torch.FloatTensor of size 2x6]
    


##### 2. torch.autograd can make a computational graph -> auto-compute gradients
The second feature is the autograd package that provides the ability to define a computational graph so that we can automatically compute gradients. In the computational graph, a node is an array and an edge is an operation on the array. To make a computational graph, we make a node by wrapping an array inside the **torch.aurograd.Variable()** function. All operations that we do on this node from then on will be defined as edges in the computational graph. The edges of the graph also result in new nodes in the computational graph. Each node in the graph has a **.data** property which is a multi-dimensional array and a **.grad** property which is it's gradient with respect to some scalar value (**.grad** is also a **.Variable()** itself). After defining the computational graph, we can calculate gradients of the loss with respect to all nodes in the graph with a single command i.e. **loss.backward()**. 


- Convert a Tensor to a node in the computational graph using *torch.autograd.Variable()*
    + access its value using *x.data*
    + access its gradient using *x.grad*
- Do operations on the *.Variable()* to make edges of the graph


```python
# d is a tensor not a node, to create a node based on it:
x = autograd.Variable(d, requires_grad=True)
print "the node's data is the tensor:", x.data.size()
print "the node's gradient is empty at creation:", x.grad # the grad is empty right now
```

    the node's data is the tensor: torch.Size([2, 2, 3])
    the node's gradient is empty at creation: None



```python
# do operation on the node to make a computational graph
y = x + 1
z = x + y
s = z.sum()
print s.creator
```

    <torch.autograd._functions.reduce.Sum object at 0x7f1e59988790>



```python
# calculate gradients
s.backward()
print "the variable now has gradients:",x.grad
```

    the variable now has gradients: Variable containing:
    (0 ,.,.) = 
      2  2  2
      2  2  2
    
    (1 ,.,.) = 
      2  2  2
      2  2  2
    [torch.FloatTensor of size 2x2x3]
    


##### 3. torch.nn contains various NN layers (linear mappings of rows of a tensor) + (nonlinearities ) -> helps build a neural net computational graph without the hassle of manipulating tensors and parameters manually

The third feature is a high-level neural networks library **torch.nn** that abstracts away all parameter handling in layers of neural networks to make it easy to define a NN in a few commands (e.g. **torch.nn.conv**). This package also comes with a set of popular loss functions (e.g. **torch.nn.MSEloss**). We start with defining a model container, for example, a model with a sequence of layers using **torch.nn.Sequential** and then list the layers that we desire in a sequence. The library handles every thing else; we can access the parameter nodes **Variable()** using **model.parameters()**


```python
# linear transformation of a 2x5 matrix into a 2x3 matrix
linear_map = nn.Linear(5,3)
print "using randomly initialized params:", linear_map.parameters
```

    using randomly initialized params: <bound method Linear.parameters of Linear (5 -> 3)>



```python
# data has 2 examples with 5 features and 3 target
data = torch.randn(2,5) # training
y = autograd.Variable(torch.randn(2,3)) # target
# make a node
x = autograd.Variable(data, requires_grad=True)
# apply transformation to a node creates a computational graph
a = linear_map(x)
z = F.relu(a)
o = F.softmax(z)
print "output of softmax as a probability distribution:", o.data.view(1,-1)

# loss function
loss_func = nn.MSELoss() #instantiate loss function
L=loss_func(z,y) # calculateMSE loss between output and target
print "Loss:", L
```

    output of softmax as a probability distribution: 
     0.2092  0.1979  0.5929  0.4343  0.3038  0.2619
    [torch.FloatTensor of size 1x6]
    
    Loss: Variable containing:
     2.9838
    [torch.FloatTensor of size 1]
    

We can also define custom layers by sub-classing **torch.nn.Module** and implementing a **forward()** function that accepts a **Variable()** as input and produces a **Variable()** as output. We can even make a dynamic network by defining a layer that morphs in time!

- When defining a custom layer, 2 functions need to be implemented:
    - *\__init\__* function has to always be inherited first, then define parameters of the layer here as the class variables i.e. *self.x*
    - forward funtion is where we pass an input through the layer, perform operations on inputs using parameters and return the output. The input needs to be an **autograd.Variable()** so that pytorch can build the computational graph of the layer.

```python
class Log_reg_classifier(nn.Module):
    def __init__(self, in_size,out_size):
        super(Log_reg_classifier,self).__init__() #always call parent's init 
        self.linear = nn.Linear(in_size, out_size) #layer parameters
        
    def forward(self,vect):
        return F.log_softmax(self.linear(vect)) # 
```


##### 4. torch.optim can do optimization -> we build a nn computational graph using torch.nn, compute gradients using torch.autograd, and then feed them into torch.optim to update network parameters

The forth feature is an optimization package **torch.optim** that works in tandem with the NN library. This library contains sophisticated optimizers like Adam, RMSprop, etc. We define an optimizer and pass network parameters and learning rate to it (i.e. **opt=torch.optim.Adam(model.parameters(), lr=learning_rate)**) and then we just call **opt.step()** to do a one-step update on our parameters. 


```python

optimizer = optim.SGD(linear_map.parameters(), lr = 1e-2) # instantiate optimizer with model params + learning rate

# epoch loop: we run following until convergence
optimizer.zero_grad() # make gradients zero
L.backward(retain_variables = True)
optimizer.step()
print L
```

    Variable containing:
     2.9838
    [torch.FloatTensor of size 1]
    

Building a neural net is easy. Here is an example showing how things work together:

```python

# define model
model = Log_reg_classifier(10,2)

# define loss function
loss_func = nn.MSELoss() 

# define optimizer
optimizer = optim.SGD(model.parameters(),lr=1e-1)

# send data through model in minibatches for 10 epochs
for epoch in range(10):
    for minibatch, target in data:
        model.zero_grad() # pytorch accumulates gradients, making them zero for each minibatch
        
        #forward pass
        out = model(autograd.Variable(minibatch))
        
        #backward pass 
        L = loss_func(out,target) #calculate loss
        L.backward() # calculate gradients
        optimizer.step() # make an update step
```


##### 5. torch.jit can compile python code -> useful for production of models
Regardless of whether you use tracing or @script, the result is a python-free representation of your model, which can be used to optimize the model or to export the model from python for use in production environments.


- The PyTorch tracer, torch.jit.trace, is a function that records all the native PyTorch operations performed in a code region, along with the data dependencies between them. In fact, PyTorch has had a tracer since 0.3, which has been used for exporting models through ONNX. What changes now, is that you no longer necessarily need to take the trace and run it elsewhere - PyTorch can re-execute it for you, using a carefully designed high-performance C++ runtime. As we develop PyTorch 1.0 this runtime will integrate all the optimizations and hardware integrations that Caffe2 provides.

- Tracing mode is a great way to minimize the impact on your code, but we’re also very excited about the models that fundamentally make use of control flow such as RNNs. Our solution to this is a scripting mode. In this case you write out a regular Python function, except that you can no longer use certain more complicated language features. Once you isolated the desired functionality, you let us know that you’d like the function to get compiled by decorating it with an @script decorator. This annotation will transform your python function directly into our high-performance C++ runtime. This lets us recover all the PyTorch operations along with loops and conditionals. They will be embedded into our internal representation of this function, and will be accounted for every time this function is run.

```python
from torch.jit import script

@script
def rnn_loop(x):
    hidden = None
    for x_t in x.split(1):
        x, hidden = model(x, hidden)
    return x
```