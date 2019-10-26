---
layout: article
title: Optimizers
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---



### Adam Optimizer

- [a review of various optimizers](http://ruder.io/optimizing-gradient-descent/)

- The idea of Adam Optimizer is why use the same learning rate for every parameter, when we know that some surely need to be moved further and faster than others. The square of recent gradients tells us how much signal we’re getting for each weight, we can just divide by that to ensure even the most sluggish weights get their chance to shine. Adam takes that idea, adds on the standard approach to momentum, and (with a little tweak to keep early batches from being biased) that’s it!

- When you hear people saying that Adam doesn’t generalize as well as SGD+Momentum, you’ll nearly always find that they’re choosing poor hyper-parameters for their model. Adam generally requires more regularization than SGD, so be sure to adjust your regularization hyper-parameters when switching from SGD to Adam.
	

- SGD update rule:
```python
w = w - lr * w.grad
```
- SGD with momentum update rule:
```python
moving_avg = alpha * moving_avg + (1-alpha) * w.grad 
w = w - lr * moving_avg
```

- Adam update rule:
```python
avg_grads = beta1 * avg_grads + (1-beta1) * w.grad
avg_squared = beta2 * (avg_squared) + (1-beta2) * (w.grad ** 2)
w = w - lr * avg_grads / sqrt(avg_squared)
```

- AMSGrad update rule: (AMSGrad dictates that the term `lr/ sqrt(avg_squared)` decrease over training)
```python
avg_grads = beta1 * avg_grads + (1-beta1) * w.grad
avg_squared = beta2 * (avg_squared) + (1-beta2) * (w.grad ** 2)
max_squared = max(avg_squared, max_squared)
w = w - lr * avg_grads / sqrt(max_squared)
```


- L2 regularization: penalizes the sum of squared weights
```python
final_loss = loss + wd * all_weights.pow(2).sum() / 2
```

and the SGD update with this regularization looks like:
```python
w = w - lr * w.grad - lr * wd * w
```
In practice, `wd * w` is added to `w.grad` which works for simple SGD, but if there is also a momentum term or if it's an adaptive optimizer like Adam this is wrong. This mistake is done in almost all libraries. 

- To fix this, we set weight decay argument of the optimizers to zero, and then after the `.backward()` function calculates the gradients, we then subtract the weight decay gradients, like this:

```python
loss.backward()
for group in optimizer.param_groups():
    for param in group['params']:
        param.data = param.data.add(-wd * group['lr'], param.data)
optimizer.step()

```