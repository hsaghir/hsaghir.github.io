---
layout: article
title: Numerical Computation Deep Learning book Ch4
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

- fundamental difficulty in performing continuous math on a digital computer is that we need to represent real numbers with a finite (digital) number leading to unstability of algorithms that should theoretically work due to build up of rounding error.

- underflow happens when numbers near zero are rounded to zero.This is problematic for example, division by zero or log zero. overflow happens when numbers with large magnitude are approximated as infinity. for example softmax function over/under flows with constant probability. So every algorithm/function has to be implemented in a numerically stable way that prevents over/under flow. 

- Conditioning refers to how rapidly a function changes with respect to small changes in its inputs. If a function changes rapidly with small changes in input, the small errors in input become problematic. 

- Condition number of a matrix is the ratio of largest eigenvalue to smallest eigenvalue. If it's large, matrix inversion is sensitive to error. Poorly conditioned matrices amplify pre-existing errors due to sensitivity. 

- Note that the derivative specifies how to scale a small change in the input in order to obtain the corresponding change in the output. f(x + \epsilon) ≈ f(x) + \epsilon*df(x). This intuition leads to Gradient descent where we move in the negative direction of gradient to make the value of the function smaller. Also note that \epsilon corresponds to the notion of learning rate (How much do we want to move in the output space). 

- Gradient (all elements) becomes zero at critical points i.e. minimum/maximum/saddle. In the context of deep learning we might encounter high dimensional functions with many local minima and saddle points surrounded by flat space. This makes optimization difficult so we settle for low values of function without reaching critical points. 

- We usually work with scalar functions (loss) that have multi-dimensional inputs. So we work with partial derivatives that tell us how the function changes in each direction of input. The gradient is a vector of changes in all directions of input. Directional derivative is the projection of gradient (dot product) on a certain direction (vector). To minimize the function, we want to find the direction that the function decreases along the fastest, so if we write the maximization problem, we'll find that the direction of the gradient is the direction along which the function decreases/increases the fastest. 

- If the function is also a vector, then it's derivative is a matrix called Jacobian which is simply the dervative wrt every input and output elements. The second derivative of a scalar function will therefore be a matrix (Hessian) since the gradient is a vector function and it's derivative will have a Jacobian. The second derivative is intuitively the curvature of the function which tells us how the gradient changes. So it's useful in optimization since it can tell us how much improvement a gradient step will yield (what multiple of the learning rate decrease we will see in the function). 

- Hessian is a real symmetric (derivative is commutative) matrix so can be decomposed to eigenvalues and eigenvectors. The directional second derivative is given by $d^T H d$ where d is a unit vector in an arbitrary direction. If d is an eigenvector of Hessian, then the second derivative in that direction will be the corresponding eigenvalue, otherwise a combination of eigenvalues.

- If we write second order taylor expansion for one gradient step ahead (x0-lr*g), we'll have a $lr(g^T H g)$ term appear which means that the value of second derivative in direction of gradient times learning rate will influence where we end up in the next gradient step. So we can set the learning rate in a way that leads to the maximum decrease in function value in the next gradient descent step; that is estimated by (g^T g/ gT H g). Therefore, if the direction of gradient is the same as an eigenvalue of H, the optimal lr is 1/eigenvalue. So in a sense, eigenvalues of Hessian scale the maximum improvement possible in next gradient step (optimal lr). 

- Second derivative at critical points tell the type of the point. positive definite Hessian means minimum, negative definite Hessian maximum, and at least one positive one negetive eigenvalues in Hessian means saddle.

- When Hessian has poor condition number (max eigenval/min eigenval), gradient descent doesn't work well since it doesn't know that gradients changes fast in one direction but not another so setting learning rate will be challenging. Second order methods (i.e. Newton / HFO) solve this. We write first order taylor series, differentiate, and solve for critical point which gives us the location of local minimum in a single step (x0-H x0^-1 g) but not very useful near a saddle (GD is better in that case since it doesn't get attracted to the saddle). 

- Functions in deep learning are usually not convex (positive semi-definite Hessian), and have saddle points so optimizations don't usually come with garauntee. We can sometimes get garauntees by constraining functions to those that whose rate of change (abs(f(x)-f(y))) are bounded by some constant L(x-y) (Lipschitz continuity), since then we can say that small changes in input result in a bounded change in output. 

- We might want to do constrained optimization. generalized Lagrange multipliers (KKT) approach convert constraints in the form of equalities and inequalities into a new loss function (lagrangian) with a sum of original function and a linear combination of the equalities and inequalities. The coeffiecients of the equalities/inequalities are called lagrange multipliers. optimizing the original loss under constraints (minimization) is the same as optimizing the lagrangian (min x max multipliers optimization) without constraints.

- Inequality constraints are interesting in KKT. To gain some intuition for this idea, we can say that either the solution is on the boundary imposed by the inequality and we must use its KKT multiplier to influence the solution to x, or the inequality has no influence on the solution and we represent this by zeroing out its KKT multiplier

- Least squares is an example of variational approach (optimization) to solving a quadratic cost of (Ax+b)^2. compute the gradient and use GD or compute second derivative and use Newton's method to find optimum point. Constrained version can be solved using KKT and linear algebra or optimization.





There are a few classes of optimization algorithms that are frequently used in machine learning. 

- Gradient-based:
  1st order - (SGD, Adagrad, Adam,  Batch Gradient Descent)
  2nd order - (either by computing the Hessian or approximating it e.g. Newton method, conjugate gradient, scaled conjugate gradient, HFO)

- Inference Algorithms:
  Expectation Maximization (EM)
  Message passing

Search based techniques:
genetic algorithms, simulated annealing, Morkov Chain Monte Carlo?l 
These techniques usually don't require the function being optimised to be differentiable, they try to find a solution by sampling from a probability distribution.






You've defined your neural net architecture. How the heck do you train it? The basic workhorse for neural net training is [stochastic gradient descent (SGD)](https://metacademy.org/concepts/stochastic_gradient_descent), where one visits a single training example at a time (or a "minibatch" of training examples), and takes a small step to reduce the loss on those examples. This requires computing the [gradient](https://metacademy.org/concepts/gradient) of the loss function, which can be done using [backpropagation](https://metacademy.org/concepts/backpropagation). Be sure to [check your gradient computations](http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization) with finite differences to make sure you've derived them correctly. SGD is conceptually simple and easy to implement, and with a bit of tuning, can work very well in practice.

There is a broad class of optimization problems known as [convex optimization](https://metacademy.org/concepts/convex_optimization), where SGD and other local search algorithms are guaranteed to find the global optimum. This occurs because the function being optimized is "bowl shaped" (convex) and local improvements in the optimization function work towards the global optimum. Much of machine learning research is focused on trying to formulate things as convex optimization problems. Unfortunately, deep neural net training is usually not convex, so you are only guaranteed to find a local optimum. This is a bit disappointing, but ultimately it's [something we can live with](http://videolectures.net/eml07_lecun_wia/). For most feed-forward networks and generative networks, the local optima tend to be pretty reasonable. (Recurrent neural nets are a different story --- more on that below.)

A bigger problem than local optima is that the curvature of the loss function can be pretty extreme. While neural net training isn't convex, the problem of curvature also shows up for convex problems, and many of the techniques for dealing with it are borrowed from convex optimization. As general background, it's useful to read the following sections of Boyd and Vandenberghe's book, [Convex Optimization](http://www.stanford.edu/~boyd/cvxbook/):

-   [Sections 9.2-9.3](http://www.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=477) talk about gradient descent, the canonical first-order optimization method (i.e. a method which only uses first derivatives)
-   [Section 9.5](http://www.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf#page=498) talks about Newton's method, the canonical second-order optimization method (i.e. a method which accounts for second derivatives, or curvature)

While Newton's method is very good at dealing with curvature, it is impractical for large-scale neural net training for two reasons. First, it is a batch method, so it requires visiting every training example in order to make a single step. Second, it requires constructing and inverting the Hessian matrix, whose dimension is the number of parameters. ([Matrix inversion](https://metacademy.org/concepts/computing_matrix_inverses) is only practical up to tens of thousands of parameters, whereas neural nets typically have millions.) Still, it serves as an idealized second-order training method which one can try to approximate. Practical algorithms for doing so include:

-   [conjugate gradient](http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf)
-   limited memory BFGS

Compared with most neural net models, training RBMs introduces another complication: computing the objective function requires computing the partition function, and computing the gradient requires performing [inference](https://metacademy.org/concepts/inference_in_mrfs). Both of these problems are [intractable](https://metacademy.org/concepts/complexity_of_inference). (This is true for [learning Markov random fields (MRFs)](https://metacademy.org/concepts/mrf_parameter_learning) more generally.) [Contrastive divergence](http://learning.cs.toronto.edu/~hinton/csc2535/readings/nccd.pdf)and [persistent contrastive divergence](http://www.cs.utoronto.ca/~tijmen/pcd/pcd.pdf) are widely used approximations to the gradient which often work quite well in practice. Evaluating the models remains a difficult problem, though. One can [estimate the model likelihood](http://www.cs.utoronto.ca/~rsalakhu/papers/dbn_ais.pdf) using [annealed importance sampling](https://metacademy.org/concepts/annealed_importance_sampling), but this is delicate, and failures in estimation tend to overstate the model's performance.

Even once you understand the math behind these algorithms, the devil's in the details. Here are some good practical guides for getting these algorithms to work in practice:

-   G. Hinton. [A practical guide to training restricted Boltzmann machines.](http://www.csri.utoronto.ca/~hinton/absps/guideTR.pdf) 2010.
-   J. Martens and I. Sutskever. [Training deep and recurrent networks with Hessian-free optimization.](http://www.cs.utoronto.ca/~ilya/pubs/2012/HF_for_dnns_and_rnns.pdf) Neural Networks: Tricks of the Trade, 2012.
-   Y. Bengio. [Practical recommendations for gradient-based training of deep architectures.](http://arxiv.org/pdf/1206.5533) Neural Networks: Tricks of the Trade, 2012.
-   L. Bottou. [Stochastic gradient descent tricks.](http://research.microsoft.com/pubs/192769/tricks-2012.pdf) Neural Networks: Tricks of the Trade, 2012.