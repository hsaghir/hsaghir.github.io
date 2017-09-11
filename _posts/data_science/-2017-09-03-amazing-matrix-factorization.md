---
layout: article
title: The amazing power of matrix factorizations
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

Many common modeling assumptions, or combinations thereof, can be expressed by a class of probabilistic models called matrix decompositions. In a matrix decomposition model, component matrices are first sampled independently from a small set of priors, and then combined using simple algebraic operations. The space of such models is compositional: each model is described recursively in terms of simpler matrix decomposition models and the operations used to combine them.


### Automatic model selection

Most such work focuses on determining the particular factorization and/or conditional independence structure within a fixed model class. Our concern, however, is with identifying the overall form of the model. Right now, the choice of modeling assumptions is heavily dependent on the intuition of the human researcher; we are interested in determining appropriate modeling assumptions automatically.

### Factorization Machines

A factorization machine is like a linear model, except multiplicative interaction terms between the variables are modeled as well. The input to a factorization machine layer is a vector, and the output is a scalar. A linear model, given a vector `x` models its output `y` as

$$ y(x) = w_0 + \sum_i w_1i x_i$$

where `w` are the learnable weights of the model.

However, the interactions between the input variables `x_i` are purely additive. In some cases, it might be useful to model the interactions between your variables, e.g., `x_i * x_j`. You could add terms into your model like


$$ y(x) = w_0 + \sum_i w_1i x_i + \sum_i \sum_j w_2ij x_i x_j$$

However, this introduces a large number of `w2` variables. Specifically, there are `O(n^2)` parameters introduced in this formulation, one for each interaction pair. A factorization machine approximates `w2` using low dimensional factors, i.e.,

$$ y(x) = w_0 + \sum_i w_1i x_i + \sum_i \sum_j <vi,vj> x_i x_j$$

where each `v_i` is a low-dimensional vector. This is the forward pass of a second order factorization machine. This low-rank re-formulation has reduced the number of additional parameters for the factorization machine to `O(k*n)`. Magically, the forward (and backward) pass can be reformulated so that it can be computed in `O(k*n)`, rather than the naive `O(k*n^2)` formulation above.

- Kernel trick is used in SVMs to define a convenient similarity function that is the result of a dot product in a very high dimensional space. Matrix factorization models assume that the large data matrix are the higher dimensional mappings of a lower dimensional space and try to find the lower dimensional representation.

- Factorization machines use ideas from both SVM and matrix factorizations. They use the lower dimensional space idea of matrix factorization as a space where relationships among features are encoded. Then they use the kernel trick of SVMs by saying that some engineered features are the similarity kernels of that lower dimensional space. 

- For example, they use feature engineering intuitions e.g. that features are more important in pairs (i.e. similar to a polynomial-2 kernel in SVM). This feature engineering will produce an insanely large feature space. The key idea is that they now define the weights of feature pairs as the similarity kernel function $$w_ij = <v_i,v_j>$$ and assume that the $$v_i$$ space is a lower dimensional space that can be found with matrix factorization.

- FMs are general predictor working with any real valued feature vector. 
    + In contrast to SVMs, FMs model all interactions between variables using factorized parameters. Thus they are able to estimate interactions even in problems with huge sparsity (like recommender systems) where SVMs fail. 
    + In contrast to matrix factorization models, the drawback of matrix factorization models is that they are not applicable for general prediction tasks but work only with special input data. Furthermore their model equations and optimization algorithms are derived individually for each task. We show that FMs can mimic these models just by specifying the input data (i.e. the feature vectors).


### SVM
- Support vector machines are linear classifiers that interpret a linear transformation of the data as a hyperplane $$wx+b = 0$$ separating the two classess. SVMs try to find the hyperplane with the maximum seperation between classes (i.e. maximum k where $$wx+b>k$$ for class 1 and $$wx+b<-k$$ for class -1). The seperation margine we want to maximize is the normalized distance between the two parallel hyperplanes $$(1)wx+b=k , (2)wx+b=-k$$ that touch the data in the two classes i.e. $$d = (1)-(2) = \frac{2k}{|w|}$$. So this will be a constrained optimization problem with $$max \frac{2k}{|w|}$$ subject to $$y_i f(x_i) > k with y_i=1 for class 1 and y_i=-1 for class -1$$. This is a quadratic optimization with a unique maximum. 

- The constraints are hard at the moment, we can make it a soft constraint by adding new terms and removing constraints. We allow an $$\epsilon$$ violation of the margin i.e. $$y_i f(x_i)>k-\epsilon$$ and introduce a hyperparameter $$C$$ that determines the degree of importance of the violations i.e.  $$C(\epsilon)$$. Therefore the new cost function becomes $$min |w|^2 + C\sum \epsilon$$. Replacing $$\epsilon$$ with the constraint formula gives us the hinge loss definition i.e. $$ max[0, (1-y_i f(x_i))]$$.
 
- The total cost function of the SVM problem then is $$min (|w|^2 + C \sum_i max[0, (1-y_i f(x_i))] )$$. The first term of the total cost can be interpreted as L2 regularization and the second term as hinge loss.

- We can use SGD to solve this optimization problem. However, since the hinge loss is not differentiable, the two sub-gradients for different cases of $$ max[0, (1-y_i f(x_i))]$ would be to use the gradient of the differentiable part when that's the case and use $$w$$ otherwise.

#### Alternative SVM view and VC dimension
- Another way to think about SVMs is to put a bubble around each data point in data space; as we increase the radius of these bubbles, the set of possible solutions that don't violate the data bubbles and seperate the two classes decreases until a single solution is found which is the optimum solution with maximum bubble radius possible. Note that in this analogy, only the data points from the two classes that are close to each other will determine the decision boundry. Other points are simply having fun with their bubbles far away from the hyperplane. These critical points are called support vectors. 

- A breakthrough in statistical theory of machine learning by Vapnik and Chernovsky (VC) was the finding that the test error is bounded by the training error plus a monotonicly increasing funtion of a quantity called VC dimension. VC dimension is the minimum of data dimensionality and a ratio of bubbles {a bubble around all data over the bubble of support vectors}. This is extremely important since if we can reduce the bubble ratio, we can get around the curse of dimensionality in data! Therefore, there are two ways in which we can minimize the ratio. 1) is to keep the numerator fixed by constraining the scale of the scale of the data to be one and the maximize the margine. 2) is to keep the denominator fixed by imposing the constraint that margin be one, [therefore the the output of the model (wx+b) would be bigger than one for class one and smaller than -1 for class -1; if d is labels +1 or -1 then $d(wx+b)>1$], and minimize the data coefficients (w) to best rescale the data bubble. So the optimization problem accordint to (2) would be $min: L=1/2 w^Tw subject to d(wx+b)>1$. This can be rewritten using lagrange multipliers as $min: J(w, b, \alpha)= 1/2 w^Tw - \sum(\alpha_i d_i (w^Tx_i+b)) + \sum(\alpha_i))$ where $\alpha_i$ are lagrange multipliers for data samples $x_i$. Imagine a horse seat; This objective has to be minimized wrt (w, b) to get to minimum of the seat and maximized wrt $\alpha$ to get to the top of the seat (saddle point). 

- By taking the partial derivative of J wrt [w,b] and equating to zero, we can get the optimal $w=\sum(\alpha_i d_i x_i)$. We also know from KKT optimization (ch4 DL book) for non-equality constraints that $\alpha_i [d_i (w^Tx_i+b)-1]=0$ which will give us b. This constraint also shows that for non support vectors the $\alpha$ is zero and therefore, in linear SVM, the boundary is a linear combination of support vectors. Replacing the optimum [w,b] into the cost function J will yield a convext cost in terms of only $\alpha$, which we can solve using a QP solver. 

#### Kernel trick
- A kernel is basically a mapping from original data space to a target (usually much bigger) space $$v -> \Phi(v)$$. As one can imagine, computing such mappings are very hard. The key idea is to bypass the calculations of direct mappings of vectors based on the fact that a similariy function between vectors in the target high dimensional space $$K(v1, v2) = \phi(v1).\phi(v2)$$ can be directly calculated from the low-dimensional represenation without knowing the target high dimensional representation. 

- So based on this knowledge, we usually reverse everything i.e. define a similarity function that is convenient in the original space, and then hope that the target high dimensional space where this similarity function is calculated has a better representational power than the original space. For example consider the exponential kernel (similarity) as $$K(u,v) = \exp(u-v)$$ which is easy to compute but we don't even know the high dimensional mapping.

- A key innovation in machine learning is based on the observation that most machine learning algorithms can be written in terms of dot product between examples. Therefore, using kernel trick, we can think of the dot product as a similarity function of high dimensional mappings of data points without actually doing (or knowing) the mapping. 

- For example in linear SVM shown above, by replacing w with it's optimal value we can rewrite $$ w^T x+b = b + \sum(\alpha_i x^T x_i) $$. This makes it possible to just transform the example space, x, by some nonlinear function as a pre-processing before passing it through a linear SVM. The kernel trick introduces a kernel function (similarity function) $$k(x,x_i)=\phi(x).\phi(x_i)$$ which is similar to performing dot product on transformed version of input space x. The added value is that in most cases the nonlinear transformation $$\phi(.)$$ might be intractable, but we can evaluate the kernel (similarity) k more efficiently. 

- Additionally, since the $$w^T x_prime +b$$ on the transformed data space is still linear in coefficient w, we get a convex optimization problem with convergence gaurantee!

- The most common kernel (similarity function) is the Gaussian kernel $$k(u,v)= N(u-v, \sigma^2 I)$$, also known as radial basis function which corresponds to a dot product in an infinite dimensional space! A drawback of kernel methods is that the cost of evaluating the decision boundary is high because the boundary is a linear combination of kernelled version of all examples. SVMs are a little better in this regard, since to classify a new example, they only need to evaluate the kernel for support vectors (other examples don't contribute to the decision boundary).

### Factorization Machines

A factorization machine is like a linear model, except multiplicative interaction terms between the variables are modeled as well. The input to a factorization machine layer is a vector, and the output is a scalar. A linear model, given a vector `x` models its output `y` as

$$ y(x) = w_0 + \sum_i w_1i x_i$$

where `w` are the learnable weights of the model.

However, the interactions between the input variables `x_i` are purely additive. In some cases, it might be useful to model the interactions between your variables, e.g., `x_i * x_j`. You could add terms into your model like


$$ y(x) = w_0 + \sum_i w_1i x_i + \sum_i \sum_j w_2ij x_i x_j$$

However, this introduces a large number of `w2` variables. Specifically, there are `O(n^2)` parameters introduced in this formulation, one for each interaction pair. A factorization machine approximates `w2` using low dimensional factors, i.e.,

$$ y(x) = w_0 + \sum_i w_1i x_i + \sum_i \sum_j <vi,vj> x_i x_j$$

where each `v_i` is a low-dimensional vector. This is the forward pass of a second order factorization machine. This low-rank re-formulation has reduced the number of additional parameters for the factorization machine to `O(k*n)`. Magically, the forward (and backward) pass can be reformulated so that it can be computed in `O(k*n)`, rather than the naive `O(k*n^2)` formulation above.

- Kernel trick is used in SVMs to define a convenient similarity function that is the result of a dot product in a very high dimensional space. Matrix factorization models assume that the large data matrix are the higher dimensional mappings of a lower dimensional space and try to find the lower dimensional representation.

- Factorization machines use ideas from both SVM and matrix factorizations. They use the lower dimensional space idea of matrix factorization as a space where relationships among features are encoded. Then they use the kernel trick of SVMs by saying that some engineered features are the similarity kernels of that lower dimensional space. 

- For example, they use feature engineering intuitions e.g. that features are more important in pairs (i.e. similar to a polynomial-2 kernel in SVM). This feature engineering will produce an insanely large feature space. The key idea is that they now define the weights of feature pairs as the similarity kernel function $$w_ij = <v_i,v_j>$$ and assume that the $$v_i$$ space is a lower dimensional space that can be found with matrix factorization.

- FMs are general predictor working with any real valued feature vector. 
    + In contrast to SVMs, FMs model all interactions between variables using factorized parameters. Thus they are able to estimate interactions even in problems with huge sparsity (like recommender systems) where SVMs fail. 
    + In contrast to matrix factorization models, the drawback of matrix factorization models is that they are not applicable for general prediction tasks but work only with special input data. Furthermore their model equations and optimization algorithms are derived individually for each task. We show that FMs can mimic these models just by specifying the input data (i.e. the feature vectors).