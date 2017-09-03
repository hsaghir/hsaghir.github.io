---
layout: article
title: Machine learning basics Deep Learning book Ch4
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

### ML
- Statistics focuses on finding causal relationships or estimating confidence intervals of functions (usually simple / linear models). Machine learning focuses more on complex function estimation and prediction. Bayesian ML focuses on complex function estimations along with confidence estimations.  

- Learning is defined as using data points to improve on a task as measured by a performance measure. Designing a performance measure (loss function) is an art and case specific. Supervised Learning [Classification, Regression, Structured Output prediction] involves conditional probability estimation of outputs from inputs P(Y|X). Unsipervised learning [e.g. generative modelling, denoising, imputation, clustering, etc] involves a Density estimation i.e. learning P(X). In a modelling effort we might introduce model variables Z and try to do a density estimation on P(X,Z) instead. This involves finding a function from (N dimensional) data space to a probability surface in (N+1 dimension). Semi-Supervised learning is a combination of supervised and unsupervised data while multi-instance learning is the case where individual samples don't have a label but we know a certain type of sample exists among a group.

- Using the chain rule of probabilities we can transform an unsupervised learning problem to many supervised learning problems $$P(X)=\prod_i P(x_i|x_1, x_2, .. x_{i-1})$$. In the same sense, we can solve a supervised learning problem using the Bayes theorem by estimating $$P(Y|X)=P(X,Y)/P(X)$$. 

### Bias, Variance and the Bayes error
- In machine Learning we minimize training error but we actually care about minimizing test error that we can't directly access. The least amount of error possible incurred by a perfect model that knows the data generation distribution is called Bayes error. If the data generating distribution for the training and test sets are different, we will have high error rate for our test set and we can't do much but to get more data that is more similar in data generation distribution to our test set. 

However, If we assume that both train and test sets are iid samples (independantly sampled from same distribution) of one data generating distribution, then the expectation of error for a fixed model would be exactly the same for both train and test samples. However, in the learning process, we sample the data generating distribution (batch) and modify model parameters to reduce training error. Therefore, the expectation of the error for samples drawn for training would be smaller (or equal) than the expectation of error for samples drawn for test (no learning). 

- A well-performing machine learning algorithm will have 1) low training error (bias) and 2) small gap between train and test errors (variance). These two correspond to model capacity, a simple model might underfit and have large bias while a complex model might overfit and have large variance. The recipe is that we usually increase model complexity if the bias is high to better fit the data. If the variance is high, we try applying regularization techniques to reduce overfitting. 

- Point estimate finds single set of values for model parameters. For example, sample mean/std are point estimates for population mean/std under a Gaussian model. Point estimate from samples (as opposed to population) have error which are quantified by bias and variance of the estimator. Bias is $bias(param)=(E[sample param] - population param)$. For example, if we assume a Gaussian distribution model, sample mean estimator is unbiased estimate of real mean but sample variance estimator underestimates population variance. Consistency is when the bias converges to zero with increasing amount of data. 

- Variance of a point estimator is simply the variance of the estimate over data. Its square root is called standard error which provides a measure of how much variation in our point estimate we expect as we calculate the estimate on a different sample. Standard error for sample mean point estimator is $SE(sample mean)= (population std) / sqrt(number of samples)$. We don't know population std so we use sample std which we know is still biased, but if the number of samples is large it's reasonable. According to central limit theorem, which says the mean will be approximately distributed with a normal distribution, we can calculate 95% confidence interval using mean and standard error (95%CI= mean +- 1.96*SE).

- There is a trade-off between bias and variance of an estimator, therefore, we might use a biased point estimator in cases where we are interested in low estimator variance. Cross-validation helps with deciding about the tradeoff. An example of where this concept might be useful is batch normalization in deep nets. The statistics of each batch is slightly different from another and therefore as it propagates through a deep net, it changes drastically. This makes learning difficult and batch normalization solves it by normalizing each batch at each layer using sample mean and variance point estimates (low estimator variance is desired in training). However, since sample variance is biased, in testing where we care about low bias in our prediction we use the unbiased sample variance point estimator instead.

- The no free lunch theorem says if we average over all possible data generation distributions, all machine learning algorithms will have the same test error rate. This means that there is no best universal algorithm for all data sets (data generation distributions). We should therefore focus on finding data generation distributions we encounter in the real world, and the machine learning algorithms that work well for them. 

- Hyperparameters are parameters that are set but not learned e.g. learning rate. We therefore, divide the training set into training and validation sets to be able to set hyperparameters for the learning algorithm for example adaptive gradient descent (adagrad). Dividing the data to fixed test and training doesn't give much certainty about generalization if test set is small so we do k-fold cross validation (divide data to k segments and repeat learning k times keeping a different segment for test each time) to get mean and variance of generalization error.


### Maximum Likelihood
- Maximum likelihood principle is a popular point estimator which states that model parameters should be chosen that maximize the probability of data. Assuming independant samples, that's $$max(\prod p_model(x_i))$$. We take log to convert the product to sum and help with numerical underflow i.e. max [sum(log p_model(x_i))]. By dividing by constant m number of data, we can write the sum as the expectation of model distribution $max(E_{p_em}[log p_model(x)])$ over emprical distribution of data p_em(x). This way we can interpret maximum likelihood as minimizing the KL divergence between the model distribution p_model(x) and the emprical distribution p_em(x) i.e. KL(p_model/p_em)=E_{p_em}[log p_em(x) - log p_model(x)]. Any loss function consisting of a negative log model probability (likelihood) is a cross-entropy between the model likelihood and emprical distribution defined by the training set. Maximum likelihood is the best estimator assymptotically as the number of data tends to infinity if the true data distribution is in the model distribution family.

- In the supervised learning case, we write $$p_model(y|x)$$ and follow the same procedure for maximum likelihood estimation. For example, in linear regression we assume a Gaussian model $$p(y|x)=N(wx, \sigma^2)$$ with a fixed variance and a linear function of mean. Performing maximum liklihood on this model results in optimizing mean squared error. When the number of examples is small in the model, we will overfit and have a large gap between train (sample) and test (population) errors (large variance). Therefore, we trade off a biased version of maximum likelihood (i.e. add regularization) for less variance in the model. 

### Bayesian inference
- As opposed to point estimates for model parameters in frequentist stats, Bayesian approach views model parameters as random variables instead of data points. Therefore, it assumes a distribution for model parameters which is first defined by priors (usually a high entropy distribution like uniform or Gaussian), then data are plugged in and decrease the entropy by concentrating around likely values which establish the posterior belief. The Bayesian approach has two important difference; 1) every prediction will also be a probability distribution (belief) reflecting the model uncertainty about it (robust against overfitting) 2) Another difference is in the application of priors which shift the starting probability space to regions prefered in the beginning. In practice priors are often used to convey preference for models that are simple or smooth. 

- As an example, consider Bayesian linear regression. We consider multivariate Gaussian model $p(y/x)=N(wx, I)$. Assuming an iid Gaussian prior on weights, we can use the Bayesian formula to do Bayesian inference. Assuming a zero mean and alpha scaling for covariance on priors, we get the same result as maximum likelihood linear regression with a regularization of {alpha * w^Tw}. 

- MAP in an alternative point estimate to maximum likelihood that allows priors to influence parameters choice. MAP estimate states that one should choose parameters that maximize the posterior i.e. $max(p(param/x)=p(x/param)p(x))$. Applying log to convert product to sum and prevent underflow we get $max(log p(x/param)+ log p(x))$. The first term is the maximum likelihood term while the second term corresponds to prior distribution. MAP with a Gaussian prior on weights corresponds to maximum likelihood with weight decay regularization.

### GLM
- A generalized linear model (GLM) can be defined as $$Y_out=WX$$ where X is input data and w are parameters. Setting the gradient of quadratic cost for this model to zero to find optimum parameters that minimize quadratic cost will return least square solution analytically as $$W= ((X X^T)^-1)(X^T)(Y_target)$ called normal equation. We can make polynomial or arbitrary nonlinear models of X by applying arbitrary nonlinear functions to X and extending the input matrix and corresponding number of variables $$W$$. However, since the model is still linear wrt to parameters $$W$$, it's still a linear model (GLM) and the least square solution is given by normal equation although it can model nonlinearity ! We might want to add L1 or L2 regularization terms to the squared error in which case we can find parameters by SGD instead of normal equation.

### Logistic regression
- We can perform classification using the linear regression approach by using a logistic sigmoid on the output of linear regression $$\sigmoid(wx+b)$$ to squash it into [0,1] interval which can be interpreted as probability of either class 0 or 1. Optimizing the maximum likelihood (negative log likelihood) loss is done via SGD.

### Cross-entropy and Softmax
- A popular cost function in classification problems is cross-entropy loss coupled with softmax function. Softmax function transforms the outputs of the model into numbers in [0,1] in a way that they add up to 1. This way the outputs can be interpreted as a probability distribution. The cross-entropy loss measures the difference between two probability distributions. It is now clear that if we can use softmax to interpret multiple outputs as a probability distribution, we should use cross entropy as a measure of difference between model output and target values.

- Another way to interpret cross-entropy is to see it as negative log-likelihood for the data target $$y_i^'$$, under a model $$y_i$$. i.e. $$-log p(data|model)$$

$$H_{y^'} (y) = \sum_i (y_i^')\log(y_i)$$

Due to the $$\log$$ for the model output, if the model assigns zero probability to a class while the real probability is non-zero. The model is doing very bad and the distance between the model distribution and the real distribution is infinity $$\log(0)$$.

To avoid this problem people use sigmoid or "softmax" functions at the output of the model so as to leave at least some chance for every option.

$$softmax(x) = \frac{\exp(x_i)}{\sum_i \exp(x_i)}$$

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


### K-NN
- Another type of supervised (and unsupervised) learning algorithms is K-nearest neighbors which basically doesn't do any learning. It just remembers all the points and interpolates between the opinion of k-nearest neighbors of each new point at test time. However, it can't understand patterns in a certain feature of the data. For example, if only one feature of a 10 dimensional input is relevant, it still will find the nearest neighbors based on all 10 dimensions. Therefore, it's not very well suited to small datasets. 

### Decision Trees
- Decision trees hierarchically partition the input space with partitions that are alligned along the input feature axes. For example, if we have 3d data, the first level nodes might partition the input space along x1 and x2 while the second level nodes partition along all three axes. Their problem is that if the decision boundary is not axis alligned (e.g. a linear non-perpendicular boundary), they will have problem and will have to do many partitions to zig-zag around the line! Their learning can be non-parametric if the tree size is not constrained but with regularization, they will have a finite set of parameters. 

### Unsupervised learning
- An unsupervised learning task is finding simpler representations by finding lower-dimensional, sparse, or independant representations. For example, PCA transforms the data into a representation that linearly disentangles unknown factors of variation underlying the data (each eigenvector is linearly uncorrelated with anotherss). We are also interested in represenations that disentangle nonlinear dependencies.

- Another unsupervised learning task is clustering where we assign every example to a cluster. K-means for example performs two step optimization to do clustering by first assigning the examples to random clusters based on their distance and in the second step refine the clusters by calculating the centroids of those clusters. This continues until convergence (EM algo). We might use clusters with a one-hot vector as a representation but that's not the best distributed representation since that doesn't convey enough information about similarity of two representations. 

### Learning 
- Gradient descent needs the expectation (sum) of gradients for all data points at every update which is costly O(m). Stochastic Gradient Descent (SGD) notes that the expectation can be approximated by a sample (minibatch) from the dataset O(1). 

- Prior to deep learning the main way of learning nonlinear models was to use the kernel trick with a linear model. This required constructing an mxm matrix K(x_i,x_j) which is not scalable O(m^2); but deep learning with SGD scales very well (amorized cost of O(1)).

- Remember that all machine learning algorithms consist of several parts; i.e. dataset, model, cost function, and optimization algo that are relatively independant of each other and can be combined to make new machine learning algorithms. 

- There are some problems with traditional machine learning that DL tries to solve. 1) Curse of dimensionality which means as the number of dimensions increase, the possible data configurations increase exponentially. Therefore, we encounter an statistical challenge as the number of examples in dataset would be much less than possible configs in high dimensions. 2) Another problem is the smoothness assumption of most traditional machine learning algos. This works well in low dimensions and when test data is in the same region as the training data where an interpolation can provide correct answer but face problem in cases of high dimensional data and when test data comes from another region of the input space. For example a checker board with alternating black and white color can be correctly predicted with traditional machine learning if the test data comes from the region where we have seen examples due to smoothness prior. However, extrapolation can not be done well. Deep learning solves these two problems by a different assumption. DL assumes that the data was generated by a composition of factors at multiple levels of a hierarchy. This assumption allows composing features at multiple scales and therefore, representing an exponential number of regions with not as many examples.

- A manifold is a set of connected points with lower dimension embedded in a higher dimensional space where each dimension corresponds to a local direction of variation (e.g. roads are 1d manifolds in 3d space). Usually one can traverse a manifold along these directions using transformations. There is evidence for existance of manifolds in AI tasks is that the probability distribution for many real images, sounds, etc are highly concentrated. Finding lower dimensional coordinate systems inside manifolds is desirable to be able to traverse them. 


