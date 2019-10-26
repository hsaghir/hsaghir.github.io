---
layout: article
title: A primer on neural network learning rules
comments: true
image:
  teaser: jupyter-main-logo.svg
---

Bayesian Optimization
Bayesian Optimization is an optimization algorithm that is used when three conditions are met: (1) you have a black box that evaluates the objective function (2) you do not have a black box (or any box) that evaluates the gradient of your objective function (3) your objective function is expensive to evaluate. Expensive could be in time or money or any other resource that you care about. Because it is expensive to evaluate the objective function, it would be a big waste of resources to numerically estimate the gradient and then use some sort of gradient algorithm. A key observation is that if it is so expensive to evaluate the objective function, then it is reasonable to put substantial effort into deciding where to evaluate the objective function next, since you want few but very informative evaluations. This is done by using a Gaussian Process (GP) to model the (unknown) objective function. The GP has a mean and variance at every point. Intuitively, if the mean is low, you may be near a minimum, which is interesting, so you want to evaluate more in that vicinity. On the other hand, if the variance is very high, you really do not know much about the objective function in that area, so you want to evaluate more. This is the standard exploitation vs. exploration tradeoff (the former and the latter, respectively). There is something called Expected Improvement which quantifies one particular tradeoff between these two competing incentives. Bayesian Optimization works by solving an easier optimization problem to decide where to evaluate the objective function next (namely, find the location that maximizes the expected improvement). Again, solving this easier optimization problem is presumably still much cheaper than actually evaluating the objective function, so it is worth the effort (I think because its objective function is not expensive to evaluate). The fact that Bayesian Optimization explicitly models the exploration vs. exploitation tradeoff prevents it from getting stuck in local minima, since it will eventually get "bored" of a local minimum once the uncertainty is very low in that region.

Capacity, Complexity (of a model class)
I believe these two terms are not the same but I do not know the difference between them. To the best of my knowledge they refer to the "size" of a class of models. For example, the class of cubic polynomials is strictly "bigger" than the class of quadratic polynomials, because all quadratic functions are also cubics, but there are many (infinitely) more cubics that are not quadratic. VC dimension is a way of quantifying this.

Curse of Dimensionality
This generally refers to the situation when you have some_function_that_you_want_to_be_small = xdxd, where dd is the dimension of the space. You may notice that if x>1x>1, some_function_that_you_want_to_be_small will not be small for large dd. For example, if you want to a systematic search over a space of parameters, this becomes increasingly impossible as the dimension of the space increases. There are many examples. One important intuition at high dimension is that if you have a hypersphere, most of the hypersphere's volume is contained in a very thin shell at the boundary. This has various implications for example in rejection sampling at high dimension.

Deep Belief Network (DBN)
coming soon (submit your own definition here)

Deep Learning
The spirit of deep learning is to learn multiple layers of abstraction for some data. For example, to recognize images, we might want to first examine the pixels and recognize edges; then examine the edges and recognize contours; examine the contours to find shapes; the shapes to find objects; and so on. In practice, deep learning refers to models that have multiple layers. For example, neural networks or deep belief nets with many layers could fall under the realm of deep learning. The phrase "deep learning" is moreso a brand name for a class of methods than it is a concept.

Density Estimation
The problem of, given a data set, estimating the probability distribution from which these data presumably came.

Directed Graphical Model
A directed graphical model is a type of graphical model that uses an undirected graph to represent a probability distribution. In particular, a directed edge from X to Y indicates that Y may depend causally on X (note though that a distribution in which X and Y are independent is still consistent with the graphical representation; the graph only indicates that there may be dependence. However, for a graph in which X and Y are completely disconnected, a distribution in which X and Y depend on each other would not be consistent with the graph). The causal interpretation of the directed edge is somewhat abstract. For example, consider a graph with two nodes, X and Y, with an edge from X to Y. Then, imagine X comes from a fair coin flip, and then Y is set to the opposite of X. In the description, Y depends causally on X, but the implied joint probability distribution simply says that with probability one half X is 1 and Y is 0 and with probability one half X is 0 and Y is 1; that is, it makes no distinction between X and Y. This means that the directed graphical model with X pointing to Y is the same as the directed graphical model with Y pointing to X. However, this does not mean that any directed graphical model is equivalent to the same graph with all its arrows switched around. Consider for example the graph in which there are three nodes (random variables) X, Y, and Z, with X pointing to Y and Z. This graph encodes the fact that Y and Z are conditionally independent, given X. That is distinct from the flipped graph (with Y and Z pointing to X), which say that Y and Z are independent (but in fact become conditionally dependent, given X).

Discriminative, Generative
Discriminative models learn to discriminate between different inputs. For example, classifying images as containing a dog or not containing a dog is a discriminative task. An example of a discriminative model is a support-vector machine. Generative models usually involve probabilities and their distinguishing feature is that you can generate new data from them. For example, if you tried to estimate the probability distribution over images containing dogs and a different distribution over images not containing dogs, you would have generatively modeled the situation. You could use these distributions to sample new images of dogs (or new images not containing dogs). If you wanted to use this generative model for a discriminative task, you could: given an image, you could see which of the two distributions assigns higher probability to that image and then choose that as your result. Thus, there is a distinction here between discriminative model and discriminative task: it may be possible to use generative models for discriminative tasks.

Expectation Maximization (a.k.a. The EM Algorithm)
coming soon (submit your own definition here)

Frequentist, Bayesian
The Bayesian view is essentially that everything should be done with Bayes' rule: computing posterior probabilities by multiplying priors with likelihoods. In a Bayesian approach, you usually have a posterior distribution over models. Then, if you want to use this model for something, like making a prediction, you integrate over your posterior distribution of models to get a sort of "expected value" of the thing you are trying to predict. Frequentist is often used to mean not Bayesian. In a frequentist approach, you typically find a "best" solution (i.e., model) to the problem you are trying to solve. You then use this best model to make the prediction. I believe there is a relationship between the frequentist approach and discriminative models, and likewise for the Bayesian approach and generative models; but I think the correspondence is not exact. Please email me if you have more ideas about this.

Probabilistic Graphical Model (a.k.a. Graphical Model)
Probabilistic Graphical Models are ways of encoding the structure (independencies) of a probability distribution into a picture. The two main types of graphical models are directed graphical models and undirected graphical models (probability distributions represented by directed and undirected graphs respectively). The general idea is that each node in the graph represents a random variable, and a connection between two nodes indicates a possible dependence between the random variables. So, for example, a fully disconnected graph would represent a fully independent set of random variables, meaning the distribution could be fully factored as P(x,y,z,...)=P(x)P(y)P(z)...P(x,y,z,...)=P(x)P(y)P(z)... Note that the graphs represent structures, not probabilities themselves. To fully specify a probability distribution some additional information is needed besides just the graph.

Importance Sampling
coming soon (submit your own definition here)

Kernel Trick
The kernel trick is a mathematical observation. Let's say you have two functions f(x)f(x) and g(x)g(x). If you observe that the equations in front of you only involve the dot products f(x)⋅g(y)f(x)⋅g(y) and never f(x)f(x) or g(y)g(y) individually then you can use the kernel trick. Instead of explicitly defining f(x)f(x) and g(y)g(y), you could just define the kernel K(x,y)K(x,y) which is defined as f(x)⋅g(y)f(x)⋅g(y). In the case where you have a finite data set, you only need a Kernel Matrix to define all possible dot products that might arise. Or, you could define a Kernel Function k(x,y)k(x,y) that is defined for all values of xx and yy, not just a finite set. In Support-Vector Machines you might see something called a "radial basis function". This just means using the kernel trick where the kernel function is a Gaussian.

Lasso Regression, Ridge Regression
Lasso regression means regression with L1 regularization. Ridge regression means regression with L2 regularization.

Learning, Inference
Learning refers to learning unknown parameters in your model. Your model might be wrong, so these parameters might have nothing to do with the real world -- but they can still be useful in modeling your system. Inference refers inferring unknown properties of the real world. This can be confusing both because learning and inference refer to figuring out unknown things. An example from human vision: when you see something, your brain receives a bunch of pixels and you need to infer what it is that you are seeing (e.g., a tree). This happens on a very short time scale. On a longer time scale, your brain is also learning -- after seeing several trees of a new sort, some connections are strengthened in the brain and you now have a better model of the world that allows you to better infer what you are seeing in the future.

L1 norm, L2 norm, etc.
The L* norms are ways of measuring the length of a vector. The L2 norm of a vector is the standard Euclidean length: the square root of the sum of the squares of the entries. This can be written more generally as the p-th root of the sum of xpxp, where the x's are the entires. Then p=2p=2 gives the L2 norm. p=1p=1 gives the L1 norm (the sum of the absolute values of the entires). The L0 norm is just the number of nonzero entires in the vector. The infinity norm is the size of the element with biggest absolute value.

Machine Learning
Machine learning is about the theory and practice of learning from data. Data often contain useful information and insight that is not readily apparent when "looking" at it. Machine learning attempts to extract this information and insight.

Maximum A Posteriori (a.k.a. MAP)
coming soon (submit your own definition here)

Maximum Likelihood
coming soon (submit your own definition here)

Markov Chain Monte Carlo (MCMC)
MCMC is used when you want to sample from a probability distribution that (a) you do not have in closed form but (b) you can evaluate it ''up to a normalizing constant''. This means that you have a black box than can compute kP(x)kP(x), for some number kk, but you do not know kk. In other words, you can only compute the unnormalized distribution. MCMC allows you to generate samples from the distribution using a Markov Chain.

Metropolis Hastings
coming soon (submit your own definition here)

Neural Network (a.k.a. Neural Net)
Neural networks are a specific class of functions y=g(x)y=g(x). One way to define them is recursively: g()g() is a neural network if it is a repeated application of the transformation f(Wx)f(Wx) where WW is a (not necessarily square) matrix and f()f() is a scalar function that is applied element-wise to the elements of the vector WxWx. Note that the matrices WiWi are constrained in their shape -- the shapes must be such that the product W1⋅W2⋅...WNW1⋅W2⋅...WN actually makes sense. The function f()f() can be any nonlinear function but is commonly the hyperbolic tangent function or the sigmoid function 1/(1+exp(−t))1/(1+exp⁡(−t)). It is usually the same at every step (also known as layer), unlike the W matrices which are not the same at every step. Neural networks are used as discriminative models in supervised learning tasks. They are popular because the learning phase (learning the W matrices) is easily done using the backpropagation algorithm, which is actually just a fancy name for using the chain rule in this context. Some intuition about neural networks: the reason we need these nonlinearities f() is that if we just applied a lot of matrices to an input, this would be the same as just applying one matrix multiplication, since the composition of a bunch of linear transformations is still just a linear function. By adding in the nonlinearity, the more layers the network has, the more complicated the functions it represents. There is some other confusing jargon here: 'unit' and 'neuron'. They both mean the same thing, namely the intermediate values that the input takes as it is processed towards the output. For example, if the input x is 10 dimensional, we say there are 10 input units. Then if the first W is a 20x10 matrix, the next intermediate representation is a vector of size 20, so we say there are 20 "hidden units" there. And so on. This is more clear if you see the graphical representation of a neural network. See, e.g., this image.

Occam's Razor
The idea that the simplest explanation tends to be the best explanation.

Overfitting
This is when you learn something too specific to your data set and therefore fail to generalize well to unseen data. For example, if you are doing regression with 20 data points and you use a 19 degree polynomial (which has 20 free parameters), you have a 20x20 linear system which you can solve. The resulting polynomial will pass through all 20 points but it will probably be "overfitting" because it is curvier than it should be and may behave very crazy between the data points. This causes it to do badly for unseen data. Another example is if you are a student and you memorize the answers to all 1000 questions in your textbook without understanding the material. Then, you will do amazingly well when doing problems from your text book (a.k.a. your training set) but you will fail miserably when given new problems (you are failing to ''generalize''). Our weapon against overfitting is Occam's Razor, encoded as regularization. There is something called Bayesian Occam's Razor which indicates that Bayesian methods suffer less from overfitting.

Parametric, Nonparametric
A parametric model is a model with a set of parameters. Each setting of the parameters corresponds to a different model. For example, a cubic polynomial is a parametric model: it has 4 parameters. While the cubic polynomial is used to model one-dimensional data (y as a function of x, where x and y are scalars), the space of cubic models is 4 dimensional, because of the 4 parameters. The learning of parametric models often takes the form of an optimization problem: minimize some loss function over the space of models. Nonparametric models are models whose complexity grow with the amount of data. It is often said that "nonparametric" is a misnomer and "infinitely parametric" is closer to the truth. Some examples of nonparametric models are the Chinese Restaurant Process, Indian Buffet Process, Gaussian Process. Conjecture: all nonparametric models end in the word "process".

Restricted Boltzmann Machine (RBM)
coming soon (submit your own definition here)

Regression
The supervised learning task of fitting a function to a set of data points.

Regularization
Regularization is a way of applying Occam's Razor to real problems. Imagine you are fitting a polynomial to some data by minimizing squared error (the sum of the squares of the differences between data and the predictions). As discussed above, if we use a degree 20 polynomial (21 parameters), we might overfit (especially if we only have 20 or so data points), because our model is too complicated. One way of address this is just using a less powerful model, for example a degree 3 polynomial. Another method is regularization. We regularize the problem by adding an extra term to the optimization problem: instead of minimizing the L2 prediction error, we minimize the L2 error of the model plus the L1 norm of the _model itself_. This should sound crazy -- it is. Here is the intuition for why this comes about: the proposed solution of using a cubic polynomial instead of a degree 20 polynomial is too restrictive -- perhaps we need a quartic term. So, instead, let's use a degree 20 polynomial but only allow 4 of the 21 coefficients to be nonzero (this is called looking for a "sparse" solution, because the parameter vector is sparse -- it contains mostly zeros). Then, if indeed a cubic polynomial is the best solution, the algorithm will find this solution. But, if it is not, the algorithm has the freedom to choose a different set of 4 terms (out of 21). (In other words, we have reduced the space of models to be something smaller than the class of degree 21 polynomials but larger than the class of degree 3 polynomials, hoping that we got it just right.) To implement this idea, our new optimization problem is: minimize the L2 error, subject to the constraint that at most 4 of these parameters are nonzero (i.e., the L0 norm of the parameter vector is less than or equal to 4). Following Lagrange Multiplier intuition, we can transform the constraint into another term of the objective function (this may be called "dualizing" the constraint). Now, we take a leap of faith and say: well, we have some extra term in the objective function. It probably should contain the L0 norm of the parameter vector, but the L0 norm is messy and not smooth or differentiable (and smoothness/differentiability is important for optimization). So, let's relax this and use the L1 norm of the parameter vector instead (or L2) -- L1 is mostly differentiable and it should give us something reasonable. Then we do some retroactive justification: well, minimizing the L1 norm of the parameter vector means that the optimization will try to make the parameters small, which is pretty much what we wanted anyway. The sad part here is that by doing this we have obfuscated our assumptions. Our new L1 (or L2) regularization does correspond to some assumptions -- everything does -- but the assumptions are not clear anymore, they are not something nice like "at most 4 of the parameters can be nonzero". (But maybe this is not sad. After all, the initial assumption that "at most 4 parameters should be nonzero" was completely arbitrary: it was just a way of reducing the model space that was easy to describe in words. Perhaps there is no reason to believe the new assumptions are any worse.) This issue of hidden assumptions is where the Bayesian view really shines. As it turns out, this regularization has a very clear interpretation in the Bayesian sense -- it corresponds to a particular prior (Laplace for L1; Gaussian for L2) on the parameter vector. I like this because, even if the assumptions are arbitrary, I like to know what they are.

Rejection Sampling
coming soon (submit your own definition here)

Precision Recall Curve, ROC Curve
coming soon (submit your own definition here)

Sampling
Sampling refers to generating samples from a given probability distribution. This is typically achieved using pseudorandom numbers generated by a computer. While this is easy for certain distributions like the uniform or binomial distributions, it can hard for an arbitrary distribution. Basic sampling methods include rejection sampling and importance sampling. For the even more challenging case when the distribution in question is only known up to a constant factor, see MCMC.

Supervised learning, Unsupervised learning
Supervised learning is learning a map y=f(x)y=f(x) given example (x,y)(x,y) pairs. The goal is to predict the yy-values for unseen xx-values. An example is a Support-Vector Machine. Unsupervised learning is less well-defined. It is about learning structure from data. An example is density estimation. For a more philosophical discussion, see Roger Grosse's blog post on this topic.

Support-Vector Machine (SVM)
A discriminative model for 2-class classification. And SVM learns a hyperplane through the input space, thus dividing the space in two. To make predictions with an SVM, you just need to check which side of the hyperplane your input is. The SVM tries to find the maximin hyperplane. This means that it seeks a hyperplane that maximizes the minimum distance between the data and the hyperplane. This is also called "max margin" in order to be confusing. The simplest SVM is the linear SVM, which just learns a hyperplane. Fancier SVMs use the Kernel Trick to boost the data into higher dimensions and learn a hyperplane in this higher dimensional space. In the original space, this corresponds to learning a separating boundary that has a more complicated shape than just a simple hyperplane. An interesting feature of SVMs is that typically only a small number of data points actually affect the solution (because of the nature of maximin problems). These points are called "support vectors". In the dual representation, only the dual variables corresponding to these support vectors are nonzero (because only these constraints are active), so we say that the SVM is sparse in the dual representation.

Training Set, Test Set, Validation Set
One way to avoid overfitting is to split up your data into different (disjoint) sets. If you train on the training set and test on the evaluation set, then presumably you won't suffer from overfitting. The test set is something that should, in theory, only be used once, for testing. If it is used more than once it becomes compromised and you end up training on the test set, which is bad.

Undirected Graphical Model
coming soon (submit your own definition here)

Fisher information
This is how Fisher information is generally presented in machine learning textbooks. But I would choose a different starting point: Fisher information is the second derivative of KL divergence.

    \begin{align*} {\bf F}_\theta &= \nabla^2_{\theta^\prime} {\rm D}(\theta^\prime \| \theta)|_{\theta^\prime=\theta} \\ &= \nabla^2_{\theta^\prime} {\rm D}(\theta \| \theta^\prime)|_{\theta^\prime=\theta} \\ \end{align*}

(Yes, you read that correctly — both directions of KL divergence have the same second derivative at the point where the distributions match, so locally, KL divergence is approximately symmetric.) In other words, by taking the second-order Taylor expansion, we can approximate the KL divergence between two nearby distributions with parameters \theta and \theta^\prime in terms of Fisher information:

    \[ {\rm D}(\theta^\prime \| \theta) \approx \frac{1}{2} (\theta^\prime - \theta)^T {\bf F}_\theta (\theta^\prime - \theta). \]

Since KL divergence is roughly analogous to a distance measure between distributions, this means Fisher information serves as a local distance metric between distributions. It tells you, in effect, how much you change the distribution if you move the parameters a little bit in a given direction.

Exponential Families and Maximum Entropy
An exponential family parametrized by \boldsymbol\theta \in \mathbb R^d is the set of probability distributions that can be expressed as

    \[ p({\bf x} \,|\, \boldsymbol\theta) =\frac{1}{Z(\boldsymbol\theta)} h({\bf x}) \exp\left( \boldsymbol\theta^{\mathsf T}\boldsymbol\phi({\bf x}) \right) ,\]

for given functions Z(\boldsymbol\theta)} (the partition function), h({\bf x}), and \boldsymbol\phi({\bf x}) (the vector of sufficient statistics). Exponential families can be discrete or continuous, and examples include Gaussian distributions, Poisson distributions, and gamma distributions. Exponential families have a number of desirable properties. For instance, they have conjugate priors and they can summarize arbitrary amounts of data using a fixed-size vector of sufficient statistics. But in addition to their convenience, their use is theoretically justified.

Suppose we would like to find a particular probability distribution p({\bf x}). All we know about p({\bf x}) is that \mathbb E_{p({\bf x})}[f_k] = c_k for a specific collection of functions f_1, \ldots, f_K. Which distribution should we use? The principle of maximum entropy asserts that when choosing among all valid distributions, we should choose the one with maximum entropy. In other words, we should choose the distribution which has the greatest uncertainty consistent with the constraints.



The Natural Gradient
ommon activity in statistics and machine learning is optimization. For instance, finding maximum likelihood and maximum a posteriori estimates require maximizing the likilihood function and posterior distribution respectively. Another example, and the motivating example for this post, is using variational inference to approximate a posterior distribution. Suppose we are interested in a posterior distribution, p, that we cannot compute analytically. We will approximate p with the variational distribution q(\phi) that is parameterized by the variational parameters \phi. Variational inference then proceeds to minimize the KL divergence from q to p, KL(q||p).

When G is non-trivial the vector G(\phi)^{-1}\nabla L(\phi) is called the natural gradient and as we have just seen is the direction that L increases most quickly when the parameter space is a manifold with Riemannian metric G. This tells us that a more efficient variational inference algorithm is to follow the natural gradient of the variational parameters, where the Riemannian metric tensor is just the Fisher information matrix of the variational distribution.

As a concrete example, imagine two people standing on two different mountain tops. If one of the people is Superman (or anybody else who can fly) then the distance they would fly directly to the other person is the Euclidean distance (in R^3). If both people were normal and needed to walk on the surface of the Earth the Riemannian metric tensor tells us what this distance is from the Euclidean distance. Don’t take this illustration too seriously since technically all of this needs to take place in a differential patch (a small rectangle whose side lengths go to 0). Intuitively, the Riemannian metric tensor describes how the geometry of a manifold affects a differential patch, d\phi, at the point \phi. The length of a line between two points on d\phi is the distance between them. The Riemannian metric tensor either stretches or shrinks that line and the resulting length is the distance between the two points on the manifold.

