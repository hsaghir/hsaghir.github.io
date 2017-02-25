---
layout: article
title: Regularization Deep Learning book Ch7
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

- Strategies designed for reducing test error even at the expense of training error are called regularization. Regularization involves trading off incresed estimator bias for reduced estimator variance. Some put additional constraints on the loss function (i.e. add a norm of parameters) or add terms to it to encode some form of prior knowledge on the energy/probability goo and some implement Occam's razor to encourage simpler models.  Other strategies involve combining multiple models in ensembles to increase generalizability.  


## Constraining model capacity
- We usually only regularize weights and leave biases alone to not induce too much variance on the estimator. A weight controls two variables while a bias controls only one so it requires less data for training. Regularizing bias can lead to significant under-fitting. 

- We can think of Regularization in terms of effects of priors in MAP inference. L2 norm is equivalent to using a Gaussian prior on weights while L1 norm is equivalent to an isotropic laplace distribution prior on weights. Alternatively, we can think of regularization in terms of constraining the space of possible weights using lagrange (KKT) multipliers in a constraint optimization problem. With L2 regularization, weights are constrained to remain inside an L2 ball while with L1, weights are constrained to remain inside within an L1 norm distance. Using constrained optimization view, we can use more complex constraints on the loss function explicitly instead of penalties and this can also prevent positive feedback loops that we might encounter with using penalties.


- L2 norm $\lambda norm(w)^2$ also known as weight decay, ridge regression, Tikhonov regularization. Imagine a quadratic loss as contours of ellipses with minimum in the middle that we want to get to. Since it's an ellipse, moving along one axis (a combination of certain weights corresponding to the direction of the eigenvalues of the Hessian) gets us to the minimum loss faster resulting in large values for those weights and small values for others. This means large variance (i.e. test error) in our estimator. We trade off some bias for reducing this variance using L2 norm. If we look at the gradient descent with weight decay term, will see that the weight vector will multiplicatively shrink by a percentage of itself at each step $w-\alpha*w$. Over time, this reduction penalizes bigger weights more than smaller weights resulting in a more balanced weight vector. 

- L1 regularization $\lambda norm(w)$, has a less smoothing effect on weights admitting some weights to be be bigger than other while some are close to zero. Therefore, L1 regularizer provides a sparse weight vector. It is used in combination with a linear model and a MSE cost for feature selection in LASSO due to sparsity it provides.

- Imagine a linear model like PCA. The dataset might have no variance in some eigen directions or simply not enough data in that direction which leads to a singular design data matrix. Now think about the quadratic cost contours explained in L2 norm above. Some directions decrease this cost very fast but will lead to high generalization error. If we add L2 regularization to it, we can balance the weight vector. Therefore, Inversing the singular data matrix which is done through the Moore-Penrose psuedo inverse $X^T*X + \alpha*I$, is equivalent to performing ridge regression/ weight decay on the linear model. Therefore, psuedo inverse stablizes the under-determined linear system using regularization!

## Data Augmentation
- one strategy to generate new data is to use existing (x,y) pairs to generate new data by applying transformations (that don't change label) in a classification task. Makes model transformation invariant. Very well suited to high dimentional data like images. Another strategy is to add noise to inputs or hidden representations to generate new data and make model robust to noise.

## Noise
- Adding noise to input is equivalent to imposing a norm regularization on weights. Another way of using noise for regularization is adding it directly to weights which is equivalent to stochastic implementation of Bayesian inference over weights. 

- If we add iid Gaussian noise to weights at each presentation of input, the minimization of the objective function will be the same as before with the addition of a new regularization term $E[(\grad_w{output})^2]$. This encourages the weights to go to minimum regions with flat surroundings where small perturbations of weights has little effects.

- most data sets have some mistakes in the labels. We can add noise to labels (label smoothing) by for example using a softmax output instead of a hard 0/1 output. 

## semi-supervised learning
- The idea is to learn a representaion for the input (using both supervised and unsupervised inputs) that can be beneficial in learning output (supervised) mapping. This can be implemented as learning representation in an unsupervised fashion (for example, PCA, AE, etc) and then passing features to a classifier for the supervised learning. Another approach is to learn a generative model (p(x)) that shares parameters with a discriminative model P(y/x). We can then trade-off the supervised criterion $-log(p(y/x))$ with the unsupervised one $-log(p(x))$. This way the generative model acts as a prior (regularization) about the supervised problem. 

## Multi-task learning
- Combining a few learning tasks (both supervised and unsupervised) to pool data from different tasks and get a better generalization. It is done using a hierarchical representation where early layers share feature on all data and deeper layers diverge to do different classification / generation tasks. It can be seen as imposing a soft constraints on parameters of the network especially shared ones. 

## Early stopping
- It is often the case where the training error keeps dimishing while validation error does not. This is a harbinger of over-fitting and we stop training if we see validation error not decreasing for some amount of time. Another way of doing this is to store parameters of the network every time validation error improves significantly and continue until end and choose the parameter set with least validation error. In order to use all data (i.e. validation set), one might use the early stopping parameters as initialization for a second round of training this time with all data but the same number of epochs that early stopping found. 

- A way to view early stopping is as an efficient way to set training epochs hyperparameter. Training epochs effectively determines the model capacity. 

- It can also be viewed as a form of soft regularization where the space of parameters are restricted to a constrained space around initial values since both number of epochs and learning rate are restricted assuming bounded gradients. In this sense early stopping regularization is equivalent to weight decay with the difference that early stopping automatically determines the amount of regularization needed. 

## parameters sharing / weight tying
- Another form of regularization. It's also a way of incorporating prior knowledge about data domain into parameters. Parameter sharing imposes hard constraints by forcing parameters to be the same, for example in convnets we impose the prior that the statistical properties are translation invariant. Another way is to use a regularization constraint on weights in the loss function. For example with imposing the prior knowlegde that the weights of two networks should be close (for example in semi-supervised learning) which we can explicitly impose by regularizing the weights through adding a quadratic weights distance to the loss function. 

## Sparse representations
- Another type of regularization is to penalize unit activations so that minimum number of units are active at a time (sparsity). L1 regularization will cause parameters sparsity. We might explicitly apply the sparsity regularizer (e.g. L1, t-student prior, KL divergence penality, or average value close to zero) on the hidden representation. Hard sparsity constraints are also possible by imposing that only a certain number of units or smaller can be active at a time using constraint optimization. 

## Ensemble methods (model averaging)
- Bagging (boosting aggregate) trains several models seperately and then combine by having them vote to increase generalization. The reason it works is that if models do not usually make the same mistakes are their errors are not perfectly correlated. In that case, the expected error of the ensemble decreases linearly with the number of models (with uncorrelated mistakes).

- The ensemble can be made using different types of models. Bagging uses the same model and loss function. It samples k datasets of the same size (with replacement) from the original dataset. Differences in hyperparameters, initialization, batching, datasets will lead to partially independant errors.

## Drop-out

- Drop-out is very flexible and computationally inexpensive and usually works better than other inexpensive regularizers like weight decay or sparsity. However, it can be combined with other regularizers. When the dataset is too big or too small regularization is less effective. Drop-out is equal to adaptive weight decay for linear models where the decay coefficient for each weight is determined by its variance.  

- Drop-out can be seen as an approximate bagging strategy with weight sharing for large models. In Drop-out, with each mini-batch, we randomly sample a mask of zeros (probability defined as a hyperparameter) to apply to the output of input and hidden units. This is similar to bagging with an exponential number of models amortized using a single network where models share parameters and each model is trained on a mini-batch. If each model is producing a probability distribution at the output, the output of bagging is the arithmetical mean of all these exponential number of models which is intractable. We can approximate it using Monte Carlo estimate (mean of output from about 20 models). In this context, the bagging drop-out approach is also called inference. 

- Weight scaling inference rule is a better approach than Monte Carlo approximation of ensemble output. We approximate the ensemble with geometric mean instead of arthimetic mean. A key insight is that this geometric mean can be approximated with the output of the model with all units included and each weight multiplied by its probability of inclusion. Since we usually use 0.5 probability for hidden units, we can simply halve (1/2) the weights at the end of training and use the model. Note that this geometrical mean needs to be normalized to be a probability distribution.

- Multiplying weights by a noise $N(1,I)$ instead of zeroing out weights is an alternative that doesn't need weight scaling and has been shown to work well. Note that such noises are multiplicative (opposed to additive) since they appear in multiple layers, therefore, the network can't just take the easy way out of noise by making parameters bigger to damp the effect of noise. This way, noise keeps destroying learned information and leads to robustness. For example, if one neuron learns nose feature to distinguish a face, dropping it will force other neurons to learn other features (or redundant nose feature) to distinguish the face resulting in robustness.

## Adversarial training
- NNs can get very close to human level error rate in some cases, but they fail miserably on some carefully designed "adversarial" examples. These examples can be found using optimization by searching for a data point $x_prime$ close to point $x$, where, the output of the network for the two are very different. Training the network on such adversaial examples encourages the network to be locally constant in the neighborhood of the data manifold.

- Goodfellow found that one reason for adversarial examples are the extreme linearity and high dimentional nature of deep networks. Since there are many inputs and the function is linear, even small perturbations on all inputs (e.g. adding noise) can lead to a very different output. Interestingly, these adversarial examples are usually not even distinguishable by humans but even NN with very low error rate fail on them! 

## Tangent distance and tangent prop
- Manifold hypothesis in machine learning states that data is located on a lower dimentional manifold in a high dimentional space. Tangent distance makes this hypothesis explicit by requiring that the distance measure for data points be the distance of the tangent planes of their respective manifolds instead of euclidean distance. Tangent prop algorithm encourages the network function to have small changes in the direction of the manifold but large changes in the direction of gradient.

- This is done by having the gradient of net outputs be orthogonal to manifold tangent vectors (parallel to the direction of manifold). This is added as a regularization term (dot product of gradients and manifold tangents). Tangent vectors are hand crafted using known transformation invariance for data (e.g. translation or rotation for iamges). This is similar to data augmentation where we use prior knowledge (i.e. transformation invariance) to create new data. Tangent distance however only encodes infinitesmall perturbations (data augmentation can encode large perturbations as well) and doesn't work properly with ReLU units since such units don't saturate and have to turn off to decrease gradient (data augmentation works well here). A more intelligent way is to 1) use an AutoEncoder for estimating the manifold tangent vectors and then 2) use a tangent regularizer with those tangent vectors. 

- Tangent prop and dataset augmentation require that model to be invariant in certain directions while adversarial training and double backprop encourage the changes to be small in all directions when input change is small. Adversarial training finds inputs near original inputs and trains the model to produce same output for them as well. Double backprop regularizes the Jacobian (gradient wrt all weights) to be small. The same way data augmentation is non-infinitesmall version of tangent prop, adversarial training is the non-infinitesmall version of double bacl-prop. 

