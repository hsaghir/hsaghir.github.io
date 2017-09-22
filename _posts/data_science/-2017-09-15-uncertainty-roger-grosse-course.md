---
layout: article
title: Uncertainty -  Roger Grosse course
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

# [First Lecture]
course focus is uncertainty in function approximation (applications: improving exploration/generalization)

why uncertainty?
- know how reliable a prediction is (confidence calibration)
- prevent model from overfitting:
- ensembling: smooth prediction by weighted averaging
- model selection: which model is best
- sparsification: encode with fewer bits, which connections are not as important
- applications:
    + active learning (which data to label)
    + bandits: improve performance in presence of feedback (explore vs. exploit)
    + bayesian optimization: optimize an expensive blackbox function
    + model-based RL (learn models?)
    + adversarial robustness: knowing uncertainty of a prediction?


workshop quality paper:
    + Tutorial - review article: explain relationships/differences/implement them/ and come up with illustrative examples / run experiments on toy problems
    + apply existing algos in a new setting
    + invent a new algo


Bayesian history:
- Tomas Bayes (1763) invented Bayes rule
- Laplace further developed (1774)
- Metropolis (extended by Hasting 1970)
- Stuart/Geman invented Gibbs sampling (1990)
- 1990s Hamiltonian Monte Carlo / Bayesian Neural nets / PGM /Seq Monte carlo / Variational inference
- 1997 BUGS probabilistic programming lang
- 2000s Bayesian non-parametrics
- 2010 - Stochastic variational inference
- 2012 - Stan lang
- 2014 - Stochastic gradient variational Bayes

NN history:
- 1949 Hebbian rule "fire together, wire together"
- 1957 perceptron algo
- 1969 Minsky/Papert book "Perceptron" on limitations of linear models -> winter
- 1982 Hopfield nets (models of associative memory)
- 1988 Backprop
- 1989 Conv Nets
- 1990s NN winter
- 1997 LSTM
- 2006 Hinton deep learning (DBM/DBN - layer-wise RBM) 
- 2010 GPU
- 2012 Alexnet wins ImageNet obj recognition
- 2016 AlphaGo defeats humans


### Confidence Calibration:
the relationship between model confidence and True events:
- If model predicts rain with 90% probability, what fraction of the time is the model right (90%?)? This is a problem of calibrating the confidence of the model so that model confidence and true probability of occurance match. 

- Negative Log-likelihood is the proper scoring rule. It encourages honesty of the model. That if the model probability output matches the target distribution, it gets proper score.

- The error function (objective) might be different. Sometimes test error might continue to decrease but NLL might not. 

- (Guo et al 2017) tampered the model probability using a temprature parameter inside a softmax logit (T determined from valid set). This helped with calibrating the model so that error and NLL decrease similarly.


### Bayesian modeling:
- The problem with maximum likelihood is that if there is too little data, the model can overfit. For example, if we have only 2 observations of coingflips which are both heads? Max likelihood assigns all probability to Heads. That's not right (it's overfitting). This is called data sparsity problem.
- In Max Likelihood, the observations are treated as random variables but parameters are not. Bayesian approach treats the params as random variables as well. 
- The first problem in Baysian modeling is the choice of prior. 
    + uninformative priors: Gaussian/Uniform prior
    + informative priors: we know 0.5 is more likely for coin probability for example than .99. We can use a Beta distribution as an informative prior to encode this info into the model.(Beta distribution is conjugate which makes the inference easier). Beta dist is commonly used as a prior for Bernouli distribution likelihoods so that the posterior form is also a Beta distribution. 
    + we can integrate out parameters to get posterior of future obserations give current observations (posterior predictive distribution) i.e. prediction. 

- Max a posterior (MAP) finds the most liekly parameters under the posterior. It is like max likelihood but just takes the prior into account. it turns out to be similar to regularization terms.

lessons learned:
    + max likelihood and MAP are about optimization, while Bayesian modeling is about integration.
    + Bayesian solution with uniform prior is robust to data sparsity.

#### application- Bandit problems:
- You have n slot machines and each pays 1$ with an unknwon prob. You have T ties, and you'd like to max your total winning
    + greedy strategy: pick the one that has paid out most freq so far
    + pick the arm (machine) whose params are most uncertain about
    + \epsilon-greedy: do greedy with probability 1-\epsilon but pick a random arm with probability \epsilon (explore)
    + Thompson sampling: explore first till you get good certainty, then exploit.

- thompson sampling is an elegant solution (a balanced exploration vs exploration) [Russo et al 2017]
    + for each arm we do a Bayesian model and estimate a posterior, 
    + sample a bunch from each model, and pick the max. 
    + if these are the current posteriors over the three arms, which one will it pick next?


#[second lecture]

- Linear Regression:
    + Linear regression w/ **basis function**
        * Applying a (nonlinear kernel/ feature map/feature engineering) to feature space before passing through linear regression, we can do nonlinear regression.
        * Neural nets learn the basis function

    + Alternatively, Linear regression w/ **smoothing spline**
        * Fit an arbitrary function but encourage it to be smooth. This is called smoothing spline. Cost is MSE + a regularizer with integral of squared second order derivative of the fitted funtion (F). If the regularizer coefficient is zero, we get overfitting. Infinity coefficient gives us linear regression.
        * Fitted function F is unconstraied, but by solving the variational optimization problem, we find that the optimal F can be expressed as a linear combnination of data-dependant basis functions. $$F(x) = \sum w_i \phi_i(x)$$. $$i$$ are data points!
        * Putting the optimal F into the regularizer, the regularizer can be written as a quadratic function of the weights of bases $$W$$.
        * Therefore, we can optimize the above costs directly over functions (variational optimization) and obtain a analytic solution. 
        * Kernel regression -> variational optimization over functions (Splines)

- Occam's Razor: Intuitive Bayesian interpolation -> More complex functions can explain more datasets, therefore, they should be assigning less probability to each. Simpler models, can explain less datasets, so the ones that it can explain are assigned more probability. Therefore, less complex models that can explain the data are better. 
    + Baysian Occam's Razor is formulated in terms of Bayesian information criterion. BIC is an asymptotic, with lots of data, approximation of the integral of marginal likelihood over possible models. It is calculated as $$0.5 D \log N$$ where D is the dimension of parameter space and N is number of data points. 

- There is a Bayesian version of all theses. To define a Bayesian analogue of smoothing splines, let's convert it to a Bayesian basis function regression problem:
    + The likelihood is easy
    + We'd like a prior that prefers smoothness (similar to smoothing splines)
- The only requirement of the kernel for co-variance of GP is that the co-variance maotrix mush be positive semi-definite. 


## Choosing GP/SVM Kernels:
- the choice of kernel (a.k.a. covariance function) determines almost all the generalization properties of a GP model. You are the expert on your modeling problem - so you're the person best qualified to choose the kernel!
    + The Gaussian (radial basis) kernel. It is very common and has some nice properties. It is universal, and you can integrate it against most functions that you need to. Every function in its prior has infinitely many derivatives. It also has only two parameters

    + Rational Quadratic Kernel: equivalent to adding together many SE kernels with different lengthscales. GP priors with this kernel expect to see functions which vary smoothly across many lengthscales.

    + The periodic kernel allows one to model functions which repeat themselves exactly. Its parameters are easily interpretable. The period p simply determines the distnace between repititions of the function. The lengthscale (variance) determines the lengthscale function in the same way as in the SE kernel.

    + Locally Periodic Kernel: A SE kernel times a periodic results in functions which are periodic, but which can slowly vary over time. 

    + Linear kernel: If you use just a linear kernel in a GP, you're simply doing Bayesian linear regression. works in O(N) time. it's non-stationary. A stationary covariance function is one that only depends on the difference in position of its two inputs, and not their absolute location.

    + Mattern Kernel is similar to Gaussian kernerl but less smooth (seems to be able to model long-correlations and self-similarity)

    + Combining kernels: 
        * sum of two kernels is a kernel (it's the sum of functions of the kernels, so we can sample from the first and then sample from the second and then add them up. Assumes linear combinations and non-interactions between the two componentes). The multiplication of two kernels is a kernel. This is useful in high dimensional problems where we can assume non-interactions and factorizations between dimensions.
        * multiplications of kernels: Intuitively, kernels are like similarity functions. mulitplication is like AND of similarity functions while addition is like OR of similarity function.



- There is a simple way to do GP regression over categorical variables. Simply represent your categorical variable as a by a one-of-k encoding. Then, simply put a product of SE kernels on those dimensions. This is the same as putting one SE ARD kernel on all of them. The lengthscale hyperparameter will now encode whether, when that coding is active, the rest of the function changes. If you notice that the estimated lengthscales for your categorical variables is short, your model is saying that it's not sharing any information between data of different categories. 



