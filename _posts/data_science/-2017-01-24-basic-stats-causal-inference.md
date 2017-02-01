---
layout: article
title: Basic statistics and relationship to probability 
comments: true
categories: data_science
image:
  teaser: VAE_intuitions/manifold.jpg
---

Am I interested in...?:
description (association) - correlations, factor analysis, path analysis
explanation (prediction) - regression, logistic regression, discriminant analysis
intervention (group differences) - t-test, anova, manova, chi square

#Central Tendency

## Mode
the most commonly occurring value

## Median
the center value after sorting

## Mean
the mathematical average


## Variance
average squared deviation of each number from its mean. a measure of how spread out a distribution is.

## Standard Deviation
square root of the variance. how much scores deviate from the mean.

# Hypothesis Testing
A statistical test provides a mechanism for making quantitative decisions about a process or processes. The intent is to determine whether there is enough evidence to "reject" a conjecture or hypothesis about the process. The conjecture is called the null hypothesis. Not rejecting may be a good result if we want to continue to act as if we "believe" the null hypothesis is true. Or it may be a disappointing result, possibly indicating we may not yet have enough data to "prove" something by rejecting the null hypothesis.

Hypothesis testing is the use of statistics to determine the probability that a given hypothesis is true. The first step is to assume an initial hypothesis or an initial guess. Hypothesis testing is a true false kind of test not a Wh question! For example, we guess that lung cancer is related to smoking. If we have some data from samples of smoking and non-smoking people, we can check to see if out guess is correct or not. 

The procedure is that we model the data with the assumption of a probability distribution that is plausible for the event under study, and then calculate the probability of our guess. It is common in statistics to call the probability of less than 5% for chance a significant relationship although that is debatable! In traditional statistics (tests like t-test, ANOVA, etc), the assumption is that the probability distribution for the event under study is Normal distribution since it's commonplace in nature and that a linear regression can model the event. These are also debatable assumptions! The common process of hypothesis testing consists of four steps.

1. Formulate the null hypothesis H_0 (commonly, that the observations are the result of pure chance) and the alternative hypothesis H_a (commonly, that the observations show a real effect combined with a component of chance  variation).

2. Identify a test statistic that can be used to assess the truth of the null hypothesis.

3. Compute the P-value ( the probability that a test statistic at least as significant as the one observed would be obtained assuming that the null hypothesis were true). The smaller the P-value, the stronger the evidence against the null hypothesis.

4. Compare the p-value to an acceptable significance value  alpha (sometimes called an alpha value). If p<=alpha, that the observed effect is statistically significant, the null hypothesis is ruled out, and the alternative hypothesis is valid.


# Modeling 
The following modeling methods (i.e. t-test, ANOVA, MANOVA, Linear regression, etc) are all different special cases of the General Linear Model (GLM), but since they have been historically developed in different disciplines, they have various names and literature. In 1972, Nelder and Wedderburn introduced Generalized Linear Models. They discovered a unifying framework for a class of regression models whose outputs (predictions) follows a member of exponential family of probability distributions (i.e. Gaussian, binomial, poisson, Gamma, Inverse Gaussian, Geometric and negetive binomial). In this framework, the output is a random variable with a pre-specified probability distribution (for example Normal) whose parameters are calculated from the data (for example mean and variance) and then the probability distribution for coefficients in the linear combination of variables is computed. A linear regression model, hinges upon the following assumptions:

- each observation of the model output follows a Normal distribution $y ~ N(\mu, \sigma^2)$
- All observations have the same variance $\sigma^2$
- The output (predicted value) is a linear combination of variables. 

Nelder and Wedderburn discovered that they can extend linear regression by having the predictor be a linear combination of functions of variables and re-parameterize to make the model a linear combination of more complex variables (or their combinations). Therefore, a nonlinear relationship of two variables can now be represented as a new variable in the generalized linear model. Traditionally, linear regression model parameters were found using maximum likelihood inference. The generalization of linear regression models brought about challenges in the inference problem and new techniques were introduced for parameter estimation in GLMS! The formula for a GLM is:

$Y = XB + U$ , where:

Y is a matrix with one ore more outcome variables (or dependant var);
X is a matrix of predictors (or independant);
B is a vector or matrix of to-be-estimated model parameters;
U contains the model’s errors.

The Generalized Linear model can be written as: Y = f(XB + U). By choosing f appropriately, one could for example obtain probit or logit regression estimators. A GLM is constructed by first choosing explanatory variables X, a probability distribution that is a member of the exponential family, and an appropriate link function f is chosen for which the mapped values accord with the variation function of the distribution. The exponential family distribution allows continuous, discrete, proportional, count and binary prediction Y variables. The link function f, relates the mean of prediction random variable Y to the linear combination of  variables and the variance of prediction to a function of the mean. GLMs are further characterised by following assumptions:

- output is a random variable which follows an exponential probability distribution
- The variance may only change as a function of the mean. 
- observations (Y) are statistically independant. 
- The parameter estimation problem (inference) has solutions using algorithms ( i.e. efficient IRLS algorithm, Newton-Raphson for more complicated models, etc)

Since all outcomes (Y) are assumed to be independant, their joint distribution is factorized as the product of a exponential family distribution that is parameterized by a local mean and a global variance (Since same variance is assumed for all observations). We then solve for parameters that maximize this likelihood. The log likelihood converts the factorized product to a sum and we can easily compute derivatives to find the parameters for maximum likelihood. To fit a model with variables in the linear regression model, we make the local mean parameter a function of the linear combination of the variables in the model and their coefficients. We can again follow maximum likelihood procedure to find the model parameters. With data in hand for observations and model variables, a least square of an optimization routine can find parameters for the model. Note that in this formulation the only random variable is the outcome (Y) and the other variables are considered deterministic. 





Finally, we could extend the generalized linear model to allow latent variables at different levels of analysis. This is what is done in (generalized) Structural Equation Modeling. Structural equation models are similar to directed graphical models, but they are generally only linear-Gaussian, while directed graphical models can have any exponential family conditional distribution. Structured equation models are also more directly related to causality, while directed graphical models concern themselves with conditional distributions. Due to some technicalities, structural equation models can have cycles, unlike directed graphical models. Also, structural equation models tend to be less Bayesian.

Structural equation models and Bayesian networks appear so intimately connected that it could be easy to forget the differences. The structural equation model is an algebraic object. As long as the causal graph remains acyclic, algebraic manipulations are interpreted as interventions on the causal system. The Bayesian network is a generative statistical model representing a class of joint probability distributions, and, as such, does not support algebraic manipulations. However, the symbolic representation of its Markov factorization is an algebraic object, essentially equivalent to the structural equation model.


# Differences of Groups

## Chi Square
Chi square measure is sort of the discrete version of KL-divergence for the distance between two probability distribution. 

Chi Square test compares observed frequencies in the data sample to expected frequencies (e.g. Normal distribution)

## t-Test
looks at differences between **two groups** on some dependant variable of interest. For example, it can determine whether the means of two independent groups differ.

## ANOVA
tests the significance of group differences between **two or more groups**. It only determines that there is a difference between groups, but doesn’t tell which is different. For example, do score for students in class 1 , 2, and 3 differ? The t-test (comparing two groups) is a special case of ANOVA. 

## ANCOVA
same as ANOVA, but adds control of one or more **covariates**. For example, do score for students in class 1 , 2, and 3 differ after controlling for single/dual parent.

## MANOVA
same as ANOVA, but you can **study** two or more **related** dependant variables while controlling for the **correlation** between the dependant variables. if the dependant variables are not correlated, then separate ANOVAs are appropriate. For example, Does ethnicity affect reading score, math score, and overall score among students?

## MANCOVA
same as MANOVA, but adds control of one or more covariates that may influence the dependant variable. For example, does ethnicity affect reading score, math score, and overall score among students after controlling for single/dual parent?

# Relationships

## Correlation
used with two variables to determine a relationship/association. do two variables covary? It does not distinguish between independent and dependent variables.

## Multiple Regression
used with several independent variables and one dependent variable. It is usually used for prediction and can identify the best set of predictor variables. you can enter many independant variables and it tells you which are best predictors by looking at all of them
at the same time. For example, predicting math score (dependant) from age, sex, and iq (independant variables). 

## Path Analysis
looks at direct and indirect effects of predictor variables. It is used for relationships/causality, for example, Child abuse causes drug use which leads to suicidal tendencies.

# Group Membership

## Logistic Regression
It is like multiple regression, but the DV is a dichotomous variable (has two levels). logistic regression estimates the probability of the each state as the values of the independant variables change. For example, What is the probability of a suicide occurring at various levels of alcohol use?



# Causal inference 

Modern statistical thinking makes a clear distinction between the statistical model and the world. The actual mechanisms underlying the data are considered unknown. The statistical models do not need to reproduce these mechanisms to emulate the observable data (Breiman, 2001). Better models are sometimes obtained by deliberately avoiding to reproduce the true mechanisms (Vapnik, 1982, Section 8.6). We can approach the manipulability puzzle in the same spirit by viewing causation as a reasoning model (Bottou, 2011) rather than a property of the world. Causes and effects are simply the pieces of an abstract reasoning game. Causal statements that are not empirically testable acquire validity when they are used as intermediate steps when one reasons about manipulations or interventions amenable to experimental validation.

conceptual framework and algorithmic tools needed for causal inference have been developed in the past two decades due to on advances in three areas:

1. Nonparametric structural equations
2. Graphical models
3. Symbiosis between counterfactual and graphical methods.


