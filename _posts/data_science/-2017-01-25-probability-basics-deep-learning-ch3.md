---
layout: article
title: Probabilistic graphical models and bayesian inference
comments: true
image:
  teaser: jupyter-main-logo.svg
---
Mathematics is a wonderful tool for understanding and modeling the world. There are different branches of mathematics that deal with different class of things that we want to model from the real world. One such branch is probability theory. Will it rain tomorrow?

How would you answer this question? How do we predict things? We humans use our ability to reason, meaning that we imediately remember our beliefs about raining and look for different evidence that we think are related. We combine our beliefs about the problem and the evidence available to us and generalize them to make a prediction. For example, we might believe that simply looking at the sky and seeing if there are clouds is a good way to roughly predict the weather. or we might see if there is wind. This belief needs evidence to actually predict. So we might go outside and look at the sky. If we see clouds, we make a simple and rough prediction that the probability that it will rain tomorrow is high given that it is cloudy. 

Now how would you enable a computer to reason? How would you model the same thing you did above so that a machine can do the same? Well, one way is write a bunch of if-then rules but they don't provide a reasoning and generalization capability like humans. As you can imagine this approach would very soon make things very messy as the number of rules exponentially grow with combinations of factors we want to consider. For example, what if there is grey clouds? What if there are less clouds? What if we also want to check the wind speed in every hour leading to tomorrow before making a decision? That could easily get out of hands before we can tell a computer how to reason.

There is, however, a better tool for this purpose that is closer to what we humans actually do. It involves the notion of beliefs for a computer. A belief is modeled with a probability distribution in the Bayesian system of thinking.



# Probability theory
Probability theory was originally developed to analyze the frequencies of events. So it was used for frequent events like tossing a coin (frequentist). How about probability of being sick? it doesn't refer to frequency anymore but the degree of belief (Bayesian)! The same set of axioms, rules and formulae will be used in computing both frequentist and Bayesian probability. Probability can be seen as the extension of logic to deal with uncertainty. Probability theory provides a set of formal rules for determining the likelihood of a proposition being true given the likelihood of other propositions.

A random variable takes random values. On its own it is just the set of possible values; a probability distribution has to be attached to it to represent how likely is each state. To be a probability mass function on a random variable x, a function P must satisfy the following properties:
- The domain of P must be the set of all possible states of x.
- Each state has a  $0< p <1$ and the sum of probabilities of all states is 1 (i.e. it is normalized!). 

the p(z;a) notation means function p with argument z parameterized by a. joint probability is the probability of more than one random variable P(X,Y). The probability of a subset of random variables from a joint probability is called marginal probability and is calculated using the sum rule. The name comes from summing in the margin of a paper. the probability of some event, given that some other event has happened is called a conditional probability and can be computed by dividing the joint probability by a marginal probability. Conditional probability does not represent causality. Computing the consequences of an action is called making an intervention query which is in the domain of causal modeling. Any joint probability distribution over many random variables may be decomposed into conditional distributions over only one variable (chain rule or product rule). Two random variables x, y are independent if their joint can be expressed as a product of two factors, one involving only x and one involving only y. Two random variables x and y are conditionally independent given a random variable z, if p(x|z) and p(y|z) are independant.

## Expected Value
EV of a distribution is the mean value in the long run (equals to average in infinity). Expectation of a function over a probability distribution is the mean value that function f takes on when its argument x is drawn from distribution P and is calculated as the sum over x of the product of the distribution and the function! E[·] averages over the values of all the random variables inside the brackets. E[.] is linear.

Variance gives how much the values of a function of a random variable varies as we sample different values of x from its probability distribution. It is defined as expectation of the square of a function minus its expectation. Covariance gives how much two values are linearly related to each other. It is defined as the expectation of the prodcut of first function minus its expectation times second function minus its expectation. correlation normalize the contribution of each variable in order to measure only how much the variables are related, rather than also being affected by the scale of the separate variables. If two variables are independant they have zero covariance but not vice versa since covariance is linear, independance is more general and includes nonlinear relationships as well. The covariance matrix of a random vector x ∈ R^n is an n×n matrix with variance on the diagonal.



## Probability distribution
If Z is discrete random variable, then its distribution is called a probability mass function, denoted P(Z=k); a continuous random variable has a probability density function denoted by f(z|λ). Bayesian inference is concerned with what λ is likely to be by assigning a probability distribution to λ. Recall that under Bayesian philosophy, we can assign probabilities if we interpret them as beliefs. Bernouli is a distribution over a single binary random variable and is controlled by a single parameter. The multinoulli or categorical distribution is a distribution over a single discrete variable with k different states parametrized by a vector p. 

Gaussian distribution is parameterized by two parameters mean and variance. In the absence of prior knowledge about what form a distribution over the real numbers should take, the normal distribution is a good default choice since central limit theorem shows that the sum of many independent random variables is approximately normally distributed, and out of all possible probability distributions with the same variance, the normal distribution encodes the maximum amount of uncertainty. multivariate normal distribution generalizes Gaussian and may be parametrized with a positive definite symmetric covariance matrix and a vector mean (since covariance of random variables i with j is the same as j with i). Since any symmetric matrix is diagonalizable, covariance matrix can always be written in terms of its eigenvalues and eigenvectors. A full-rank covariance matrix, allows it to control the variance separately along an arbitrary basis of directions. A diagonal covariance matrix, means it can control the variance separately along each axis-aligned direction. 

When we wish to evaluate the PDF of a normal (e.g. sample from it), the covariance is not a computationally efficient way to parametrize the distribution, since we need to invert covariance matrix to evaluate the PDF (variance is denominator in Normal dist). We can instead use a precision matrix β. We often fix the covariance matrix to be a diagonal (iid). An even simpler version is the isotropic Gaussian distribution, whose covariance is a scalar times the identity matrix.

Laplace distribution is similar to Gaussian except for the fact that the distance from mean and standard deviation are not squared. Absolute value is used instead of squared. The result is that Laplace has a sharp point and fatter tails than Gaussian. Exponential distribution is just the positive half of Laplace distribution when the mean is zero (no absolute value to make it symmetrical). Gumbel distribution is the exponential of laplace distribution (without absolute value), double exponential!, used to model distribution of maximum values. For example, maximum rain falls per year. Assuming each year rain fall is laplace, the distribution of maximums of years is Gumbel. Dirac delta distribution puts all the probability mass on a single point and integrates to 1. Emprical distribution is the sum of delta functions scaled by m for m points. It the discrete case emprical distribution is just multinouli (multiple bernouli) with probability according to empirical frequency (frequency of occurance) but in the continuous case, it needs delta function. Empirical distribution is usually used for assigning probabiliy distribution to a dataset of m points according to their frequency therfore it maximizes the likelihood. It is useful as the distribution that we sample from when we train a model on this dataset (e.g. **stochastic** gradient descent).

Combining other simpler probability distributions constructs a mixture distribution. The way they are combined is just putting a multinouli distibution on top of component distributions. So first a sample from the top chooses the component and then a sample from the component is the data! Latent variables and joint probability distributions can be understood with the mixture concept. For example, if c is latent random variable and x is observable, the joint P(x,z)=P(x|z)P(z), can be thought of as the top distribution over latent P(z) (i.e. prior), times the distribution over x given from z, P(x|z) (i.e. likelihood). In a way, the notion of joint probability over observed x and latent z, makes it easier to understand the complex probability P(x) (factorizes distribution on x). If we don't know the existance of latent z, we'll just assume a complex distribution on observable P(x). Assuming or knowing the latent exists, makes it easier to express the probability distribution of the observed using joint distribution,P(x,z) , in terms of a mixture of the top distribution P(z) and the component distribution P(x|z). Mixture of Gaussians is a very common mixture where components P(x|z) are in general independant Gaussians that are separately parameterized. 

Some functions arise often while working with probability distributions:

- Sigmoid, its range is between zero and one and it saturates when its argument is very positive or very negative. It's a continuous version of step function. It's used to produce parameter of a Bernouli distribution. If sigmoid is y, its gradient is y(1-y). sigmoid(-x) is 1-sigmoid(x). Sigmoid to power -1 is called logit.

- softmax function, is a generalization of the logistic function that "squashes" a K-dimensional vector of real values to a K-dimensional vector of real values in the range (0, 1) that add up to 1. It is useful to represent a categorical distribution.

- SoftPlus, its the log of sigmoid(-x) or the integration of sigmoid from -infinity to zero. So it's range is zero to infinity. It's a continuous version of max of zero and ramp. It can be useful for producing variance of normal dist

## Measure theory for continuous variables:
it is useful to understand the intuition that a set of "measure zero" occupies no volume in the space we are measuring. For example, within R2, a line has measure zero, while a filled polygon has positive measure. Likewise, an individual point has measure zero. Any union of countably many sets that each have measure zero also has measure zero. A property that holds "almost everywhere" holds throughout all of space except for on a set of measure zero. Because the exceptions occupy a negligible amount of space, they can be safely ignored for many applications. Some important results in probability theory hold for all discrete values but only hold “almost everywhere” for continuous values. Another technical detail of continuous variables relates to handling continuous random variables that are deterministic functions of one another. One must be careful since the probability distribution on each random variable must add up to 1 so some rules might not apply to the deterministic relationship of random variables (see ch2 of deep learning book!?).


## Information theory:
Is concerned with quantifying how much information is present in a signal. It is used in ML to characterize probability distributions or quantify similarity between probability distributions. Basic intuition behind information theory is that learning that an unlikely event has occurred is more informative than learning that a likely event has occurred. To quantify the intuition, Likely events should have low information content, Less likely events should have higher information content, and Independent events should have additive information. Information content of an event x satisfies all three using I(x) = −log P(x). Information content of a probability distribution (Shannon entropy) is just expectation on the information content of all possible events $E_P [-log P(x)]= \sum -p_i log(p_i)$.When x is continuous, the Shannon entropy is known as the differential entropy. Distributions that are nearly deterministic (where the outcome is nearly certain) have low entropy; distributions that are closer to uniform have high entropy (everything is equally possible, lots of uncertainty). 

KL-divergence meausres a sort of distance between two distributions but it's not a true distance since it's not symmetric $ KL(P||Q) = E_P[log P(x) − logQ(x)] $. Some applications require an approximation that usually places high probability anywhere that the true distribution places high probability, while other applications require an approximation that rarely places high probability anywhere that the true distribution places low probability. For example, minimizing in KL(p||q), we select a q that has high probability where p has high probability so When p has multiple modes, q chooses to blur the modes together, in order to put high probability mass on all of them. On the other hand, in minimizing KL(q||p), we select a q that has low probability where p has low probability. When p has multiple modes that are sufficiently widely separated, as in this figure, the KL divergence is minimized by choosing a single mode (mode collapsing), in order to avoid putting probability mass in the low-probability areas between modes of p. Cross-entropy is closely related to the KL divergence with the the difference that the log P(x) is lacking (E_P[log Q(x)]) so minimizing the cross-entropy with respect to Q is equivalent to minimizing the KL divergence since log P is not dependant on Q. There are other divergences as well that one might be interested in looking into.


## Bayesian
Bayesians, have a more intuitive approach. Bayesians interpret a probability as measure of belief. we assign the belief (probability) measure to an individual, not to Nature. we denote our belief about event A as P(A). We call this quantity the prior probability. We denote our updated belief as P(A|X), interpreted as the probability of A given the evidence X. We call the updated belief the posterior probability so as to contrast it with the prior probability. 

Bayesian inference is simply updating your beliefs after considering new evidence. A Bayesian can rarely be certain about a result, but he or she can be very confident. 

Denote N as the number of instances of evidence we possess. As we gather an infinite amount of evidence, say as N→∞, our Bayesian results (often) align with frequentist results. Hence for large N, statistical inference is more or less objective. On the other hand, for small N, inference is much more unstable: frequentist estimates have more variance and larger confidence intervals. This is where Bayesian analysis excels. By introducing a prior, and returning probabilities (instead of a scalar estimate), we preserve the uncertainty that reflects the instability of statistical inference of a small N dataset.


Subjective vs Objective priors
Bayesian priors can be classified into two classes: objective priors, which aim to allow the data to influence the posterior the most, and subjective priors, which allow the practitioner to express his or her views into the prior. We must remember that choosing a prior, whether subjective or objective, is still part of the modeling process. If the posterior does not make sense, then clearly one had an idea what the posterior should look like (not what one hopes it looks like), implying that the current prior does not contain all the prior information and should be updated. At this point, we can discard the current prior and choose a more reflective one.Empirical Bayes
While not a true Bayesian method, empirical Bayes is a trick that combines frequentist and Bayesian inference. As mentioned previously, for (almost) every inference problem there is a Bayesian method and a frequentist method. The significant difference between the two is that Bayesian methods have a prior distribution, with hyperparameters αα, while empirical methods do not have any notion of a prior. Empirical Bayes combines the two methods by using frequentist methods to select αα, and then proceeds with Bayesian methods on the original problem.

A very simple example follows: suppose we wish to estimate the parameter μμ of a Normal distribution, with σ=5σ=5. Since μμ could range over the whole real line, we can use a Normal distribution as a prior for μμ. How to select the prior's hyperparameters, denoted (μp,σ2pμp,σp2)? The σ2pσp2 parameter can be chosen to reflect the uncertainty we have. For μpμp, we have two options:

Empirical Bayes suggests using the empirical sample mean, which will center the prior around the observed empirical mean:
μp=1N∑i=0NXi
Traditional Bayesian inference suggests using prior knowledge, or a more objective prior (zero mean and fat standard deviation).

Empirical Bayes can be argued as being semi-objective, since while the choice of prior model is ours (hence subjective), the parameters are solely determined by the data.

Personally, I feel that Empirical Bayes is double-counting the data. That is, we are using the data twice: once in the prior, which will influence our results towards the observed data, and again in the inferential engine of MCMC. This double-counting will understate our true uncertainty. To minimize this double-counting, I would only suggest using Empirical Bayes when you have lots of observations, else the prior will have too strong of an influence. I would also recommend, if possible, to maintain high uncertainty (either by setting a large σ2pσp2 or equivalent.)

Empirical Bayes
While not a true Bayesian method, empirical Bayes is a trick that combines frequentist and Bayesian inference. As mentioned previously, for (almost) every inference problem there is a Bayesian method and a frequentist method. The significant difference between the two is that Bayesian methods have a prior distribution, with hyperparameters αα, while empirical methods do not have any notion of a prior. Empirical Bayes combines the two methods by using frequentist methods to select αα, and then proceeds with Bayesian methods on the original problem.

A very simple example follows: suppose we wish to estimate the parameter μμ of a Normal distribution, with σ=5σ=5. Since μμ could range over the whole real line, we can use a Normal distribution as a prior for μμ. How to select the prior's hyperparameters, denoted (μp,σ2pμp,σp2)? The σ2pσp2 parameter can be chosen to reflect the uncertainty we have. For μpμp, we have two options:

Empirical Bayes suggests using the empirical sample mean, which will center the prior around the observed empirical mean:
μp=1N∑i=0NXi
Traditional Bayesian inference suggests using prior knowledge, or a more objective prior (zero mean and fat standard deviation).

Empirical Bayes can be argued as being semi-objective, since while the choice of prior model is ours (hence subjective), the parameters are solely determined by the data.

Personally, I feel that Empirical Bayes is double-counting the data. That is, we are using the data twice: once in the prior, which will influence our results towards the observed data, and again in the inferential engine of MCMC. This double-counting will understate our true uncertainty. To minimize this double-counting, I would only suggest using Empirical Bayes when you have lots of observations, else the prior will have too strong of an influence. I would also recommend, if possible, to maintain high uncertainty (either by setting a large σ2pσp2 or equivalent.)



Maximum likelyhood (MLE) vs Maximum a posteriori (MAP)
The maximum likelihood estimate is the value of parameters for which the
observed data is most probable (the likelihood function). The MAP estimate is the value with maximum probability under the posterior. the MAP estimate is found by solving a penalized likelihood problem. We will veer away from the MLE if it has low probability under the prior. as we see more data, The prior term will be swallowed by the likelihood term, and the estimate will look more and more like the MLE.


Conjugacy.
If the posterior distribution is in the same family of distributions
as the prior distribution, i.e., the distribution which we placed on the parameter.
This is a property of pairs of distributions called conjugacy. The beta/Bernoulli
are a conjugate pair. Conjugacy is an important concept throughout the rest of the
course. Conjugate pairs are useful. As you will see, we usually cannot compute the posterior exactly. However, local conjugacy—where individual pieces of a graphical model
form conjugate pairs—will be important in approximate posterior inference. , if the prior gives rise to a posterior having the same functional form, is called a conjugate prior.


models:
Mixture models. Mixture models cluster the data into groups. The hidden variables
are the cluster assignments for each data point and the cluster parameters for each
cluster.

Factor models. Factor models embed high-dimensional data in a low-dimensional
space.These models relate closely to ideas you have heard of like factor analysis,
principal component analysis, and independent component analysis. The posterior factorization describes a low-dimensional space (components) and each data point’s position in it (weights).

Generalized linear models. Regression models or generalized linear models describe
the relationship between input variables and response variables (or output
variables). The hidden variables are the weights of each type of input in the linear
combination, called the coefficients.

random variable is any “probabilistic” outcome. Technically, a random variable is a function from the sample space to the reals [0,1].It is useful to think of quantities that involve some degree of uncertainty as random variables. Random variables take on values in a sample space. They can be discrete or continuous. We call the values of random variables atoms and a set of atoms is called an event. It is helpful to think of a random variable in terms of a surface in a d+1 dimensional space where d is the number of dimensions of the random variable and the +1 part is an additional dimension for the surface to apear on. So a single 1d random variable is a curve in a 2d space. An atom is a point on the curve and an event is a subset of the curve. 

Typically, we reason about collections of random variables. The joint distribution is a
distribution over the configuration of all the random variables in the collection.
For example, imagine flipping 2 coins. The joint distribution is over the space of all
possible outcomes of the four coins [(H H),(H T),(T H),(T T)]. Notice you can think of this as a single random variable with 4 values. The notion of probabilities as surfaces is helpful here. The joint probability is the union of the surfaces of the random variables in m+n+1 dimensional space where m is the dimension of the first and n is the dimension of the second variable. 

Given a collection of random variables, we are often only interested in a subset of them. For example, compute P(X) from a joint distribution P(X,Y). This is called the marginal of X and we compute it by summing over all possible values of Y for each x. Using the surface analogy, this is the sum of all projections of the 2+1d space of (X,Y) to 1+1d space for (X). 

A conditional distribution is the distribution of a random variable given the value of other random variables. For exmaple the probability that the second coin is tails if the first coin is also tails. This is a different distribution for each value of the first event. The notion of a the joint surface is again helpful here. The conditional is just the cross section of the joint at a different point (the value of the first variable).

The Bayes rule comes from the chain rule of probabilies and the notion of marginal. It can be used to calculate a conditional probability using its joint probability and a marginal. 

Independence. Random variables X and Y are independent if knowing about X
tells us nothing about Y. This means that their joint distribution factorizes P(X, Y)=P(X)P(Y) and the joint surface should be combined differently using multiplication. Using the surface analogy, the joint surface is the multiplication of the two surfaces. For example, if the distribution on first variable is ax and on second is bx, the joint distribution is their multiplication abx^2. Now if we compute marginal, i.e. integrate one variable out, we get the original distributions. 




# Inference

let's say there are 2 coins that might have some form of deformity which make one side more probable to come up as we toss them. Let's call the probability of head theta1 for coin1 and theta2 for coin2. You don't know the real probability since all you see is deformed coins. How would you estimate this probability?

A very intuitive way we all do this is to just toss them a number of times and count the number of heads that come up for each. The estimate you just calculated is called maximum likelyhood estimate! If you do this coin tossing enough times, you can say that the true probability of the coin is very close to your estimate. This method you just used to estimate the true probability of the coin using random coin tosses is called monte carlo method!

Let's dig deeper into what we did to get to an estimate. In order to calculate our maximum likelyhood estimate, we tossed the coins a number of times. An implicit assumption we used here is that if we toss the coin randomly, we can get draws from the real probability distribution we are interested in. the coin toss distribution is Gaussian or 

When we setup a Bayesian inference problem with N unknowns, we are implicitly creating an N dimensional space for the prior distributions to exist in. Associated  is an additional dimension, that reflects the prior probability of a particular point. If these surfaces describe our prior distributions on the unknowns, what happens to our space after we incorporate our observed data X? The data X does not change the space, but it changes the surface of the space by pulling and stretching the fabric of the prior surface to reflect where the true parameters likely live.  More data means more pulling and stretching, and our original shape becomes mangled or insignificant compared to the newly formed shape. Less data, and our original shape is more present. Regardless, the resulting surface describes the posterior distribution.

The tendency of the observed data to push up the posterior probability in certain areas is checked by the prior probability distribution, so that lower prior probability means more resistance. Thus in the double-exponential prior case above, a mountain (or multiple mountains) that might erupt near the (0,0) corner would be much higher than mountains that erupt closer to (5,5), since there is more resistance (low prior probability) near (5,5). The peak reflects the posterior probability of where the true parameters are likely to be found. Importantly, if the prior has assigned a probability of 0, then no posterior probability will be assigned there. 