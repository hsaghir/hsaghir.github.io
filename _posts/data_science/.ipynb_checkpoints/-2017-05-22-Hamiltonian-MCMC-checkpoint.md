---
layout: article
title: Hamiltonian MCMC intuitions
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

# MCMC 
- All the problems in Bayesian statistics come down to an integration (i.e. expectation) and all inference methodologies try to estimate an integral in some way. 
- A density function quantifies our beliefs about likelihood of events using a shape that puts more density on places with more likelihood. For example, imagine a 2d Gaussian, the shell of the bell in 3d space is the density function(quantified using a Gaussian function). While the density usually scales linearly with dimensions, the volume of space under this shell scales exponentially with dimensions. The probability mass is the integral of density times volume. Therefore, in higher dimensions, the mode of the density and the space with highest probability mass are not aligned due to this asymmetry in scaling. 
- Therefore, it is important to quantify probability mass especially in higher dimensions (i.e. compute expectation integral) as opposed to just estimating modes of density function (i.e. MAP inference).  
- There are a couple of ways to estimate the expectation integral i.e. the mass under the probability density shell. 
  + Simple numerical integrations using a grid of squares in all dimensions is called quadrature and scales exponentially with number of dimensions.
  + Monte Carlo estimate of the integral which means taking independent samples from the distribution and using an average to estimate the expectation (integral of probability mass). 
    * The cost of this approach is constant and the variance of the Monte Carlo estimator doesn't on the number of dimensions. It works as far as we can get independent samples from the distribution but how do we get samples? 
    * Turns out the independent sampling also scales exponentially with dimensions since we still have to explore the high dimensional landscape. 
    * The trick to make Monte Carlo work is not to use independent samples, but rather dependent samples and take the correlation of samples into account in the Monte Carlo estimator. This is where a Markov Chain that has a stationary distribution at the target distribution comes in.
    * A Markov chain is an operation; It take a distribution to another distribution. So if we set the Markov chain to preserve our posterior distribution (i.e. posterior at both input and output), we can get dependent samples by running the Markov chain. 
    * The problem is we don't know how to start with the stationary distribution (posterior) that the Markov chain is supposed to preserve. If we knew, it'd be as easy as just running the Markov chain to explore it.
    * So we end up starting with a random distribution and run the chain over and over till it converges to the stationary distribution where the probability mass is, at which point, further running the chain starts exploring the probability mass (meaning we can start collecting dependent samples from our posterior).
    * The posterior distribution represented by an undirected graphical model  is a Markov chain by itself. We'll have to run the chain till it converges to get the posterior. 
    * There are no sufficient conditions for convergence, so we have to apply all the necessary conditions (e.g. invariance). A solution is to run many differently initialized chains in parallel and compare. 


# MCMC Using Hamiltonian Dynamics

- MCMC originally came out of trying to simulate molecules (1953 paper by Metropolis), so there's some physics and chemistry background and terminology. HMC originates from a 1987 paper. Hamiltonian/Hybrid Monte Carlo. Hamiltonian is more descriptive.

- The eventual goal is inference which means calculating the expectation integral of some distribution. We do this by sampling from the distribution and using Monte Carlo to estimate the integral. Therefore, in MCMC, we have some probability distribution that we want to sample from. 

- To use HMC, we form a total energy surface (Hamiltonian function) using the probability distribution we want to sample from (i.e. potential energy) and a helping variable as the momentum (representing Kinetic energy; typically iid Gaussian). Position corresponds to the variables of interest since we want to sample from the probability distribution. The potential energy is minus the log of the probability density for position variables. Momentum variables, one for each position variable, will be introduced artificially.

- One HMC step involves updating momentum variables based on a pre-defined schedule and then using the Kintetic and total energy (Hamiltonian) to calculate the potential energy update. This update is determined by computing a trajectory according to Hamiltonian dynamics, implemented with the leapfrog method. The potential energy update is then used in a accept/reject scheme using the Metropolis Hasting algorithm. This way we can explore the probability surface using the trajectories and then accept/reject the sample where we end up on the surface. 

- Think of HMC as a frictionless puck that slides over an energy surface. The state of this system consists of the position of the puck, given by a 2D vector q (used for sampling a point from the energy surface), and the momentum of the puck (its mass times its velocity), given by a 2D vector p (helper variable used for exploring the landscape). The potential energy, U(q), of the puck is proportional to the height of the surface at its current position, and its kinetic energy, K(p), is proportional to its squared momentum. 

- On a flat part of the surface, the puck moves at a constant velocity, equal to momentum divided by mass. If it encounters a mountain, its momentum allows it to continue, with its kinetic energy decreasing and its potential energy increasing, until all the kintetic energy is converted to potential energy. At that point the puck will slide back down with potential energy converting back to kinetic energy. 

## difference with MCMC 
The *trajectories* are determined not by random walks, but by Hamiltonian dynamics by *discretizing* the differential equations. Thus getting the best of both worlds: proposals that are distant (because we explore the state space more and don't get trapped in local modes) and highly likely (well, we always want this).

Method alternates between two steps:

- Update momentum variables in a "simple" way.
- Update position variables with a Metropolis-Hastings test.


**Very helpful intuition**, in terms of a puck sliding along a surface. We have:

- 2-D vector representing position q.
- 2-D vector representing momentum p.
- Thus, the "states" here are 4-D vectors.
- Potential energy function (a.k.a. stored energy), proportional to height of
  puck, a function of q.
- Kinetic energy function (a.k.a. energy in a "body" due to motion), equal to
  $$|p|^2/(2*(mass_of_puck))$$, i.e. a function of p. Inversely related with
  potential energy as puck moves along surfaces.



## Math Notes

### Preliminaries and Basics

The "system" is defined by a Hamiltonian function, H(q,p), where q and p are
both d-dimensional vectors representing position and momentum, respectively (why
the same length?).

How is he deriving Equations 5.1 an 5.2 when we haven't even defined the
function? Or are these derivatives a *requirement* for a valid Hamiltonian
function? What's the context? I am pretty sure these are definitions we have to
assume but let's see if anything I read later clarifies it.

We define the *Hamiltonian function* as:

H(q,p) := U(q) + K(p)

and usually, U(q) = -log(p(\theta | ...)) and K(p) = (1/2) p^T M^{-1} p, where M
is a symmetric, positive definite "mass" matrix, often a multiple of the
identity. Equations 5.6 and 5.7 following that make sense.

PS: their simple example has sines and cosines in their law, and I remember that
Goodfellow's tutorial on GANs does almost the same thing! Goodfellow's tutorial
said:

> Differential equations of this form have sinusoids as their set of basis
> functions of solutions.

So just remember that when I see those style of differential equations, think
sinusoids! And we can solve "boundary points" to get the coefficients. Also, for
that example, maybe think of the posterior distribution as that clockwise ring,
so the samples are correctly sampling across that clockwise ring?

Four (probably non-exhaustive?) properties of the Hamiltonian for the purposes
of making "accurate" MCMC steps:

- **Reversibility**. We have T((q(t),p(t))) = ((q(t+s),p(t+s))) and the reverse
  is true by switching signs on the derivative equations. This makes sense.
  Also, I should probably start thinking of "states" as (q(t),p(t)) tuples.

- **Conservation of the Hamiltonian**. Yes, now I should definitely think of q
  and p as functions of t, so that we can take derivatives of H with respect to
  t, which then invokes the chain rule. Got it! Ok, what does it mean to
  "conserve the Hamiltonian?" That means, if the time changes, the
  H((q(t),p(t))) value stays the same! To prove this, we have to show that H is
  a constant, i.e. that dH/dt = 0. In fact, this invariance is key to showing
  that MCMC will sample from the "correct" distribution. I have to think about
  what this means for the probabilistic interpretation.

- **Volume Preservation**, aka Liouville's Theorem. Again, *what does this
  mean*? We have points (q(t),p(t)) which lie in some space with volume. Then
  applying the function T to that means the image has the same volume. Not quite
  helpful, but he proves it by showing that the determinant of the Jacobian
  (i.e. the Jacobian that defines the dynamics) has absolute value one, and I
  remember that the determinant is another way of describing volume.

- **Symplecticness**. This defines an extra condition on the Jacobian matrix of
  the dynamics, and which also implies volume preservation.

Three out of these four are preserved even when discretizing in computer
simulations. The one that isn't is the Hamiltonian conservation.


### More Advanced Material

What does discretization mean in our context? The main continuous variable we
have discussed is the time t, so we have to deal with small time increments
\epsilon. (Isn't this the same as other continuous functions? For instance, do
we need discretization to approximate a Gaussian density function? Why is there
a special focus on discretization? I'm not sure. TODO find out.)

Two main ways to do this:

- **Euler's Method**. This looks like it has something to do with the definition
  of the derivative. Is the poor performance of raw Euler's method (see Figure
  5.1(a)) analogous to compounding errors (i.e. *covariate shift*) in imitation
  learning? A modification is possible. [TODO understand technical details]

- **The Leapfrog Method**. This has the same flavor as Euler's method, but is
  better. We perform *half-steps*. [TODO understand technical details]

Yeah, I'm pretty sure that for Figure 5.1, we should think of the Hamiltonian
function as the distribution which we want MCMC to approximate. After all, a
circle is some distribution in some dimension space!  Note also the discussion
on *shear* transformations, which preserves volume.

Both of these methods are supposed to estimate p and q after some number of time
steps. I think that makes sense. Is this related to what I think of as MCMC
sampling \theta points? TODO find out! Also, how related are these updates to
the momentum SGD?

- **Section 5** is about variations of HMC, involving splitting of the
  Hamiltonian (not what I'd like to do) and others such as the Langevin Method
  and partial momentum refresh (interesting!).


## Takeaways

Biggest takeaway: if we're doing MCMC for "nontrivial" applications, we *have*
to use Hamiltonian Dynamics in some way. I'm not sure what the alternative is
(and don't say random walk proposals!). BTW it looks like the author has already
been using Hamiltonian Dynamics for neural network models. These probably
weren't deep networks, though; is there a way to scale it up?


# Bayesian Learning via Stochastic Gradient Langevin Dynamics

Note: I have already discussed some of the material [in a blog post](https://danieltakeshi.github.io/2016-06-19-some-recent-results-on-minibatch-markov-chain-monte-carlo-methods/).

General idea: this paper is about the problem of Bayesian MCMC methods, which have grown out of favor compared to optimization methods since it's harder to apply minibatch techniques with MCMC methods (which require the full posterior for "exactness," which technically means detailed balance in our context). They use updates that are similar to stochastic gradient descent, but which inject the correct amount of noise in them so that the resulting set of samples "moves around enough" to approximate a *distribution* rather than a single point estimate.


## Introductory Material

The key passage:

> One class of methods "left-behind" by the recent advances in large scale machine learning are the Bayesian methods. This has partially to do with the negative results in Bayesian online parameter estimation (Andrieu et al., 1999), but also the fact that each iteration of typical Markov chain Monte Carlo (MCMC) algorithms requires computations over the whole dataset.

You know, the "left behind" stuff are like Trump voters, and the "optimization methods" which can easily use stochastic gradient descent are like the "current" people.

OK, this is the *real* key passage:

> Our method combines Robbins-Monro type algorithms which stochastically optimize a likelihood, with Langevin dynamics which injects noise into the parameter updates in such a way that the trajectory of the parameters will converge to the full posterior distribution rather than just the maximum a posteriori mode.

There's also discussion of this algorithm being a "two-phase" algorithm, the first being like SGD and the second being like Bayesian MCMC.


## The Algorithm

Their key idea is to use Equation 4, which combines the best aspects of Equations 2 and 3.

I didn't quite get the entirety of the argument in Section 3 about why the \theta_t samples collectively approach samples from the posterior distribution as t -> infinity. The other way to frame their argument here is that they're showing Equation 4 approaches Equation 3, where Equation 3 is the "exact" one that they wish to simulate. Also, using Equation 3 means there's no need for rejection tests since the (Langevin) dynamics will accurately approximate the "system," i.e. the posterior.

For their analysis, they use Equation 6, which indeed is a zero-mean random variable, whose variance depends on the minibatch in question. The mean of the minibatch is \sum_{i=1}^N \nabla \log p(x_i | \theta) but that gets subtracted away, hence zero mean.

OK, I understand Equation 7, yes as t -> infinity, it approaches Langevin dynamics exactness since \epsilon_t^2 is dominated by \epsilon_t without the square. And Langevin dynamics (IF it's exact) means we can ignore the expensive MH test.

OK, that's good, we know it approaches Langevin dynamics. But the *next* step is to show that the sequence of samples now converges to the desired posterior. (Why are these claims not necessarily true together? I *think* it's because the previous claim assumes large t, but we start with t=1 and must actually *get* to that point, perhaps somewhere earlier before t is large, we lose the ability to arrive there?) They show this part via showing that a subsequence converges, arguing that the total injected noise between two points in the subsequence is O(sqrt(\epsilon_0)), hmm ... well if we have zero mean Gaussians with variance \epsilon_0 the square has expectation \epsilon_0 so the norm (which applies the square root) will mean sqrt(\epsilon_0), I think. Then this noise dominates the extra noise from the stochastic gradient part. (I'm still not sure at a high level how this works, I suppose it's supposed to be intuitive ...)

They claim:

> In the initial phase the stochastic gradient noise will dominate and the algorithm will imitate an efficient stochastic gradient ascent algorithm. In the later phase the injected noise will dominate, so the algorithm will imitate a Langevin dynamics MH algorithm, and the algorithm will transition smoothly between the two.

Then Section 4 is dedicated to providing the transition point, when we should start collecting samples only *after* the algorith has entered its posterior sampling phase, i.e. when it "becomes Langevin dynamics."


## Experiments

I have some extensive experience trying to get their first experiment to work. SGLD works (i.e. I verified their results), and so does our method. I'd like to modify our code for that experiment to make it more robust, though, but I never find the time. =(

I haven't looked at the second one in much detail.

I did not have time to digest the ICA experiments and it is unlikely that I ever will.


## My Thoughts and Takeaways

For the most part, this paper is clear to me (except the ICA part but that's not my concern). I will try to think about how I can use similar techniques. For our MCMC paper (on arXiv as of this writing) we didn't directly use SGLD but I think there's a way to combine this technique with ours.