---
title: "Seven textbook models are one linear-Gaussian model"
description: "PCA, factor analysis, ICA, Gaussian mixtures, vector quantization, HMMs, and Kalman filters are the same two equations with different restrictions on the latent variables. One EM recipe fits all of them."
date: 2017-01-21
tags: ["machine-learning", "unified-views", "bayesian"]
category: "data_science"
cover: "/images/kalman-portrait.jpg"
coverAlt: "Rudolf E. Kalman, inventor of the Kalman filter, photographed by ETH Zürich."
---

Seven models in the standard textbook (PCA, factor analysis, ICA, Gaussian
mixtures, vector quantization, hidden Markov models, and Kalman filters) are
the same linear-Gaussian state-space model with different restrictions on
the latent variable. Once you see that, the EM algorithm stops being seven
separate derivations and starts being one derivation you reuse.

This is a companion to the
[earlier post](/blog/2016-12-15-graphical-models/) on graphical models.
That one said: most probabilistic models factor into a graph and compute
with message passing. This one says: the *linear-Gaussian* corner of that
space is a strip of country small enough to map on one page.

## The two equations

The parent model is a discrete-time linear dynamical system with Gaussian
noise:

$$
x_{t+1} = A\, x_t + w_t, \qquad w_t \sim \mathcal{N}(0, Q)
$$

$$
y_t = C\, x_t + v_t, \qquad v_t \sim \mathcal{N}(0, R).
$$

$x_t$ is the hidden state, $y_t$ is the observation, $A$ is the transition
matrix, $C$ is the emission (or generative) matrix, and $w_t, v_t$ are
independent Gaussian noises with covariances $Q$ and $R$. Two facts make
this model tractable. Gaussians stay Gaussian under linear operations, so
every marginal and conditional is Gaussian. And the Markov property means
each $y_t$ depends only on $x_t$, so conditional on the latent chain the
observations factor.

Seven standard models fall out by choosing what $x_t$ looks like and which
parameters you freeze.

## The seven costumes

**Factor analysis.** No dynamics ($A = 0$), a single snapshot instead of a
sequence, continuous Gaussian $x$, diagonal observation noise $R$. The
latent $x$ explains the correlations in $y$; the diagonal $R$ absorbs the
rest.

**PCA.** Factor analysis with isotropic observation noise ($R = \sigma^2 I$)
and the noise sent to zero. The principal directions are the eigenvectors
of the sample covariance.

**ICA.** Same static structure as factor analysis, but with a non-Gaussian
prior on $x$. Strictly this leaves the linear-Gaussian family, but the
state-space template still tells you what to estimate.

**Gaussian mixtures and vector quantization.** No dynamics, but now $x_t$ is
a one-hot categorical (a cluster assignment). The emission $C x + v$ reduces
to picking a cluster mean and adding Gaussian noise. Vector quantization is
the hard-assignment limit where the noise goes to zero.

**Hidden Markov model.** Discrete $x_t$ with Markov dynamics ($A$ becomes a
transition matrix), and the emission is whatever distribution you want over
$y_t \mid x_t$.

**Kalman filter and linear dynamical system.** Continuous Gaussian $x_t$,
continuous $y_t$, dynamics $A$ and emission $C$ both in play. This is the
parent model written out with nothing switched off.

You can draw a 2x2 table with "static vs. sequential" on one axis and
"continuous vs. discrete latent" on the other, and every cell is one of
the above.

| | Static ($x$ is one snapshot) | Sequential ($x_t$ evolves) |
|---|---|---|
| **Continuous latent** | Factor analysis, PCA, ICA | Kalman filter, LDS |
| **Discrete latent** | Mixture of Gaussians, VQ | HMM |

## One fitting recipe: EM

The reason this unification is useful in practice is that one algorithm fits
all of them. EM (expectation-maximization) has two steps that keep the same
shape across the seven models.

1. **E-step.** Given current parameters, compute the posterior over latents.
   For PCA and factor analysis this is a single Gaussian conditional. For
   mixtures this is a softmax responsibility per data point. For HMMs it is
   the forward-backward algorithm. For Kalman filters it is the
   Rauch-Tung-Striebel smoother. All four are instances of message passing
   on the same underlying graph.

2. **M-step.** Given the latent posteriors, fit the parameters $(A, C, Q, R)$
   by weighted least squares. The weights come from the E-step. For each
   specific model this reduces to a closed-form update: eigendecomposition
   for PCA, cluster mean and covariance for Gaussian mixtures, Baum-Welch
   for HMMs, and the Kalman-filter ML update for LDS.

Seeing EM once on the general linear-Gaussian model means you have derived
all seven specific EM algorithms at once. The converse is also true: a
reader who has only seen Baum-Welch can read the Kalman smoother update
and recognize it as the same two steps.

## A couple of subtleties

A few details are worth flagging because they trip people on first pass.

**Degeneracy.** In the general LDS, all the structure in $Q$ can be absorbed
into $A$ and $C$, so you can safely assume $Q$ is diagonal. The same is not
true for $R$, because $y$ is observed and you cannot rescale it freely.

**What question are you answering?** Fitting a linear-Gaussian model splits
into two kinds of task. When the latent has a physical meaning (position in
a tracking problem, phoneme in a speech problem) and the matrices are known
from physics or from a pre-trained model, you are doing filtering or
smoothing, and the quantity of interest is the posterior over states. When
the latent structure is what you are trying to discover (the hidden factors
in factor analysis, the clusters in a mixture model, the regimes of an
economic time series), you are doing learning, and the quantity of interest
is the parameters. EM handles both because it alternates between them.

**Identifiability.** Several of the parameter settings are equivalent up to
an invertible linear map on the latent, because $C x = (C M^{-1})(M x)$ for
any invertible $M$. This is what makes factor analysis and PCA
rotation-invariant, and it is what ICA exploits (a non-Gaussian prior on $x$
breaks the symmetry and identifies a specific rotation).

## Why this matters in 2026

Linear-Gaussian state space models stopped being the headline once recurrent
networks and transformers took over sequence modeling. They did not
disappear. Two threads keep the family alive.

The first is that deep state-space models (S4, S5, Mamba, and the line of
linear-recurrence architectures that followed) are linear-Gaussian LDS with
structured $A$ matrices and learned nonlinear emissions. The HiPPO
parameterization, the diagonal-plus-low-rank trick, and the selective
state-space idea all land inside this family. Knowing the classical LDS
makes it easy to read the new papers.

The second is that initialization and interpretation still lean on the
linear-Gaussian base case. A deep sequence model often gets initialized
near a linear-Gaussian solution, and its first-order behaviour on short
sequences is approximated by the same Kalman-filter math.

The seven costumes are still worth knowing, because the body underneath
did not change.

## Reading

- Roweis and Ghahramani, [*A Unifying Review of Linear Gaussian Models*](https://www.cs.nyu.edu/~roweis/papers/NC110201.pdf)
  (Neural Computation, 1999), the canonical reference for this framing.
- Bishop, *Pattern Recognition and Machine Learning* (2006), chapters
  12-13, for worked derivations of each special case.
- Murphy, *Machine Learning: A Probabilistic Perspective* (2012),
  chapter 13, for the EM derivations side by side.

---

*Cover image: Rudolf E. Kalman (1930-2016), [ETH Library Zürich](https://commons.wikimedia.org/wiki/File:ETH-BIB-Kalman,_Rudolf_E._(1930-2016)-HK_04-01925.jpg), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*
