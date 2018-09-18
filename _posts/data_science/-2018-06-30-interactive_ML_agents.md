---
layout: article
title: ML agents in interaction
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---


# Problem: Unifying all ML/RL as interacting agents
Most problems in machine learning have been formulated as a single model with an optimization problem over a single objective. However, all ML problems can be formulated as interactions between multiple agents. In the simplest case, a single model learning from data, the interaction is between a learning agent (model) with a cost objective term and a non-learning agent that doesn't contribute to the objective (data sampler).

- RL and supervised ML have traditionally been used in separate problem domains but they both learn to map (input/states) to (output/actions). The real key difference from mathematical POV is actually the objective funtions. This is much more clear when the output has a structure, for example, in the case of image captioning, 
    + in supervised ML we have a dataset of many (image, caption) pairs, and we want to find a mapping between inputs X and captions Y. Therefore, we optimize the conditional log-likelihood objective p(Y|X) for our model/policy. 
    + in unsupervised ML we have a dataset of (images) and we want to find the manifold of the data to push energy down on, therefore, we use maximum likelihood objective. However, for every datapoint, we would ideally need a contrasting datapoint outside the manifol to push the energy up on. therefore, we may be able to find a mapping between on manifold and out of manifold data points i.e. a pair of (X, X_fake) from only sparse reward signals that are obtained only after a data manifold is formed. 
    + While in RL formulation of same problem, we use a simulator that runs for a sequence of steps (episode) providing an image (state) at each step and asking us to choose a word (action). If we run this simulator for some time, we can build a dataset of trajectories of (image, action) pairs and a bunch of raters provide the reward of getting the captioning right at the end of each captioning episode. Therefore, here we optimize the expectation of the reward function for our model/policy. 

    + A key insight is that the optimial model/policy and the set of actions/predictions that lead to it is a single value that can be written as a dirac delta function of the optimal model/policy. But we can do a continuous relaxation on it to get the soft optimal model/policy which as $$\pi^* (a) = \frac{1}{Z} \exp(\frac{r(a)}{\tau})$$ which would be equal to the hard optimal model/policy at temprature zero $$\tau = 0$$. Now since the optimal policy is continuous we can say write the ML and RL objectives as the two directiond of the KL divergence between a model and it's soft optimal when the temprature is zero i.e.
        * conditional log-likelihood is $$D_KL (\pi^* || \pi_\theta)$$ 
            + This is the mode covering direction, now if the temprature is not zero, we get  reward augmentation for maximum likelihood objective that encourages better non-greedy decoding. 
        * expected reward is $$D_KL (\pi_\theta || \pi^* )$$ 
            + This is the mode seeking direction, now if the temprature is not zero, we get entropy regularization for expected reward objective that encourages exploration.

        * Therefore, the key to connect these two paradaigms is the 'entropy of the policy'. 


- Unsupervised ML can be thought of as multiple agents where one pushes the energy landscape down on the data manifold and others pushes energy up in the rest of the energy landscape (i.e. a GAN). 





- ML models make predictions but we usually want to take actions in the real world. How do we map predictions that do not know anything about the effects of taking actions in the world to best actions? Interactive models (with the help of RL) might be able to address these limitations.

- Distribution shift usually renders well-preforming models moot. For example, if a NN classifier trained on a dataset of images preforms well, it might not work as well if a new type of camera is used to capture images in future. Interactive models might be able to help here as well. 

## What if samplers have objectives too?

Observations made in a domain represent samples of some broader idealized and unknown population of all possible observations that could be made in the domain. Sampling consists of selecting some part of the population to observe so that one may estimate something about the whole population. sampling as a field sits neatly between pure uncontrolled observation and controlled experimentation.  Sampling is usually distinguished from the closely related field of experimental design, in that in experiments one deliberately perturbs some part of the population in order to see what the effect of that action is. […] Sampling is also usually distinguished from observational studies, in which one has little or no control over how the observations on the population were obtained. 

In machine learning, datasets are treated as observational studies where the model is trained on all available data points in a random order. Three main sampling methods used in ML are:
    - Simple Random Sampling: Samples are drawn with a uniform probability from the domain.
    - Systematic Sampling: Samples are drawn using a pre-specified pattern, such as at intervals.
    - Stratified Sampling: Samples are drawn within pre-specified categories (i.e. strata).


Can we do better than presentation of all datapoints in a random order for achieving better performances on our defined goals? Some of the cases where an adaptive sampler might be useful are:
    - Sampling observations:
        - Active learning (which minimum set of samples provide the most information for learning?)
        - Curriculum learning (In which order should the samples be presented?)
    - generalizationa and bias:
        - which subset of data points are most representative of the population (contribute most to generalization) and have least bias?
    - Resampling: 
        - economically using data samples to improve the accuracy and quantify the uncertainty of a population parameter.

If the data sampler is also learning and have an objective, then the sampling process will be adaptive.


## Multiple Learning agents
A number of problems consist of a hybrid of several learning agents (models), each of which passes information to other models but tries to minimize its own private loss function. This upsets many of the assumptions behind most learning algorithms, SGD optimization usually results in pathological behavior such as oscillations or collapse onto degenerate solutions. it has been hypothesized that the combination of many different local losses underlies the functioning of the brain as well. 


- Following are example where optimization has been notoriously hard, 
    - GANs: formulate the unsupervised learning problem as a game between two opponents - a generator G which samples from a distribution, and a discriminator D which classifies the samples as real or false. 
        + To make sure the generator has gradients from which to learn even when the discriminator’s classification accuracy is high, the generator’s loss function is usually formulated as maximizing the probability of classifying a sample as true rather than minimizing its probability of being classified false.
    - Actor Critic RL methods (a single Generator takes an action)
        + While most RL algorithms either focus on learning a value function, or a policy directly, AC methods learn both simultaneously - the actor being the policy and the critic being the value function. 
    - Multiple dialog agents in a conversation (Multiple models take turns in generating an utterance)
    - Even a sequence model with a teacher forcing the right output?


## optimization
- [The Mechanics of n-Player Differentiable Games](https://arxiv.org/pdf/1802.05642.pdf) This paper argues that in interactive ML, the loss function consists of competing terms that constitute games. It analyzes the possible games into three categories based on the Hessain of multiple terms of the loss function w.r.t. their respective variables. If we re-write the Hessian in terms of the addition of a symmetric ((H+H')/ 2) and an anti-symmetric function ((H-H')/2), the games are categorized into three classes. 
	+ first class is potential games where the anti-symmetric term of the Hessian is zero. The constituting terms of the loss in this case, form gradients in the same direction for example, a single objective classification problem. In such scenarios SGD works well since the direction of the first order gradient of the loss constitutes a gradient field and we can follow it to get to a local minimum. 
	+ second class of games where the symmetric term is zero, are what the paper calls Hamiltonian games. Hamiltonian games are similar to energy conserving physical systems that constitute a limit cycle in the gradient field. The direction of the first order gradient is tangent to this limit cycle, therefore, we can't really reduce the loss. for this class of games, the paper suggests Synthetic Gradient Averaging (SGA), that is a transformation on the gradient to map it to the direction perpendicular to the limit cycle. This gradient has similarities to second order and natural gradient methods that map the gradient from a euclidean space to a hamiltonian space. The paper suggests to move in the direction of $$\epsilon + \lambda A^T \epsilon$$, where $$\epsilon$$ is SGD gradient, $$A^T$$ is the anti-symmetric part of the Hessian matrix. 
	+ the third class of games are general games, where we don't have only a potential game or a Hamiltonian game but a mixture of both. In these situations, my physical intuition is that physical system containing all interacting models will be a non-energy conserving system dissipating or adding energy. Therefore, we won't have limit-cycle-like balances in the total energy gradient field landscape and one of the interacting systems may dominate the others. GANs are usually systems of this kind and we usually see one of the involved systems dominate the other, therefore, achieving a balanced minimum is usually hard.  