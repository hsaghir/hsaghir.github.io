---
layout: article
title: Connections betwee supervised, unsupervised, and reinforcement learning
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---


## How do Supervised learning and RL relate?

[Lecture on "Towards a unified view of supervised learning and reinforcement learning" at UC Berkeley, 2017](https://www.youtube.com/watch?v=fZNyHoXgV7M&index=24&list=PLkFD6_40KJIwTmSbCv9OVJB3YaO4sFwkX)

In both, we want to learn a mapping from inputs or states to outputs or actions:
    + In supervised learning, we can sample from model, calculated error and optimize parameters. What need to be concerned about is overfitting and model being certain when it hasn't seen enough data points. 
    + In RL, we can sample from our policy function, calculate reward, and optimize parameters (be greedy). What we need to be concerned about is exploitation before enough exploration. 
    + The key concept that connects both paradigms is uncertainty which is quantified using entropy. If we add entropy to our supervised objective, we are regularizing it so that it doesn't overfit. If we add entropy to our expected reward, we are encouraging it to explore more before exploitation.

In supervised learning, we usually have an iid assumption between samples of the dataset where in RL, the samples are a trajectory where iid assumption doesn't hold. However, if we have a structured prediction supervised learning (e.g. image captioning), then we have dependencies across the sequence of predictions for each sample.

### Problem setup

We want to learn a mapping, $$X \rightarrow a$$:
    + from the inputs $$X = [x_1, x_2, ..., X_T]$$. (Inputs might be a trajectory, i.e. iid assumption won't hold)
    + to outputs / actions $$a = [a_1, a_2, ... , a_T]$$
    + that maximizes a non-decomposable expected reward (objective) function $$r(a|X)$$
        - In classical supervised learning, the objective is decomposable due to iid assumption of samples to the distance of model output with target.
        - In RL, the instead of target values, we have a bunch of rating/reward functions that may be sparse (not available for all inputs), and also may be delayed till the end. 
    + Model or policy is probability of actions/outputs given inputs $$\pi_\theta (a|x) = \prod_i \pi_\theta (a_i|X_{1,..., i-1}, a_{1,..., i-1})$$. 
    + The policy or model that maximizes the expected reward or objective, $$E[r(a|X)]$$, is the optimal policy/model $$\pi^* (a|x)$$. If actions are discrete, the optimal policy is represented using a categorical variable. We can use the softmax trick to make it continuous with a temprature parameter ($$\tau$$), i.e. $$\pi^*(a|x) = \frac{1}{Z} \exp(\frac{r(a|x)}{\tau})$$.
        - If we use the soft optimal model, the conditional log-likelihood of the model is exactly the KL divergence between the optimal model and the model i.e. $$KL[\frac{\pi^* (a|x)}{\pi_\theta (a|x)}]$$ when the temprature parameter is zero.
        - If we use the soft optimal policy, the expected reward is exactly the KL divergence in the other direction, i.e. between the policy function and the optimal policy i.e. $$KL[\frac{\pi_\theta (a|x)}{\pi^* (a|x)}]$$ again at temperature zero. 

In classical supervised learning, we minimize the conditional log-likelihood objective function, while in RL, we optimize the expected reward function. These two are actually pretty much very close concepts but are not well understood. The key question is what divergence functions can we use to better do this optimization in both domains. 
    + We can use a non-zero temperature and add it to the maximum likelihood objective in terms of expected reward + tempreture * entropy. [Peng & Williams]
    + we can define a conditional log-likelihood at temperatures higher than zero can call that reward augmented maximum likelihood. [Norouzi et al.]
    + We can combine the two directions of the KL to benefit from both mode seeking (RL case because it samples from policy distribution) and mode covering (Supervised learning case because we are sampling from the optimal model distribution) [UREX -> Norouzi et al]
    + We can optimze the entropy-regularized expected reward with partial rewards that are decomposed (bridging the value & policy based RL)


# How do Unsupervised and reinforcement learning relate?

- In the supervised/semi-supervised learning context, representations that disentangle causal factors are desired (Bengio, 2009) for good performance. On the other hand, psychology (e.g. Gopnik & Wellman (in press)) has argued that there is a need for interaction to discover causal structures. However, representation learning usually happens with static datasets.

- In reinforcement learning, several approaches explore mechanisms that push the internal representations of learned models to be “good” in the sense that they provide better control (causal relationship). The work in "[Independently Controllable Features](https://arxiv.org/abs/1708.01289)" hypothesizes that some of the factors explaining variations in the data correspond to aspects of the world which can be controlled by the agent. That means for each of the factors of variation, there exists a policy which will modify that factor only, and not the others. For example, the object behind a set of pixels could be acted on independently from other objects, which would explain variations in its pose and scale when we move it around. The object in this case is a “factor of variation” and we want to learn a representation that explicitly represents this factor (How does this relate to Capsules of Hinton?).While these may seem strong assumptions about the nature of the environment, our point of view is that they are similar to regularizers meant to make a difficult learning problem better constrained.

- The above work builds on autoencoder representation learning by adding above mentioned constraints on the hidden code. We learn $$n$$ hidden representations, and in tandem we learn $$n$$ policies. We would like policy $$\pi_k$$ to cause a change only in the dimension $$f_k$$ and not in any other features of the hidden code. 

- In order to quantify the change in $$f_k$$ when actions are taken according to $$\pi_k$$, we define the selectivity of a feature as the change in feature $$k$$ normalized over sum of change in all features, when action $$a$$ is taken that causes the environment to transition from state $$s$$ to state $$s'$$.

- The objective function consists of two parts. The autoencoder loss plus negative of the $$\log$$ selectivity (since we want to maximize selectivity).




