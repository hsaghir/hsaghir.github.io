---
title: "Supervised learning and reinforcement learning are the same objective"
description: "Both fit a distribution over outputs conditioned on an input. Both minimize a KL divergence between their model and an optimal target. The only differences are which distribution you sample from and which direction of the KL. Entropy regularization bridges them."
date: 2017-11-12
tags: ["machine-learning", "unified-views", "reinforcement-learning"]
category: "data_science"
cover: "/images/go-board.jpg"
coverAlt: "Close-up of black and white stones on a Go board mid-game. AlphaGo learned the same objective twice: first by supervised learning on expert games, then by self-play reinforcement learning."
---

Supervised learning and reinforcement learning look like different subjects
in a textbook. They are the same objective written two ways. Both fit a
distribution over outputs conditioned on an input. Both can be cast as
minimizing a KL divergence to an implicit "optimal" target distribution.
The only differences are which distribution you sample from and which
direction of the KL you use. Entropy regularization is the knob that turns
one into the other.

This post is a condensed form of a 2017 Berkeley
[lecture](https://www.youtube.com/watch?v=fZNyHoXgV7M) by Mohammad Norouzi.
I am writing it up now because the framing turned out to predict the shape
of 2024-2026 post-training methods (RLHF, DPO, GRPO, reasoning-model
fine-tuning) almost exactly.

## The shared setup

Both paradigms want to learn a mapping from inputs $x$ to outputs or
actions $a$, parameterized as a conditional distribution $\pi_\theta(a \mid x)$.
The output $a$ may be a single label, a caption, a sequence of tokens, or
a trajectory of actions; the shape of the output is irrelevant to the
argument.

What does change is the training signal.

- In **supervised learning**, every input $x$ comes with a target output
  $a^\star$, and you maximize
  $\log \pi_\theta(a^\star \mid x)$ averaged over the dataset.
- In **reinforcement learning**, every output $a$ gets a scalar reward
  $r(a \mid x)$, possibly sparse or delayed, and you maximize the expected
  reward $\mathbb{E}_{a \sim \pi_\theta}[r(a \mid x)]$.

The objectives read as two separate problems. They are not.

## The optimal policy is a Boltzmann distribution

Fix the reward function $r(a \mid x)$. The optimal entropy-regularized
policy at temperature $\tau$ is

$$
\pi^\star(a \mid x) \;=\; \frac{1}{Z(x)} \exp\!\left( \frac{r(a \mid x)}{\tau} \right).
$$

This is the softmax / Boltzmann distribution over outputs, with high-reward
outputs getting probability mass in proportion to $\exp(r / \tau)$. Two
limits make it intuitive. As $\tau \to 0$, $\pi^\star$ concentrates on the
argmax (greedy exploitation). As $\tau \to \infty$, $\pi^\star$ becomes
uniform (pure exploration). In between, $\tau$ trades off exploration
against exploitation in a principled way.

Supervised learning has a similar implicit target. Given a labelled dataset
$\{(x, a^\star)\}$, define $r(a \mid x) = \mathbb{1}[a = a^\star]$.
The optimal policy at $\tau \to 0$ is a delta on $a^\star$. This is just
saying: the "correct answer" is the Boltzmann distribution at zero
temperature, with reward being the indicator of correctness.

Once both paradigms have an optimal target distribution, the question is
how the model gets close to it.

## Both are KL divergences, in opposite directions

Supervised learning's cross-entropy objective, written out, is

$$
\mathcal{L}_{\text{SL}}(\theta) \;=\; -\mathbb{E}_{a \sim \pi^\star}\!\left[ \log \pi_\theta(a \mid x) \right]
\;\propto\; \mathrm{KL}\!\left[ \pi^\star \,\|\, \pi_\theta \right].
$$

You sample from $\pi^\star$ (the data distribution) and push $\pi_\theta$
toward it. This is the **mode-covering** direction of the KL: the model is
penalized whenever $\pi^\star$ puts mass somewhere that $\pi_\theta$ does
not, so $\pi_\theta$ learns to cover every mode of the data.

Reinforcement learning's objective, written with the same Boltzmann
$\pi^\star$, becomes

$$
\mathcal{L}_{\text{RL}}(\theta) \;=\; -\mathbb{E}_{a \sim \pi_\theta}\!\left[ r(a \mid x) \right]
\;\propto\; \mathrm{KL}\!\left[ \pi_\theta \,\|\, \pi^\star \right].
$$

You sample from $\pi_\theta$ (the policy) and push it toward $\pi^\star$.
This is the **mode-seeking** direction: the model is penalized whenever it
puts mass where $\pi^\star$ does not, so $\pi_\theta$ learns to concentrate
on high-reward regions.

The only structural differences between supervised learning and
reinforcement learning are:

1. **Which distribution you sample from** at training time, $\pi^\star$
   (data) for SL, $\pi_\theta$ (policy) for RL.
2. **Which direction of the KL** you optimize, mode-covering for SL,
   mode-seeking for RL.

Everything else (sample efficiency, variance, off-policy corrections,
actor-critic, baselines) is engineering around those two choices.

## Entropy regularization bridges them

Once you see that both are KL objectives, a family of intermediate methods
falls out.

**Reward-augmented maximum likelihood** (Norouzi et al., 2016) samples
proposals from $\pi^\star$ at a positive temperature and treats them as soft
targets. You get supervised-style stable training with access to the full
reward landscape, not just the argmax.

**Entropy-regularized policy gradients** add an $\mathbb{H}[\pi_\theta]$
term to the RL objective. This is exactly the KL-to-$\pi^\star$ at
positive temperature, which prevents the policy from collapsing to a
narrow mode and keeps exploration alive.

**UREX** (Under-appreciated Reward Exploration) mixes the two KL
directions so the model benefits from both mode-covering (to avoid
forgetting good solutions) and mode-seeking (to concentrate on the best
ones).

In all three cases the knob is the same. It is the temperature $\tau$ of
the Boltzmann target, or equivalently the coefficient of the entropy
regularizer.

## What this predicted about 2024-2026

This framing sat in a Berkeley lecture in 2017 and mostly waited. What
happened over the next eight years was that the methods that actually
scaled to frontier models turned out to be instantiations of it.

**RLHF** is exactly the entropy-regularized RL objective at a positive
temperature, with the reward model $r_\phi(a \mid x)$ playing the role of
the (learned) reward. The KL penalty against the base model that every
RLHF paper includes is the entropy-regularization term, written against
the prior rather than the uniform distribution.

**Direct Preference Optimization** (Rafailov et al., 2023) is the same
Boltzmann target, but the authors noticed that if you plug the optimal
policy form $\pi^\star \propto \pi_{\text{ref}} \exp(r / \tau)$ back into
the preference likelihood, the reward cancels out and you can optimize the
policy directly against preference pairs. This is the mode-covering KL
($\mathrm{KL}[\pi^\star \| \pi_\theta]$) applied to preference data, which
is why DPO looks like supervised learning.

**GRPO** and its descendants (the post-training methods behind the 2024-2025
reasoning models) are online entropy-regularized RL with group baselines
instead of value functions, but the objective is structurally the same.

**Reasoning-model fine-tuning** (OpenAI o-series, DeepSeek-R1, and the
open replications) computes a verifiable reward on math and code problems
and trains against the mode-seeking KL. The verifiable reward makes $\tau$
effectively small, so the optimal policy concentrates sharply on correct
traces.

None of these methods required a conceptual breakthrough. The scaffolding
was already in place. What was missing was a big enough base model to
make the Boltzmann target well-defined on useful tasks, and a cheap enough
reward signal (preference labels, unit tests, verifiers) to estimate the
gradient of the KL.

## Takeaways

A few things are worth remembering from this framing.

The split between supervised and reinforcement learning is about *which
distribution you sample from*, not about what you are optimizing. If your
reward is sparse and delayed, you sample from the policy and pay the
variance cost. If your reward is dense (a label, a unit test), you can
sample from the data and get a lower-variance gradient.

The KL direction is not a cosmetic choice. Mode-covering makes training
stable but leaves mass on bad outputs. Mode-seeking sharpens the policy
but risks collapse. Real methods that work at scale tend to mix both.

Temperature is the same knob as the entropy coefficient, the KL penalty,
and the preference-likelihood scaling. Every post-training paper has it;
different papers give it different names.

A method that does not fit in this frame is a signal that something is
genuinely new, and worth paying attention to. Most methods fit.

## Reading

- Norouzi et al., [*Reward Augmented Maximum Likelihood for Neural Structured Prediction*](https://arxiv.org/abs/1609.00150) (NeurIPS 2016).
- Nachum et al., [*Bridging the Gap Between Value and Policy Based Reinforcement Learning*](https://arxiv.org/abs/1702.08892) (NeurIPS 2017).
- Rafailov et al., [*Direct Preference Optimization: Your Language Model is Secretly a Reward Model*](https://arxiv.org/abs/2305.18290) (NeurIPS 2023).
- Shao et al., [*DeepSeekMath / GRPO*](https://arxiv.org/abs/2402.03300) (2024).

---

*2026 note: this post was drafted in late 2017 after a Berkeley lecture by
Mohammad Norouzi. I held it because the 2017 RL literature was fragmented
and the unification felt premature. It is less premature now.*

*Cover image: Go stones on a goban, by [Dietmar Rabich](https://commons.wikimedia.org/wiki/File:Go_--_2021_--_6732.jpg), [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).*
