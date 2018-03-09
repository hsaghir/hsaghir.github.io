---
layout: article
title: Estimating Gradients of expectations
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---


Back-propagation (Rumelhart & Hinton, 1986), computes exact gradients for deterministic and differentiable objective functions but is not applicable if there is stochasticity or non-differentiable functions involved. That is the case when we want to calculate the gradient of an expectation of a function with respect to parameters $$\theta$$ i.e. $$ \nabla_\theta (E_q(z) [f(z)])=\nabla_\theta( \int q(z)f(z))$$ . An example is ELBO where gradient is difficult to compute since the expectation integral is unknown or the ELBO is not differentiable.

# permuations:
## with Gumbel-Sinkhorn 
Learning permutation latent variable models requires an intractable marginalization over the combinatorial objects. 

The paper approximates discrete maximum-weight matching using the continuous Sinkhorn operator. Sinkhorn operator is attractive because it functions as a simple, easy-to-implement analog of the softmax operator. Gumbel-Sinkhorn is an extension of the Gumbel-Softmax to distributions over latent matchings.
https://openreview.net/pdf?id=Byt3oJ-0W

Notice that choosing a category can always be cast as a maximization problem (e.g. argmax of a softmax on categories). Similarly, one may parameterize the choice of a permutation $$P$$ through a square matrix $$X$$, as the solution to the linear assignment problem with $$P_N$$ denoting the set of permutation matrices. The matching operator can parameterize the hard choice of permutations with an argmax on the inner product of the matrix $$X$$ and the set of $$P_N$$ matrices i.e. $$M(X) = argmax <P,X>$$. They approximate $$M(X)$$ with the Sinkhorn operator. Sinkhorn normalization, or Sinkhorn balancing iteratively normalizes rows and columns of a matrix.

## Variational permutation inference
with Reparameterizing the Birkhoff polytope for variational permutation inference


## [Grammer VAE](https://arxiv.org/pdf/1703.01925.pdf)
key observation: frequently, discrete data can be represented as a parse tree (i.e. a sequence of production rules) using a context-free grammar. why not directly encode/decode to and from these parse trees, ensuring the generated outputs are always valid. 
    1. Take a valid sequence, parse it into a sequence of ordered and reversible production rules (i.e. a parse tree).
    2. Assign a one-hot representation to each production rule, feed to an LSTM encoder, learn a hidden distribution on the latent space of a VAE. 
    3. LSTM decoder generates a sequence of production rules.
    4. Do an offline semantic check to weed out nonsensical sequences. 
    5. applying sequence of production rules in order to convert to grammatically correct strings. 

Grammars exist for a wide variety of discrete domains such as symbolic expressions, standard programming languages such as C, and chemical structures. A context-free grammar (CFG) is traditionally defined as a 4-tuple G = (V, Σ, R, S): V is a finite set of non-terminal symbols; the alphabet Σ is a finite set of terminal symbols, disjoint from V ; R is a finite set of production rules; and S is a distinct non-terminal known as the start symbol. The rules R are formally described as α → β for α ∈ V and β ∈ (V ∪ Σ)* with * denoting the Kleene closure. In practice, these rules are defined as a set of mappings from a single left-hand side non-terminal in V to a sequence of terminal and/or non-terminal symbols, and can be interpreted as a rewrite rule. Note that natural language is not context-free. Application of a production rule to a non-terminal symbol defines a tree with symbols on the right-hand side of the production rule becoming child nodes for the left-hand side parent. The grammar G thus defines a set of possible trees extending from each non-terminal symbol in V. produced by recursively applying rules in R to leaf nodes until all leaf nodes are terminal symbols in Σ. 

## [Syntax-Directed VAE for Structured Data](https://openreview.net/forum?id=SyqShMZRb)
The idea is to add an 'attribute grammar', called 'stochastic lazy attribute',  to convert the step 4 from grammer VAE (the offline semantic check) into online guidance for stochastic decoding.


### project ideas:
- taking models with non-differentiable part (i.e. DRAW, Neural Turing Machine, RL, etc) and applying a continuous relaxation like RELAX/REBAR

# VQ-VAE:
- Embed observation into continuous space
- Transform the Z into a discrete variable over k categories
    - make a lookup table embedding. 
    - find the nearest neigbour categorical embedding
- Take the embedding and feed to decoder
    + KL will become a constant and can be removed from objective.
    + gradients are straight through gradient estimator (pretend the non-diffrentiable part doesn't exist)
- The latent is a matrix of categorical variables
    + if we sample each categorical independently, the reconstruction won't have a coherent structure. 
    + Therefore, they used an autoregressive model (pixelCNN) to sample the categorical latent for generation. 

# DRAW
- use RNNs as encoder/decoder
- use spatial attention mechanism
    + main challenge is where to look
        * 1. using REINFORCE
        * 2. build a fully differentiable attention (like soft attention)
        * 3. what they did was to to sample from decoder, send it to encoder, 


# Attend infer repeat
- difference from DRAW is it's focus on learning an understandable representation compared to focus of DRAW on reconstruction
- Objective is ELBO. challenges:
    + the size of latent space is a random variable itself
    + mix of continuous and discrete latent random variables (presence, where, what)

# learning hard alignment with variational inference
- a seq2seq that can work in online setting. 
    + where the input seq comes in, in real time. 
    + TIMIT dataset (speech) 
- we want the model to attend to different parts of speach and decide if it maps to a phoneme (uses hard attention which is not differentiable)
- A bernouli decides at each input if we sould output or not. Therefore, the seq will be variable size. 
- used VIMCO gradient estimator. 

# Thinking fast as slow with deep learning and tree seach (AlphaGo Zero)
- why not just use REINFORCE (why use MCTS)?
    + reinforce is just average of reward times gradient of log of some policy.
    + we can only use differentiable policies
    + reinforce has high variance
    + 

- alpha-beta tree search is expensive. Monte carlo tree seach (MCTS) is and approaximation. 
    + selece nodes according to a huristic probability function (UCP function)
    + traverse the tree to get at leaf node:
        * if this node has not been explored, run simulation get reward
        * if you have, add child node to tree, run simulation from a random child
    + update upper confidence bound(UCP) values of node along path from leaf to node
- MCTS in acion:
    + selection 
    + expansion (add it's child)
    + simulation (get reward)
    + backprop


# GANs for text
- Adversarial loss (also happens in supervised learning when we optimize hyperparams)
    + convert the minmax optimization into a nested Bilevel optimization 
    + Actor-critic method combines policy and value learning and learns them simultaneously
    + formulate a GAN as an actor-critic method (generator-> plicy, discriminator -> value, state -> real/fake image, environment-> randomly gives real/fake image, reward -> real image label)
        * the critic cannot learn the causal structure of the environment


- Adversarial Autoencoder. For each minibactch:
    + First train a VAE
    + then take the encoder as generator of a GAN and compare it with samples from a prior you want to impose. The discriminator will try to tell apart these two
    + can be done on SSVAE to encourage the categorical be closer to one-hot. 
        * train auto-encoder
        * train the two discriminators (one for continuous, one for discrete)
        * additional classification loss

# Hierarchical Multiscale RNNs

- They learned a hierarchy starting from characters. The model learns to make decisions about boundries of words, phrases, sentences, etc in the hierarchy of layers. 
 
- they only added three operations UPDATE (update params using gradients i.e. assuming the boundary hasn't been seen yet), COPY (copy hidden state to above layer i.e. when boundary is seen) and FLUSH (zero out the hidden i.e. when the boundary was seen in the state before and now we zero out the hidden) to a stacked RNN. The problem is that hard operations aren't differentiable. so they used straight through gradient.
https://arxiv.org/pdf/1609.01704.pdf


# program synthesis 

