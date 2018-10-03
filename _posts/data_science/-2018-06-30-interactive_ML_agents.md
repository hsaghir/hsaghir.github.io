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


## Adversarial training


- BEGAN:
    - an autoencoder as an energy-based discriminator(D) + an generator(G)
        - the loss function for the discriminator(D) is the difference between the reconstruction loss of the real image and the reconstruction loss of the generator.
        - the loss function for the generator (G) is the reconstruction loss of the fake image

- AAE(adversarial autoencoder):
    - a VAE + a discriminator. the image is mapped to lower dim vector (z_real) and then decoded, a sample from the prior (z_prior) is contrasted with (z_real) using the discriminator.

BiGAN:
    - a generator/decoder that produces fake image (x_fake) from noise (z_fake), an encoder that maps real image (x_real) to lower dimentional noise (z_real).

- ALI/BiGAN:
    - a generator/decoder(G), an encoder(E) and a discriminator(D). first G uses random noise (z_fake) to produce fake image (x_fake). Then the real image (x_real) is passed through the encoder to generate a lower dimentional noise vector (z_real). the discriminator consists of two two x, and z branches that get combined into a single score. 
        - first (x,z)  pairs for both real and fake are computed and passed to the discriminator, discriminator is trained with BCE or margin loss. 
        - then discriminator is fixed, and gradients are backproped to the generator and the encoder. 


## Cooperative training
### Cooperative Training as a solution to discrete GANs.

in order to estimate the target distribution well in both quality and diversity senses, an ideal algorithm for generative models should be able to optimize a symmetric divergence or distance like Jenson-Shannon Divergence instead of KL.

Each iteration of Cooperative Training mainly consists of two parts. The first part is to train a mediator Mφ, which is a predictive module that measures a mixture distribution of the learned generative distribution Gθ and target latent distribution P as
Mφ =(P + Gθ)/2. 


- Lets say you want to train a language model to generate sentences for you and don't want to train it with maximum likelihood or teacher forcing. In teacher forcing, at every time step you give the right output as the previous word instead of what the RNN has actually generated, which ignores where the network made mistakes in generating words. You ideally want to provide your network a previous word that is a combination of what it generated in previous step and what it should have generated so that it can assign credits and get gradients to fix its mistakes. 

- Training with teacher forcing and maximum likelihood is equivalent to minimizing the KL divergence between the true distribution (P) and generator distribution (G). The alternative we are looking for based on above discussion is JS divergence between the two distributions instead. Since the actual JS divergence requires the true distribition (P) which we don't know, we use a second network called the mediator to approximate the JS divergence between the two distributions. It is achived by minimizing the KL divergence between the real JS divergence (unknown) and a variational approximation to the JS divergence through the parameterized mediator distribution M. This way, the mediator can provide an approximation for the JS divergence between the true distribution (P) and generator distribution (G) that we can use to train the generator. 

- Then they introduce a clever way for calculating the gradient of the approximated JS divergence as well. the gradient for training generator via Cooperative Training can be formulated as the sum of the differences of log logits that G and M produce for a sequence (sentence), times the logits that the G produces. You start from the first word in the sequence, calculate above, and repeat for all subsequent words and then sum all up. That's the gradient for your G without the need for sampling from the softmax of logits, therefore, you can train your generator that is producing categorical variables without the need to pass gradients through the undifferentiable categorical variables. 

Algorithm (example for language models):
- you make two RNN language models as the generator (G) and the mediator (M).
- you take a bunch of real sentence from your dataset, pass to generator and get also sentences that the generator produces. 
- you put the real sentences and the generated sentences in the same minibatch and send to the mediator. You then train the mediator as a language model with this mixed minibatch data using maximum likelihood. 
- Then you take another real data batch, send through the generator to get the logits_G for each of the words in the sentence.
- Then you pass this tensor of logits_G to your mediator to get another same-size tensor of logits_M.
- Then you starting from the first word, apply $$\log$$ to the logits vectors for each word and then calculate the difference and multiply by logits_G i.e. $$logits_G * (\log(logits_G) - \log(logits_M))$$.
- then sum the above values for all the words in the sentences in the minibatch and that's the gradient of your G.
- backpropagate it to your G and train your generator to generate sequences for you!


## Cross-View Training
- The idea is to impelment semi-supervised learning by having a supervised model learn from the supervised portion of dataset and a bunch of auxillary models that learn from the unsupervised portion of data by treating the predictions of supervised model as groud truth labels. 
    - idea here is to have a shared encoder (between supervised and unsupervised models) that learns better representations from the unsupervised portion of data.
        - the shared encoder is a BiLSTM. the forward LSTM representation is used in the supervised task and the backward LSTM representation in the unsupervised tasks.
    - The auxillary models that learn from unsupervised data, see different views of the same input and all have to learn to make predictions similar to what the supervised model predicted only from their own view of the input.
    - We would then alternate between supervised and unsupervised portions of learning while having an encoder that learns from both portions.  



## Co-Training

co-training paradigm proposed by Blum and Mitchell, trains two classifiers separately on two different views, i.e. two independent sets of attributes, and uses the predictions of each classifier on unlabeled examples to augment the training set of the other, utilizing the natural redundancy in the attributes.

The standard co-training algorithm requires two sufficient and redundant views, that is, the attributes be naturally partitioned into two sets, each of which is sufficient for learning and conditionally independent to the other given the class label.

Goldman and Zhou proposed an algorithm which does not exploit attribute partition. However, it requires using two different supervised learning algorithms that partition the instance space into a set of equivalence classes, and employing time-consuming cross validation technique to determine how to label the unlabeled examples and how to produce the final hypothesis.


Let L denote the labeled example set with size |L| and U denote the unlabeled example set with size |U|. In co-training style algorithms, two classifiers are initially trained from L, each of which is then re-trained with the help of unlabeled examples that are labeled by the latest version of the other classifier. In order to determine which example in U should be labeled and which classifier should be biased in prediction, the confidence of the labeling of each classifier must be explicitly measured. Sometimes such a measuring process is quite time-consuming.

### Tri-Training [ZH Zhou, M Li]

Assume that besides these two classifiers, i.e. h1 and h2, a classifier h3 is initially trained from L. Then, for any classifier, an unlabeled example can be labeled for it as long as the other two classifiers agree on the labeling of this example, while the confidence of the labeling of the classifiers are not needed to be explicitly measured. 


For instance, if h2 and h3 agree on the labeling of an example x in U, then x can be labeled for h1. It is obvious that in such a scheme if the prediction of h2 and h3 on x is correct, then h1 will receive a valid new example for further training; otherwise h1 will get an example with noisy label. However, even in the worse case, the increase in the classification noise rate can be compensated if the amount of newly labeled examples is sufficient, under certain conditions.


Tri-training does not require sufficient and redundant views, nor does it require the use of different supervised learning algorithms whose hypothesis partitions the instance space into a set of equivalence classes.

In contrast to previous algorithms that utilize two classifiers, tritraining uses three classifiers. This setting tackles the problem of determining how to label the unlabeled examples and how to produce the final hypothesis, which contributes much to the efficiency of the algorithm.

### Curriculum learning
- the idea is that a student learns best not when complex and easy examples are randomly presented, but when a certain curriculum is followed in presentation of examples that lets model achieve best performance. The result from [this](http://ronan.collobert.com/pub/matos/2009_curriculum_icml.pdf) paper is that when training machine learning models, start with easier subtasks and gradually increase the difficulty level of the tasks.

Another [paper](https://openreview.net/forum?id=SJ1fQYlCZ) studies growing sets of data and gives evidence that we can also obtain good results by adding the samples randomly without a meaningful order. The main learning strategy considered is learning with growing sets, i.e. at each stage a new portion of samples is added to the current available training set. At the last stage, all training samples are considered. The classifier is re-learned on each stage, where optimized weights in the previous stage are given as initial weights in the next stage. Specifically, the authors argue that adding samples in random order is as beneficial as adding them with some curriculum strategy, i.e. from easiest to hardest, or reverse.

- Can we learn this curriculum?
    - a GAN sort of learns a curriculum, as the discriminator and generator gradually evolve together. The generator doesn't suddenly provide the hardest possible example, and the discriminator doesn't suddenly discriminate the hardest possible negative example from the positive ones. Both evolve together according to a learned curriculum of gradually harder negative examples. 

## The Mechanics of n-Player Differentiable Games


- a mechanical system (may be consisting of multitudes of pars) is comprehensively described by a set of position variables that reflect the degrees of the freedom of the system in space and a set of conjugate variables called momentum that describe the rate of change of the spatial degrees of freedom in time dimension.

- A Hamiltonian is a scalar function that describes the physics of the mechanical system under study by describing the interaction of the degrees of freedom of the system in space and time. 
    - If we can describe the physics of a systems by a writing its Hamiltonian, we can predict/study the temporal evolution of the system using the gradient of the Hamiltonian. The equations of motion of the system are: 
        + The gradient of the Hamiltonian w.r.t. positional variables yields the rate of change of momentum in time. 
        + The gradient of the Hamiltonian w.r.t. momentum variables yields the rate of change of position in time. 
    - In the case of physical systems, the Hamiltonian is defined as the total energy of a system obtained by the sum of the potential energy (depending only on the spatial state of the system) and kinetic energy (depending only on momentum and temporal rate of spatial state) of the system. 
        + Kinetic energy is defined as the work needed to accelerate a body of a given mass from rest to its stated velocity or decelerate a body at its stated velocity to rest. The kinetic energy of a non-rotating body is defined as $$(mv^2)/2$$ or $$(p^2)/2m$$
        + Potential energy is defined as the work needed to change the spatial position of a body from its ititial position to the stated position (common types: as gravitational, elastic, electric, and maybe neural network potential energy).



- An ideal mechanical system is energy preserving meaning that the total energy of the system (i.e. Hamiltonian) is constant and the energies are only converted from potential to kinetic or vice versa in the system. It is possible to also have energy dissipating or energy gaining systems. 


- Some vector fields are gradient fields (if the partial derivatives are symmetrical) which means that they can be written as the derivative of a scalar field. In such a case, running the dynamical system can be thought of as performing gradient descent in the scalar field. integrating such a gradient field (i.e. force) will yield the scalar field (i.e. potential energy). 



- [The Mechanics of n-Player Differentiable Games](https://arxiv.org/pdf/1802.05642.pdf) This paper argues that in interactive ML, the loss function consists of competing terms that constitute games. It analyzes the possible games into three categories based on the Hessain of multiple terms of the loss function w.r.t. their respective variables. If we re-write the Hessian in terms of the addition of a symmetric ((H+H')/ 2) and an anti-symmetric function ((H-H')/2), the games are categorized into three classes. 
	+ first class is potential games where the anti-symmetric term of the Hessian is zero. The constituting terms of the loss in this case, form gradients in the same direction for example, a single objective classification problem. In such scenarios SGD works well since the direction of the first order gradient of the loss constitutes a gradient field and we can follow it to get to a local minimum. 
	+ second class of games where the symmetric term is zero, are what the paper calls Hamiltonian games. Hamiltonian games are similar to energy conserving physical systems that constitute a limit cycle in the gradient field. The direction of the first order gradient is tangent to this limit cycle, therefore, we can't really reduce the loss. for this class of games, the paper suggests Synthetic Gradient Averaging (SGA), that is a transformation on the gradient to map it to the direction perpendicular to the limit cycle. This gradient has similarities to second order and natural gradient methods that map the gradient from a euclidean space to a hamiltonian space. The paper suggests to move in the direction of $$\epsilon + \lambda A^T \epsilon$$, where $$\epsilon$$ is SGD gradient, $$A^T$$ is the anti-symmetric part of the Hessian matrix. 
	+ the third class of games are general games, where we don't have only a potential game or a Hamiltonian game but a mixture of both. In these situations, my physical intuition is that physical system containing all interacting models will be a non-energy conserving system dissipating or adding energy. Therefore, we won't have limit-cycle-like balances in the total energy gradient field landscape and one of the interacting systems may dominate the others. GANs are usually systems of this kind and we usually see one of the involved systems dominate the other, therefore, achieving a balanced minimum is usually hard.  