---
layout: article
title: A primer on neural network learning rules
comments: true
image:
  teaser: jupyter-main-logo.svg
---

How do we neurons in the brain learn? 

- Hebbian rule: The weight connecting two neurons should be increased or decreased according to the inner product of their activations. This rule works well when the inputs are orthogonal and uncorrelated but not so well otherwise!

- Delta rule: The discrepency between the desired and output will drive the change in the weights. This rule leads to finding precise location of a local optimum. Stochastic gradient descent and backpropagation implement this rule. 

- Boltzman Machines: Boltzmann machines can be seen as the stochastic, generative counterpart of Hopfield nets. They are theoretically intriguing because of the locality and Hebbian nature of their training algorithm, and because of their parallelism and the resemblance of their dynamics to simple physical processes. 

Argues for a global search over a large solution space. 

A technique used for this purpose is to approximate this global optimum using a heuristic based on the anealing processing of metals where we warm them up (increase the tempreture and the thermodynamic energy) and then let them cool down which leads to growth of crystals (i.e. a lower energy state than initial state). Simulated anealing is implemented by a random exploration of the solution space with slowly decreasig probability of making a transition from one state to the next. This probability depends on the free energy of the network at each state and a time-varying parameter called tempreture. 


- Variatonal inference turns the problem of inference into optimization. It started in 80s, by bringing ideas like mean field from statistical physics to probabilistic methods. In the 90s, Jordan et al developed it further to generalize to more inference problems and in parallel Hinton et al developed mean field for neural nets and connected it to EM which lead to VI for HMMs and Mixtures. Modern VI touches many areas like, probabilistic programming, neural nets, reinforcement learning, convex optimization, bayesian statistics and many applications. Basic idea is to transform the integral into an expectation over a simple, known distribution using the variational approximate (ELBO). Writing the log-likelihood for the evidence $log p(x)=log \int p(x,z)$ and introducing the variational approximate $log p(x)= log \int p(x,z)*q(z|x)/q(z|x)$, we can move the $log$ inside the integral using Jensen's inequality and use expectation on q(z|x) instead of the integral to obtain the ELBO: $p(x)> E[log p(x|z)] - KL[q(z|x)||p(z)]$ 

- Adversarial learning, (i.e. density ratio estimation), is a method for fitting a goo onto a surface. This fitting happens iteratively in two phases. The classifier first defines a surface to tell the goo and the desired surface apart. Then the goo(transformation function) adjusts itself to breach the classifier surface as much as it can and therefore reshaping itself closer to the desired surface. Then the classifier tries again to tell the goo and the real surface apart capturing more intricacies in the difference of the surface and the goo. The goo then again tries to reshape itself to be able to reshape itself as much as it can in the form of the surface to fool the classifier. This goes on until the goo matches the surface very closely! The cool thing is that both the goo and real surface are intractable and yet we are still able to find a mapping function from a simple noise to a close surrogate of the surface (i.e. matched goo)!

- equilibrium prop workflow is like EM. (1) Clamp the input of the system to some input value. (2) Let the system converge until there is a stable predicted output. (3) Measure some stats within the system. (4) Clamp the output to the true output value. (5) Let the system converge again. (6) Measure the same stats as before. (7) Update the system’s parameters based on the difference in stats. / Adversarial learning has parallels to EM and the equilibrium prop. 