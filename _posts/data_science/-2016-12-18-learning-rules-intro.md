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

