---
layout: article
title: A unifying view of machine learning, model based 
comments: true
image:
  teaser: jupyter-main-logo.svg
---
  
# DL book 
Instead of using a single function to represent a probability distribution, we can split a probability distribution into many factors that we multiply together. These factorizations can greatly reduce the number of parameters needed to describe the distribution. Each factor uses a number of parameters that is exponential in the number of variables in the factor. This means that we can greatly reduce the cost of representing a distribution if we are able to find a factorization into distributions over fewer variables. We can describe these kinds of factorizations using graphs. 

directed model contains one factor for every random variable xi in the distribution, and that factor consists of the conditional distribution over xi given the parents. Undirected models use graphs with undirected edges, and they represent factorizations into a set of functions; unlike in the directed case, these functions are usually not probability distributions of any kind. Any set of nodes that are all connected to each other is called a clique and each clique is associated with a factor.The output of each factor must be non-negative, but there is no constraint that the factor must sum or integrate to 1 like a probability distribution. The probability of a configuration of random variables is proportional to the product of all of these factors.there is no guarantee that this product will sum to 1. We therefore divide by a normalizing constant Z, defined to be the sum or integral over all states of the product of the factor functions to get a normalized probability distribution.

Graphical representations of factorizations are not mutually exclusive families of probability distributions. Being directed or undirected is not a property of a probability distribution; it is a property of a particular representation of a probability distribution, but any probability distribution may be described in directed and undirected mode.



## other
- suppose that there is some unknown probability distribution over the data. p(x)
- How do you generate more data like that? or more concretely, how do you draw from that unknown distribution?

- one way is modeling. Let's say we think the data is coming from a Guassian distribution. so our latent variable is p(z)=Gaussian(u_z, sigma_z).

- generating samples like the data is then the probability of x conditioned on z or p(x|z)=probability_shape(parameters).

- the posterior is the probability of the latents given data or p(z|x)=probability_shape(params)

- According to the bayesian rule, p(z|x)=p(x|z)p(z)/p(x). but we don't know the shape and parameters of the probability distribution for the posterior, likelyhood or the marginal. 

- We can approximate the shape of the posterior with an exponential family i.e. Gaussian and focus on finding its parameters. 

- If the prior and posterior are Gaussian, then the likelyhood would be Gaussian as well. 


Predictions about future data and the future consequences of actions are uncertain. There are many forms of uncertainty in modelling. Uncertainty is introduced from measurement noise, uncertainty about which values of these parameters will be good at predicting new data, and uncertainty about the general structure of the model. The probabilistic approach to modelling uses probability theory to express all forms of uncertainty. 
Almost all machine learning tasks can be formulated as making inferences about missing or latent data from the observed data.probability distributions are used to represent
all the uncertain unobserved quantities in a model (including structural, parametric and noise-related).Then the basic rules of probability theory are used to infer the unobserved quantities given the observed data. Learning from data occurs through the
transformation of the prior probability distributions into posterior. Simple
probability distributions over single or a few variables can be composed to form the building blocks of larger, more complex models. As discussed later, probabilistic
programming offers an elegant way of generalizing graphical models, allowing a much richer representation of models.
Although conceptually simple, a fully probabilistic approach to machine learning poses a number of computational and modelling challenges. Computationally, the main challenge is that learning involves marginalizing (summing out) all the variables in the model except for the variables of interest since there are no known algorithm for calculating them exactly in polynomial time. Fortunately, a number of approximate integration algorithms have been developed, including Markov chain Monte Carlo (MCMC) methods, variational approximations, expectation propagation and sequential Monte Carlo. for Bayesian researchers the main computational problem is integration, whereas for much of the rest of the community the focus is on optimization of model parameters. However, this dichotomy is not as stark as it appears: many gradient-based optimization methods can be turned into integration methods through the use of Langevin and Hamiltonian Monte Carlo methods while integration problems can be turned into optimization problems through the use of variational approximations. Model flexibility can be achieved using either very large models compared to data (e.g. deep networks ) or nonparametric methods. The key statistical concept underlying flexible models that grow in complexity with the data is non-parametrics.
In a parametric model, there are a fixed, finite number of parameters, and no matter how much training data are observed, all the data can do is set these finitely many parameters that control future predictions. By contrast, non-parametric approaches
have predictions that grow in complexity with the amount of training data, either by considering a nested sequence of parametric models with increasing numbers of parameters or by starting out with a model with infinitely many parameters. Many non-parametric models can be derived starting from a parametric model and considering what happens as the model grows to the limit of infinitely many parameters. Bayesian approaches are not prone to overfitting since they average over, rather than fit, the parameters.Gaussian processes are a very flexible non-parametric model for unknown functions, and are widely used for regression, classification, and many other applications that require inference on functions. Dirichlet processes are a non-parametric model with a long history in statistics and are used for density estimation, clustering, time-series analysis and modelling the topics of documents. The Indian buffet process (IBP) is a non-parametric model that can be used for latent feature modelling, learning overlapping clusters, sparse matrix factorization, or to non-parametrically learn the structure of a deep network. The IBP can be thought of as a way of endowing Bayesian non-parametric models with ‘distributed representations’ where a data point can be a member of multiple clusters. An interesting link between Bayesian non-parametrics and neural networks
is that, under fairly general conditions, a neural network with infinitely many hidden units is equivalent to a Gaussian process.
The basic idea in probabilistic programming is to use computer programs to represent probabilistic models (http://probabilistic-programming.org). One way to do this is for the computer program to define a generator for data from the probabilistic model, that is, a simulator. This simulator makes calls to a random number generator in such a way that repeated runs from the simulator would sample different possible data sets from the model. This simulation framework is more general than the graphical model framework described previously since computer programs can allow constructs such as recursion (functions calling themselves) and control flow statements (for example, ‘if ’ statements
that result in multiple paths a program can follow), which are difficult or impossible to represent in a finite graph. In fact, for many of the recent probabilistic programming languages that are based on extending Turing-complete languages (a class that includes almost all commonly used languages), it is possible to represent any computable probability distribution as a probabilistic program. The full potential of probabilistic programming comes from automating the process of inferring unobserved variables in the model conditioned on the observed data.
There are several reasons why probabilistic programming could prove to be revolutionary for machine intelligence and scientific modelling. First, the universal inference engine obviates the need to manually derive inference methods for models. Second, probabilistic programming could be potentially transformative for the sciences, since it allows for rapid prototyping and testing of different models of data. Probabilistic programming
languages create a very clear separation between the model and the inference procedures, encouraging model-based thinking.


Over the past two decades, scholars working in the field of machine learning have
sought to unify such data analysis activities. Their focus has been on developing tools
for devising, analyzing, and implementing probabilistic models in generality. These
efforts have lead to the body of work on probabilistic graphical models, a marriage
of graph theory and probability theory [directed graphs (also known as Bayesian networks and belief networks), undirected graphs (also known as Markov networks and random fields)]. Graphical models provide a language for expressing assumptions about data, and a suite of efficient algorithms for reasoning and computing with those assumptions. As a onsequence, graphical models research has forged connections between signal processing, coding theory, computational biology, natural language processing, computer vision, and many other fields. With graphical models, we can clearly articulate what kinds of hidden structures are governing the data and construct complex models from simpler components—like clusters, sequences, hierarchies, and others—to tailor our models to the data at hand. 

One of the most powerful aspects of probabilistic graphical models is the relative ease with which a model can be customized to a specific application, or modified if the requirements of the application change. The key point here is that many variants are possible, and in particular a new model can readily be developed that is tailored to each particular application. A high proportion of the standard techniques used in traditional machine learning can be expressed as special cases of the graphical model framework, coupled with appropriate inference algorithms. For example, principal component analysis (PCA), factor analysis, logistic regression, Gaussian mixtures and similar models can all be represented using simple graphical structures. These can then readily be combined, for example, to form a mixture of probabilistic PCA models. To construct and use these models within a model-based machine learning framework, it is not necessary to know their names or be familiar with the specific literature on their properties.

Most useful models are difficult to compute with, however, and researchers have developed powerful approximate posterior inference algorithms for approximating these conditionals. Techniques like Markov chain Monte Carlo (MCMC) (Metropolis et al., 1953; Hastings, 1970; Geman and Geman, 1984) and variational inference (Jordan et al., 1999; Wainwright and Jordan, 2008) make it possible for us to examine large data sets with sophisticated statistical models. Moreover, these algorithms are modular—recurring components in a graphical model lead to recurring subroutines in their corresponding inference algorithms.

Building and using probabilistic models is part of an iterative process for solving data analysis problems. First, formulate a simple model based on the kinds of hidden structure that you believe exists in the data. Then, given a data set, use an inference algorithm to approximate the posterior—the conditional distribution of the hidden variables given the data—which points to the particular hidden pattens that your data exhibits. Finally, use the posterior to test the model against the data, identifying the important ways that it succeeds and fails. If satisfied, use the model to solve the problem; if not satisfied, revise the model according to the results of the criticism and repeat the cycle. 

1. Describe the Model: Describe the process that generated the data using factor graphs.
2. Condition on Observed Data: Condition the observed variables to their known quantities.
3. Perform Inference: Perform backward reasoning to update the prior distribution over the latent variables or parameters. In other words, calculate the posterior probability distributions of latent variables conditioned on observed variables.
4. Criticize model predictions and go to step 1 if not satisfied. 


The graphical model representations aim at characterizing probability distributions in terms of conditional independence statements. Factor graphs, an alternative graphical representation of probability distributions, aim at capturing factorizations. while closely related, conditional independence and factorization are not exactly the same 
concepts. Recall in particular our discussion of the parameterization of the complete graph on three nodes. 



Probabilistic graphical models
Representing all the combinations of atoms of radom variables in a table  exponentially grows with number of atoms and variables. Graphical models provide a more economic representation of the joint distribution by taking advantage of local relationships between random variables. They replace exponential growth in number of variables by exponential growth in parenthood relationships. Graphical models are a way of using graphs to represent and compute about families of probability distributions. The family is all distributions whose joint can be written in terms of the factorization implied by the graph. In a PGM, a node is conditionally independant of its ancestors given its parent. So by using the cheaper factorized representation of the joint, we are making
independence assumptions about the random variables.

In statistical and machine learning applications, we represent data as random variables
and analyze data via their joint distribution. We enjoy big savings when each data
point only depends on a couple of parents. Additionally, graphical models also give us inferential machinery for computing probabilistic quantities and answering questions about the joint, i.e., the graph. Finally, graphs are more generic than specific joint distributions. Graphical models make it possible to use the same algorithms to treat discrete / categorical variables similarly to continuous variables (traditionally, these two have been seperate fields). 

D-seperation idea is to associate "dependence" with "connectedness" in the graph(i.e., the existence of a connecting path) and "independence" with "unconnected-ness" or "separation". D-separation gives a way of determining conditional independence properties of a Bayes net from the graphical representation, but unfortunately the definition itself doesn't give a practical algorithm. Bayes ball is an efficient algorithm for computing d-separation by passing simple messages between nodes of the graph. The name "Bayes Ball" stems from the idea of balls bouncing around a directed graph, where if a ball cannot bounce between two nodes then they are [conditionally] independent. It is important to note that the balls are allowed to travel in any direction, independent of the direction of the edges in the graph. There are ten main rules for how the messages should pass that need to be memorized, the following three cases show the reasoning behind 6 of the rules. 

Consider a little sequence (x->y->z), since y is the only parent of z, then z is independant of x given y. In other words, X is the “past”, Y is the “present”, Z is the “future”. Given the present, the past is independent of the future. This is the Markov assumption and this graph is a three step Markov chain.

Consider a little tree (x<-y->z), this kind of tree structures appear in latent variable models. x and z are dependant but conditioned on y they are independant. 

Consider an inverse tree (x->y<-z), in this structure, we know that x and z are independant but x|y and z|y might not be. For example, if y is "I am late", x is "didn't wake up on time", and z is "I was abducted"; I am late given and you know that I didn't wake up on time x|z reduces probability that I was abducted z|y. 

Bayes Ball Algorithm (message passing):
Goal: We wish to determine whether a given conditional statement such as XA q XB | XC
is true given a directed graph.
The algorithm is as follows:
1. Shade nodes, XC, that are conditioned on (or observed).
2. If the ball cannot reach XB, then the nodes XA and XB must be conditionally independent.
3. If the ball can reach XB, then the nodes XA and XB are not necessarily independent.

The Hammersley-Clifford theorem says that these two families of joints—one obtained
by checking conditional independencies and the other obtained by varying
local probability tables—are the same. So by checking conditional independencies for all nodes of a graph we examine the family of joint distributions instead of the varying local probability tables. 

How can we use joint probability distributions to make meaningful statements about observed data?

(A) We build a statistical model by developing a joint distribution of hidden and observed random variables. Our data are observed variables and the hidden quantities that
govern them are hidden variables. In addition to the joint, a statistical model can be
described by its graphical model or its generative process.

(B) In a statistical model, the data are represented as shaded nodes in a graphical
model that also involves unshaded nodes. We perform probabilistic inference of
the unshaded nodes. This inference tells us about the data and helps form predictions.



# Inference 
Inference is the problem of calculating the conditional probability of one set of nodes (hidden/latent nodes) versus another set (observable/data nodes) i.e. p(z|x). It amounts to three computations. 
1. Marginalize out all other random variables except the two sets z and x to obtain p(x,z) 
2. Marginalize out z variables to get p(x). 
3. Compute the ration to obtain the conditional p(z|x)=p(x,z)/p(x)

Suppose we want to marginalize out r nodes and each takes k values. Then that's a sum over r^k configs which is interactable. So marginalization is difficult. 


- the idea behind message passing seems to be very similar to what happens in a neural network. A node sends message (fires) through an edge when it has recieved a message from all its other edges. This is the forward pass. After all nodes go through the forward pass, the messages backpropagate to nodes in the backward pass. This is the exact reverse of what happens in the first stage. A node sends its message through one edge, and when it recieves a message through that edge in the backward pass, it propagates it back to its neighboring edges that it used in the first stage for making a message. 

- the idea behind EM algorithm seems to be a two stage alternation between optimizations. When we have two sets of nodes that we want to optimize, in the E step we assume the first set and optimize the second step. In the M step, we assume the generated values in the E step and optimize the second set of nodes. 

- 











#

Factor graphs are a type of PGM that consist of circular nodes representing random variables, square nodes for the conditional probability distributions (factors), and vertices for conditional dependencies between nodes.
In a Bayesian setting, a ‘model’ consists of a specification of the joint distribution over all of the random variables in the problem $P(x1,x2,...,xk)$. 

where {x1,…,xK} includes any ‘parameters’ in the model as well as any latent (i.e. hidden) variables, along with the variables whose values are to be observed or predicted. Working with fully flexible joint distributions is, in general, intractable, and inevitably we must deal with structured models. Probabilistic graphical models represent a pictorial way of expressing how the joint distribution is factored into the product of distributions over smaller subsets of variables.

Consider a general distribution over three variables a, b and c.  this can be factorized as:

$p(a,b,c)= p(c|a,b)p(b|a)p(a)$

Note that we have not yet specified whether these variables are continuous or discrete, nor have we specified the functional form of the various factors, so it is very general. A probabilistic graphical model shows this factorization by a graph using nodes as random variables (i.e. a, b, c) and directed arrows connecting nodes indicating relationships (e.g. p(b|a)= a->b). This graph is acyclic meaning that there are no cycles from a node to itself.  The structure of the graph captures our assumptions about the plausible class of distributions that could be relevant to our application. 

From generative viewpoint, we draw a sample at each of the nodes in order, using the probability distribution at that node. This process continues sequentially until we have a sampled value for each of the variables. 

So far we have assumed that the structure of the graph is determined by the user. In practice, there may be some uncertainty over the graph structure, for example, whether particular links should be present or not, and so there is interest in being able to determine such structure from data. A powerful graphical technique to help with this is called gates [15], which allows random variables to switch between alternative graph structures, thereby introducing a higher-level graph that implicitly includes multiple underlying graph structures. Running inference on the gated graph then gives posterior distributions over different structures, conditioned on the observed data.

As we have seen, a probabilistic model defines a joint distribution over all of the variables in our application. We can partition these variables into those that are observed x (the data), those whose value we wish to know z, and the remaining latent variables w. The joint distribution can therefore be written as p(x,z,w). 

Once the observed variables in the model are fixed to their observed values, initially assumed probability distributions (i.e. priors) are updated using the Bayes’ theorem.

[1] Daffne Koller's book