---
layout: article
title: A unifying view of machine learning, model based 
comments: true
image:
  teaser: jupyter-main-logo.svg
---
  


- Parametric models assume a finite set of parameters can explain data. Therfore, given the model parameters, future predictions x, are independant of the observed data.

- Non-parametric models assume that the data distribution cannot be defined in terms of such a finite set of parameters. But they can often be defined by assuming an infinite dimensional parameter vector. Usually we think of infinte parameters as a function. Therfore, the amount of information stored in a nonparametric model grows with more data. 

- Parametric / Non-parametric -> Application
- polynomial regression / Gaussian processes -> function approx.
- logistic regression / Gaussian process classifiers -> classification
- mixture models, k-means / Dirichlet process mixtures -> clustering
- hidden Markov models / infinite HMMs -> time series
- factor analysis, pPCA, PMF / infinite latent factor models -> feature discovery


- Consider the problem of nonlinear regression: You want to learn a function f with error bars from data D = {X, y}. A Gaussian process defines a distribution over functions p(f) which can be used for Bayesian regression: p(f|D) = p(f)p(D|f)/p(D)

- A multilayer perceptron (neural network) with infinitely many hidden units and Gaussian priors on the weights (bayesian neural net) is a GP (Neal, 1996)


-In a Gaussian process, each data point is considered to be a random variable. We form a similarity matrix between all data points which is used as the covariance matrix for a multivariate Gaussian (uses negative exponensial of Euclidean distance). Since the joint distribution of data is a multivariate Gaussian, assuming that the joint distribution of test data and training data are also a multivariate Gaussian, prediction will be conditional probability of test data given training data from a multivariate Gaussian joint. In a multivariate Gaussian, joint and marginal probabilities can be analytically calculated using linear algebra on the covariance matrix. Therefore, prediction consists of simply performing linear algebra on covariance matrix (similarity matrix) of training data. Note that the choice of the distance measure (i.e. negative exponensial of Euclidean distance) is the modelling prior in a regression problem (e.g. if a linear distance is chosen, then it's linear regression!).


- Other variants of non-parametric models are 1) nearest neighbor regression, where the model would simply store all (x,y) training pairs and then just return an extrapolation of the nearest neighbor y values for each x in the test set. AS another variant, 2) we can simply wrap a parametric model inside a bigger model that scales with training data size; for example in a generalized linear model, we can simply increase the degree of polynomial by extending the data matrix X and number of parameters. 



# Conditional Random Fields
CRFs are a type of discriminative undirected probabilistic graphical model (a type of Markov network). It is used to encode known relationships between observations. a CRF can take context into account; e.g. predict sequences of labels for sequences of input samples. Probability of a sequence of labels, y, given a sequence of inputs, X is written as the normalized product of factors(factors look at a subset of y and X that are dependant and tell us how much they like their association effectively making othe parts of y and X independant of those included in that factor). 

The factor is often parameterized using exponentials (to convert the product to a sum) of parameters times features. Features are also called sufficient statistics since the whole log P(y/X) of the dataset can be written as a linear combination of features with parameters as coefficients. 

## Linear Chain CRF
The assumption in a linear chaing CRF is that each sequence observation is independant from other sequence observations. Therefore, the joint probability of all dataset will be the product of conditional probabilities of observations representing each sequence observation by a factor. 

An example suppose we have a feedforward net that does optical character recognition. If we use softmax in the output, with the feedforward assumption that all observations are independant, the total P(y/x) will be a product of p_k(y/X) for k observations P(y/X)=product(p_k(y/x)). We know that the character in the current time step is dependant on the one in previous step. Therefore, a simple CRF to model this interdependancy is to write the total P(y/X) using linear chain CRF as a product of p_k(y/X) at observation times the p_k+1(y/x) of the next observation P(y/X)=product(p_k(y/x)p_k+1(y/x)). Using the factor parameterization mentioned above, the product of exponentials will be converted to exponential of sum of factors. A sum for the net output and a sum for dependancy on the previous output. The network weights are shared among k and k+1. Therefore, we'll end up with a type of recursive NN.

In linear chain CRFs, four types of factors can be considered. 1- Factors of current labels and previous inputs, 2- Factors of current labels and current inputs, 3-Factors of current labels and future inputs, and 4- Factors of current labels and future labels.





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



## Probabilistic graphical models

There are a couple of key ideas:

1. In statistical machine learning, we represent data as random variables and analyze data points via their joint distribution. The joint probability density is the key quantity that we are looking for to do future predictions, answer questions about data, etc. This joint probability density is unknown and complex. To estimate it, we use modelling by assuming that a mixture structure of some hidden variables (model) can explain the data. 

2. PGMs give us the tools to represent this modelling effort in a better way using hidden and observable nodes of a graph. Calculating all the combinations of values of variables in a table is daunting and grows exponentially. We enjoy big savings when each data point only depends on a couple of parents. Working with joint distributions is, in general, intractable. Probabilistic graphical models represent a pictorial way of expressing how the joint is factored into the product of distributions over smaller subsets of variables. 

3. We represent conditional independence properties of a Bayes net in a graph. Using following tree cases, we can claim that a node in a very large graph is independant of every other node in the graph if it knows its parents, partners, and children. We can therefore do local computations and propagate local answers to get the joint of the whole graph. 

- Consider a little sequence (x->y->z), since y is the only parent of z, then z is independant of x given y. Given the parent, the child is independant of the grand-parent. 

- Consider a little tree (x<-y->z), this kind of tree structures appear in latent variable models. x and z are dependant but conditioned on y they are independant. x and z are siblings, given their parent, they are independant.

- Explaining away: Consider an inverse tree (x->y<-z), in this structure, we know that x and z are independant but x|y and z|y might not be. Here x and z are partners. Given their child y, they are NOT independant.

Bayes Ball Algorithm helps determine whether a given conditional statement is true such as "are XA and XB independant given XC?". There are ten main rules for how the balls (messages) should pass in a probabilistic graph that need to be memorized, the above three cases show the reasoning behind 6 of the rules. The algorithms follows:

- Shade nodes, XC, that are conditioned on (or observed).
- If the ball cannot reach XB, then the nodes XA and XB must be conditionally independent.
- If the ball can reach XB, then the nodes XA and XB are not necessarily independent.


5. The Hammersley-Clifford theorem says that two families of joints are the same; one obtained by checking conditional independencies and the other obtained by varying local probability tables. So by checking conditional independencies for all nodes of a graph we examine the family of joint distributions instead of the varying local probability tables and enjoy great savings.


6. Graphical models also give us inferential machinery for computing probabilistic quantities and answering questions about the joint, i.e., the graph. Graphical models make it possible to use the same algorithms to treat discrete / categorical variables similarly to continuous variables (traditionally, these two have been seperate fields). In a statistical model, the data are represented as shaded nodes in a graphical model. We perform probabilistic inference of the unshaded nodes. Finding the conditional probability of one set of nodes (hidden nodes) given another set (observation data nodes) i.e. p(z|x) is called inference. We can solve it using Bayes theorem which amounts to three computations. 

- Marginalize out all other random variables except the two sets z and x to obtain p(x,z) 
- Marginalize out z variables to get p(x). 
- Compute the ration to obtain the conditional p(z|x)=p(x,z)/p(x)

These marginalizations are interactable since it involves summing over combinations of all values of all variables. Conditional independence in a graph make it tractable using the property that each node is only locally dependant on only its parents, partners and children. Therefore, we can do computations locally and propagate them through the network using message passing to calculate the joint. 

The algorithms for calculating marginals is called elimination algorithm which involves calculating the marginal sums by factorizing repeated multiplications. Instead of doing nested sums on a multiplication of all probabilities (exponential time), we do a sequence of sums (polynomial time) by taking out of the sum (i.e. factorize) the probabilities that don't depend on variables being summed. This is essentially dynamic programming (memoization) where we traded off polynomial time for more storage space. We do a loop (sum), store results then do another loop which involves the memoized calculation. This is an exact algorithm; for very large graphs we might want to do approximate inference using gibbs sampling, variational inference, etc. 

Additionally, we can do this locally since we know about the conditional independance in the graph using an iterative algorithm called message passing. If we have seperated segments of a graph that form cliques, we can marginalize (sum) everything out but the connection of the clique (factor) with a seperator node. The marginalized probability is called a message and the connecting edge is the one along which we send and recieve messages. The idea behind message passing is very similar to backprop in a neural network and uses the EM algorithm i.e. a two stage alternation between optimizations when we have two sets of nodes that we want to optimize. In the E step we cling on the data to the observed nodes, calculate the beliefs (marginalize everything except for the connecting factor) of the nodes and propagate them till we generate values for the hidden set (starting with radom parameters). In the M step, we attach the generated values to hidden set and optimize the parameters using a backprop like algorithm. We continue until convergence.

- Forward pass: A node recieves a message from all its edges, locally calculates its own belief (probability) using sum-product algorithm, and sends its belief through an edge. This is the forward pass. After all nodes go through the forward pass, the messages backpropagate from the end of the chain to nodes in the backward pass to update the beliefs of all nodes. 

- Backward pass is the exact reverse of what happens in the Forward stage. Starting at the end, a node recieves a message through an edge (the same one it used to send message in forward pass), and it propagates it back all its edges that it used in the first stage for making a message. 





#

Factor graphs are a type of PGM that consist of circular nodes representing random variables, square nodes for the conditional probability distributions (factors), and vertices for conditional dependencies between nodes.
In a Bayesian setting, a ‘model’ consists of a specification of the joint distribution over all of the random variables in the problem $P(x1,x2,...,xk)$. 


Consider a general distribution over three variables a, b and c.  this can be factorized as:

$p(a,b,c)= p(c|a,b)p(b|a)p(a)$

Note that we have not yet specified whether these variables are continuous or discrete, nor have we specified the functional form of the various factors, so it is very general. A probabilistic graphical model shows this factorization by a graph using nodes as random variables (i.e. a, b, c) and directed arrows connecting nodes indicating relationships (e.g. p(b|a)= a->b). This graph is acyclic meaning that there are no cycles from a node to itself.  The structure of the graph captures our assumptions about the plausible class of distributions that could be relevant to our application. 

From generative viewpoint, we draw a sample at each of the nodes in order, using the probability distribution at that node. This process continues sequentially until we have a sampled value for each of the variables. 

So far we have assumed that the structure of the graph is determined by the user. In practice, there may be some uncertainty over the graph structure, for example, whether particular links should be present or not, and so there is interest in being able to determine such structure from data. A powerful graphical technique to help with this is called gates [15], which allows random variables to switch between alternative graph structures, thereby introducing a higher-level graph that implicitly includes multiple underlying graph structures. Running inference on the gated graph then gives posterior distributions over different structures, conditioned on the observed data.

As we have seen, a probabilistic model defines a joint distribution over all of the variables in our application. We can partition these variables into those that are observed x (the data), those whose value we wish to know z, and the remaining latent variables w. The joint distribution can therefore be written as p(x,z,w). 

Once the observed variables in the model are fixed to their observed values, initially assumed probability distributions (i.e. priors) are updated using the Bayes’ theorem.

[1] Daffne Koller's book