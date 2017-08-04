---
layout: article
title: A unifying view of machine learning, model based 
comments: true
image:
  teaser: jupyter-main-logo.svg
---
  
# Model-based Machine Learning

Over the past two decades, researchers in Machine learning have been seeking to develop a set of unifying  tools for devising, analyzing, and implementing probabilistic models in generality. Probabilistic graphical models, a marriage of graph theory and probability theory, is the result. Graphical models provide a language for expressing assumptions about data, and a suite of efficient algorithms for reasoning and computing with those assumptions. As a onsequence, graphical models research has forged connections between signal processing, coding theory, computational biology, natural language processing, computer vision, and many other fields. With graphical models, we can clearly articulate what kinds of hidden structures are governing the data and construct complex models from simpler components—like clusters, sequences, hierarchies, and others—to tailor our models to the data at hand. 

- There are different types of uncertainties in modeling data i.e. (1) measurement noise, (2) uncertainty about parameter values for optimal prediction (3) model structure uncertainty. Probability theory provides a framework and mathematical tool for modelling uncertainty. Probability distributions are used to represent all uncertainties in a model. Learning from data occurs through the transformation of the prior beliefs (probability distributions), into posterior bliefs after observing data. This is called Bayesian learning. Simple probability distributions in the form of random variables in a model are composed to form more complex distributions. The dominant paradigm in machine learning over the past two decades for representing such compositional probabilistic models has been graphical models.

- Building and using probabilistic models is part of an iterative process for solving data analysis problems. First, formulate a simple model based on the kinds of hidden structure that you believe exists in the data. Then, given a data set, use an inference algorithm to approximate the posterior—the conditional distribution of the hidden variables given the data—which points to the particular hidden pattens that your data exhibits. Finally, use the posterior to test the model against the data, identifying the important ways that it succeeds and fails. If satisfied, use the model to solve the problem; if not satisfied, revise the model according to the results of the criticism and repeat the cycle. 

1. Describe the Model: Describe the process that generated the data using factor graphs.
2. Condition on Observed Data: Condition the observed variables to their known quantities.
3. Perform Inference: Perform backward reasoning to update the prior distribution over the latent variables or parameters. In other words, calculate the posterior probability distributions of latent variables conditioned on observed variables.
4. Criticize model predictions and go to step 1 if not satisfied. 


- PGMs make it very easy to customize our modeling effort based on the problem at hand. Many machine learning techniques are special cases a graphical model, coupled with an inference procedure, for example, principal component analysis (PCA), factor analysis, logistic regression, Gaussian mixtures. Even more interestingly, even these models can be composed to form a mixture of probabilistic models.

## Bayesian belief system
- There are two simple rules that underlie probability theory: the sum rule, $$P(x)=\sum_y P(x,y)$$, and the product rule $$P(x,y)=P(x)P(y|x)$$. The sum rule states that the marginal of x is obtained by summing (or integrating for continuous variables) the joint over y. The product rule states that the joint can be decomposed as the product of the marginal and the conditional. Bayes rule is a corollary of these two rules, $$P(y|x)=\frac{P(x|y)P(y)}{P(x)}. We can apply probability theory to machine learning by replacing the symbols above: we replace x by D to denote the observed data, we replace y by $$\theta$$ to denote the unknown parameters of a model, and we condition all terms on m, the class of probabilistic models we are considering.

- Suppose that there is some unknown probability distribution over the data, $$p(x)$$, that we want to know. The way we approach finding this unknown distribution is modeling. Let's say we think the data is coming from a Guassian distribution. So our distribution, $$p(x)$$ is now a function of a latent variable that has a Gaussian distribution $$p(z)=N(u_z, \sigma_z)$$. To generate samples we need $$p(x|z)=probability_shape(parameters)$$. The posterior is the probability of the latents given data or p(z|x)=probability_shape(params)

- According to the bayesian rule, p(z|x)=p(x|z)p(z)/p(x). but we don't know the shape and parameters of the probability distribution for the posterior, likelihood or the marginal. We can approximate the shape of the posterior with an exponential family i.e. Gaussian and focus on finding its parameters. If the prior and posterior are Gaussian, then the likelihood would be Gaussian as well. 

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

- Forward pass: A node recieves a message from all its edges, locally calculates its own belief (probability) using sum-product rules, and sends its belief through an edge. This is the forward pass. After all nodes go through the forward pass, the messages backpropagate from the end of the chain to nodes in the backward pass to update the beliefs of all nodes. 

- Backward pass is the exact reverse of what happens in the Forward stage. Starting at the end, a node recieves a message through an edge (the same one it used to send message in forward pass), and it propagates it back all its edges that it used in the first stage for making a message. 

## factor graph

Factor graphs are a type of PGM that consist of circular nodes representing random variables, square nodes for the conditional probability distributions (factors), and vertices for conditional dependencies between nodes.
In a Bayesian setting, a ‘model’ consists of a specification of the joint distribution over all of the random variables in the problem $P(x1,x2,...,xk)$. 


Consider a general distribution over three variables a, b and c.  this can be factorized as:

$p(a,b,c)= p(c|a,b)p(b|a)p(a)$

Note that we have not yet specified whether these variables are continuous or discrete, nor have we specified the functional form of the various factors, so it is very general. A probabilistic graphical model shows this factorization by a graph using nodes as random variables (i.e. a, b, c) and directed arrows connecting nodes indicating relationships (e.g. p(b|a)= a->b). This graph is acyclic meaning that there are no cycles from a node to itself.  The structure of the graph captures our assumptions about the plausible class of distributions that could be relevant to our application. 

From generative viewpoint, we draw a sample at each of the nodes in order, using the probability distribution at that node. This process continues sequentially until we have a sampled value for each of the variables. 

So far we have assumed that the structure of the graph is determined by the user. In practice, there may be some uncertainty over the graph structure, for example, whether particular links should be present or not, and so there is interest in being able to determine such structure from data. A powerful graphical technique to help with this is called gates [15], which allows random variables to switch between alternative graph structures, thereby introducing a higher-level graph that implicitly includes multiple underlying graph structures. Running inference on the gated graph then gives posterior distributions over different structures, conditioned on the observed data.

As we have seen, a probabilistic model defines a joint distribution over all of the variables in our application. We can partition these variables into those that are observed x (the data), those whose value we wish to know z, and the remaining latent variables w. The joint distribution can therefore be written as p(x,z,w). 

Once the observed variables in the model are fixed to their observed values, initially assumed probability distributions (i.e. priors) are updated using the Bayes’ theorem.



# Probablistic Programming

- Probabilistic programming provides a simulation framework and the ability to generalize graphical models to allow constructs such as recursion (functions calling themselves) and control flow statements. (1) We can define a more dynamic probabilistic model through a computer program instead of a fixed graph. (2) We have access to a simulation environment where sampling from the models we define is easy. The simulator (model) makes calls to a random number generator in such a way that repeated runs from the simulator would sample different possible data sets from the model. (3) Using the simulation environment, we can automate inference through a black-box inference engine based on Monte Carlo sampling. 

- Probabilistic programming could prove to be revolutionary. First, the universal inference engine obviates the need to manually derive inference methods for models. Second, probabilistic programming could be potentially transformative for the sciences, since it allows for rapid prototyping and testing of different models of data. Third, Probabilistic programming languages create a very clear separation between the model and the inference procedures, encouraging model-based thinking. 


- Five areas in probabilistic machine learning (1) probabilistic programming, which is a general framework for expressing probabilistic models as computer programs and which could have a major impact on scientific modelling; (2)Bayesian optimization, which is an approach to globally optimizing unknown functions; (3) probabilistic data compression; (4) automating the discovery of plausible and interpretable models from data; and (5) hierarchical modelling for learning many related models, for example for personalized medicine or recommendation.


- Almost all machine-learning tasks can be formulated as making inferences about missing or latent data from the observed data. The pipeline is (1) modelling: A model can be very simple and rigid, such as a classic statistical linear regression model, or complex and flexible, such as a large and deep neural network, or even a model with infinitely many parameters. Since any sensible model will be uncertain when predicting unobserved data, uncertainty plays a fundamental part in modelling.


- Although conceptually simple, a fully probabilistic approach to machine learning poses a number of computational and modelling challenges. Computationally, the main challenge is that learning involves marginalizing (summing out) all the variables in the model except for the variables of interest. Such high-dimensional sums and integrals are generally computationally hard, in the sense that for many models there is no known polynomial time algorithm for performing them exactly. Fortunately, a number of approximate integration algorithms have been developed, including Markov chain Monte Carlo (MCMC) methods, variational approximations, expectation propagation and sequential Monte Carlo. Moreover, these algorithms are modular—recurring components in a graphical model lead to recurring subroutines in their corresponding inference algorithms.

- computational techniques are one area in which Bayesian machine learning differs from much of the rest of machine learning: for Bayesian researchers the main computational problem is integration, whereas for much of the rest of the community the focus is on optimization of model parameters. However, this dichotomy is not as stark as it appears: many gradient-based optimization methods can be turned into integration methods through the use of Langevin and Hamiltonian Monte Carlo methods, while integration problems can be turned into optimization problems through the use of variational approximations. 

- The main modelling challenge for probabilistic machine learning is that the model should be flexible enough to capture all the properties of the data required to achieve the prediction task of interest. One approach to addressing this challenge is to develop a prior distribution that encompasses an open-ended universe of models that can adapt in complexity to the data. There are essentially two ways of achieving flexibility. The model could have a large number of parameters compared with the data set (for example, neural networks). Alternatively, the model can be defined using non-parametric components. The key statistical concept underlying flexible models that grow in complexity with the data is non-parametrics.

- In a parametric model, there are a fixed, finite number of parameters, and no matter how much training data are observed, all the data can do is set these finitely many parameters that control future predictions. By contrast, non-parametric approaches have predictions that grow in complexity with the amount of training data, either by considering a nested sequence of parametric models with increasing numbers of parameters or by starting out with a model with infinitely many parameters. Both parametric and non-parametric components should be thought of as building blocks, which can be composed into more complex models.  



References:
[1]. Bishop, Christopher M. "Model-based machine learning." Phil. Trans. R. Soc. A 371.1984 (2013): 20120222.
[2]. Ghahramani, Zoubin. "Probabilistic machine learning and artificial intelligence." Nature 521.7553 (2015): 452-459.




# Non-Parametric Models
- Parametric models assume a finite set of parameters can explain data. Therfore, given the model parameters, future predictions x, are independant of the observed data.

- Non-parametric models assume that the data distribution cannot be defined in terms of such a finite set of parameters. But they can often be defined by assuming an infinite dimensional parameter vector. Usually we think of infinte parameters as a function. Therfore, the amount of information stored in a nonparametric model grows with more data. 

Parametric | Non-parametric | -> Application
polynomial regression | Gaussian processes | -> function approx.
logistic regression | Gaussian process classifiers | -> classification
mixture models, k-means \ Dirichlet process mixtures | -> clustering
hidden Markov models | infinite HMMs | -> time series
factor analysis, probabilistic PCA, PMF | infinite latent factor models | -> feature discovery


- Consider the problem of nonlinear regression: You want to learn a function f with error bars from data D = {X, y}. A Gaussian process defines a distribution over functions p(f) which can be used for Bayesian regression: p(f|D) = p(f)p(D|f)/p(D)

- A multilayer perceptron (neural network) with infinitely many hidden units and Gaussian priors on the weights (bayesian neural net) is a GP (Neal, 1996)


-In a Gaussian process, each data point is considered to be a random variable. We form a similarity matrix between all data points which is used as the covariance matrix for a multivariate Gaussian (uses negative exponensial of Euclidean distance). Since the joint distribution of data is a multivariate Gaussian, assuming that the joint distribution of test data and training data are also a multivariate Gaussian, prediction will be conditional probability of test data given training data from a multivariate Gaussian joint. In a multivariate Gaussian, joint and marginal probabilities can be analytically calculated using linear algebra on the covariance matrix. Therefore, prediction consists of simply performing linear algebra on covariance matrix (similarity matrix) of training data. Note that the choice of the distance measure (i.e. negative exponensial of Euclidean distance) is the modelling prior in a regression problem (e.g. if a linear distance is chosen, then it's linear regression!).

- Other variants of non-parametric models are 1) nearest neighbor regression, where the model would simply store all (x,y) training pairs and then just return an extrapolation of the nearest neighbor y values for each x in the test set. AS another variant, 2) we can simply wrap a parametric model inside a bigger model that scales with training data size; for example in a generalized linear model, we can simply increase the degree of polynomial by extending the data matrix X and number of parameters. 

- Gaussian processes are a very flexible non-parametric model for unknown functions, and are widely used for regression, classification, and many other applications that require inference on functions. Dirichlet processes are a non-parametric model with a long history in statistics and are used for density estimation, clustering, time-series analysis and modelling the topics of documents. To illustrate Dirichlet processes, consider an application to modelling friendships in a social network, where each person can belong to one of many communities. A Dirichlet process makes it possible to have a model whereby the number of inferred communities (that is, clusters) grows with the number of people. The Indian buffet process (IBP)40 is a non-parametric model that can be used for latent feature modelling, learning overlapping clusters, sparse matrix factorization, or to non-parametrically learn the structure of a deep network41. Elaborating the social network modelling example, an IBP-based model allows each person to belong to some subset of a large number of potential communities (for example, as defined by different families, workplaces, schools, hobbies, and so on) rather than a single community, and the probability of friendship between two people depends on the number of overlapping communities they have42. In this case, the latent features of each person correspond to the communities, which are not assumed to be observed directly. The IBP can be thought of as a way of endowing Bayesian non-parametric models with ‘distributed representations’, as popularized in the neural network literature. An interesting link between Bayesian non-parametrics and neural networks is that, under fairly general conditions, a neural network with infinitely many hidden units is equivalent to a Gaussian process.

# Conditional Random Fields
CRFs are a type of discriminative undirected probabilistic graphical model (a type of Markov network). It is used to encode known relationships between observations. a CRF can take context into account; e.g. predict sequences of labels for sequences of input samples. Probability of a sequence of labels, y, given a sequence of inputs, X is written as the normalized product of factors(factors look at a subset of y and X that are dependant and tell us how much they like their association effectively making othe parts of y and X independant of those included in that factor). 

The factor is often parameterized using exponentials (to convert the product to a sum) of parameters times features. Features are also called sufficient statistics since the whole log P(y/X) of the dataset can be written as a linear combination of features with parameters as coefficients. 

## Linear Chain CRF
The assumption in a linear chaing CRF is that each sequence observation is independant from other sequence observations. Therefore, the joint probability of all dataset will be the product of conditional probabilities of observations representing each sequence observation by a factor. 

An example suppose we have a feedforward net that does optical character recognition. If we use softmax in the output, with the feedforward assumption that all observations are independant, the total P(y/x) will be a product of p_k(y/X) for k observations P(y/X)=product(p_k(y/x)). We know that the character in the current time step is dependant on the one in previous step. Therefore, a simple CRF to model this interdependancy is to write the total P(y/X) using linear chain CRF as a product of p_k(y/X) at observation times the p_k+1(y/x) of the next observation P(y/X)=product(p_k(y/x)p_k+1(y/x)). Using the factor parameterization mentioned above, the product of exponentials will be converted to exponential of sum of factors. A sum for the net output and a sum for dependancy on the previous output. The network weights are shared among k and k+1. Therefore, we'll end up with a type of recursive NN.

In linear chain CRFs, four types of factors can be considered. 1- Factors of current labels and previous inputs, 2- Factors of current labels and current inputs, 3-Factors of current labels and future inputs, and 4- Factors of current labels and future labels.