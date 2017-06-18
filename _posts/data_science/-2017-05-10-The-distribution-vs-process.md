---
layout: article
title: Information Geometry
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

- A probability distribution is the reciprocal of energy for a random variable telling how likely the values of that random variable are. You can think of a random variable as a piece of land and the probability distribution as the density of cloud above this land. 

- A random process is a collection of random variables uniquely indexed by some mathematical set. Historically this set has been natural numbers (1,2,..) representing the evolution of some system over time. All random variable takes values from a mathematical space known as the state space. A random process can have many different realizations due to its randomness.

- When a process is associated with a ditribution, it usually means that the random variables in the process each have a probability distribution similar to that specific distribution (state space). If the random variables are indexed by a higher-dimensional Euclidean space, then the collection of random variables is usually called a random field instead.

- Based on their properties, stochastic processes can be divided into various categories, which include random walks, martingales, Markov processes, Lévy processes, Gaussian processes, and random fields as well as renewal processes and branching processes. 

- Bernoulli process, is a sequence of independent and identically distributed (iid) random variables of binary states.
- Random walks are stochastic processes that are usually defined as sums of iid random variables or random vectors in Euclidean space.
- The Wiener process (Brownian motion) is a stochastic process with stationary and independent increments that are normally distributed based on the size of the increments (continuous version of the simple random walk).
- Poisson process can be defined as a counting process, which is a stochastic process that represents the random number of points or events up to some time

- Markov processes are stochastic processes, traditionally in discrete or continuous time, that have the Markov property, which means the next value of the Markov process depends on the current value, but it is conditionally independent of the previous values of the stochastic process. In other words, the behavior of the process in the future is stochastically independent of its behavior in the past, given the current state of the process.The Brownian motion process and the Poisson process (in one dimension) are both examples of Markov processes.
- A martingale is a discrete-time or continuous-time stochastic process with the property that the expectation of the next value of a martingale is equal to the current value given all the previous values of the process. A symmetric random walk and a Wiener process (with zero drift) are both examples of martingales
- Lévy processes are types of stochastic processes that can be considered as generalizations of random walks in continuous time.
- A point process is a collection of points randomly located on some mathematical space such as the real line. 


# Bayesian nonparametrics based on Levy processes

- Lévy processes are types of stochastic processes that can be considered as generalizations of random walks in continuous time. The main defining characteristic of these processes is their stationarity property, so they were known as processes with stationary and independent increments. Increaments are all independent of each other, and the distribution of each increment only depends on the difference in time. Important stochastic processes such as the Wiener process, the homogeneous Poisson process (in one dimension), and subordinators are all Lévy processes.

- Gaussian Processes. A key fact of Gaussian processes is that they can be completely defined by their second-order statistics (Variance). Thus, if a Gaussian process is assumed to have mean zero, defining the covariance function completely defines the process' behaviour. Basic aspects that can be defined through the covariance function are the process' stationarity, isotropy, smoothness and periodicity. Gaussian processes translate as taking priors on functions and the smoothness of these priors can be induced by the covariance function. If we expect that for "near-by" input points x and x' their corresponding output points y and y' to be "near-by" also, then the assumption of continuity is present. If we wish to allow for significant displacement then we might choose a rougher covariance function. A common covariance function might be Gaussian Noise: $K_{\text{GN}}(x,x')=\sigma ^{2}\delta _{x,x'}$ where Here $d=x-x'$. The parameter $l$ is the characteristic length-scale of the process (practically, "how close" two points $x$ and $x'$ have to be to influence each other significantly). $δ$ is the Kronecker delta and $σ$ the standard deviation of the noise fluctuations.



# Bayesian Non-parametric modelling 

- The way we model data with generative models is that we design and assume a generative process that given parameters will be able to simulate our data. We then put some priors on the parameters and we'll be able to simulate new data points using the likelihood function and priors. With generative models it's always a tradeoff when to stop with priors that depends on our model since every parameter can have a prior and the parameters of the priors can also have priors, etc.

- Now using the Bayesian framework, we turn this process on its head and say we instead want to infer parameters given data which will give us the posterior about parameters after seeing data. This means we have fit the model to data by finding the right parameter values.

- We can do the modelling parametrically or non-parametrically. The non-parametric approach is useful for the growing number of clusters in data. For example, consider the problem of species clustering where a new species is discovered everyday although very few of instances of these species are encountered. What does the growing infinite number of parameters in a model mean? It presents the difference between possible component and actual clusters realized. The way we actually implement this infinite number of clusters on a computer is using an on demand draw. Although the number of components are infinite, the number of components realized in the data by random draws are actually finite which makes it possible to do calculations for only the realized part. The inferece in such models is done using approximate inference by truncating the infinite parameters or by integrating out the infinite parameters like the chinese restaurant process. 

- In this context, lets consider a clustering problem, we design a generative model of lets say 2 clusters with parameter set of {global cluster means and shapes} plus {local cluster assignment parameters} for each data point to determine which cluster the data is coming from.

- We now have to choose a prior for the parameter set above to be able to simulate data points. Lets put a Gaussian prior on the clusters $$\mu_k ~ N(\mu_0,\sum_0)$$ with k=2. The parameters of clusters are global since we don't need one parameter per data point. We also need a cluster assignment parameter $$z_n$$ for each data point. This is a local parameter since we need one for each data point. We can use a categorical distribution to choose among distribution $$z_n ~ Categorical(\rho_k)$$, and use a Beta or Drichlet distribution for the parameter $$\rho_k ~ Beta(a,b)$$.

- The Beta distribution breaks a stick of length one to two partitions (two probabilities that sum to one) based on the value of parameters $$a,b$$. A Drichlet distribution consists of successive applications of Beta distribution to further break down the stick. First Beta breaks down the stick to two parts and the next Beta breaks the remainder of the stick to two parts and so on. It's infeasible to set a seperate $$a,b$$ for each successive Beta. If we set all $$a_k=1, b_k=\alpha>0$$, we'll end up with Drichlet process stick-breaking. In fact to be more particular, the explained distributions on infinite $$\rho$$ parameters is called a Griffith-Engen_McColsky (GEM) distribution $$(\rho_1,\rho_2,...,\rho_\inf) ~ GEM(\alpha)$$.

- Therefore, our model for simulating the data is (1) drawing a cluster assingment from GEM, $$\rho=(\rho_1,\rho_2,...,\rho_\inf) ~ GEM(\alpha)$$, and then (2) draw a cluster mean from the Gaussian $$ \mu_k ~ N(\mu_0,\sum_0), k=1,2,..$$. Then (3) we'll have to plug our $$\rho$$ into a categorical distribution to draw per data point cluster assignments $$z_n ~ Categorical(\rho)$$. Now that the cluster is chosen, in step (4) the data point will be around the mean indexed by z_n i.e. $$\mu_n^* = \mu_{z_n}$$. And finally, (5) We draw the data point iid from the Gaussian $$x_n ~ N(\mu_n^*, \sum)$$

- Now if instead of steps 1,2 above we just attach the probability $$\rho_1$$ to cluster one $$\mu_1$$ and $$\rho_2$$ to cluster two $$\mu_2$$ and so on, we'll end up with a random measure as explained below $$G=\sum_0^\inf \rho_k \delta_\mu_{k} = DP(\alpha, N(\mu_0,\sum_0))$$ which we call the Drichlet process. The Drichlet process is not restricted to the Gaussian bases and can be other things, for example another Drichlet distribution on words in an LDA topic model. Steps 3,4 from the generative model above are equivalent to doing a single draw from the Drichlet process.



## Random measure 
- think of a mathematical object that consists of the sum to infinity of product of two random variables $$G =\sum_0^\inf \pi_k \delta_phi_{k}$$. For example, a random coefficient $$\pi_k$$ that comes from successively breaking a the remainder of a stick of originally length one randomly (stick-breaking process), and a random basis function $$\delta$$ of bases $$\phi_k$$. Think of spin glass in physics in 3D; the location of atoms are random basis $$\delta_phi_{k}$$ and the magnitude of bases are also draws from a random variable $$\pi$$.

- If we consider a subset of this 3D space and evaluate this measure at that subset we'll end up with a random measure. The random measures of the subsets are indeces of the random variables (i.e. random measures). Therefore, we'll end up with an indexed set of random variables which means a random process. This is a called Drichlet Process.  

## Chineses restaurant process (CRP):
- A special case of the above random measure where we are integrating out the $$GEM$$. Suppose an infinite number of tables in a chinese restaurant, the first customer comes in and sits at the first table, the second customer comes in and can either join the first customer at table one with a certain probability or start a new table and so on. Therefore, this is a preferential attachment model where $$GEM(\alpha) ~ p(cust_{n+1} joins table \pi_k | \pi)= { \frac{\pi_k}{\alpha+n} if \pi_k existing} or {\frac{\alpha}{\alpha+n} if starting new table}$$.

- Formally, the the stick partitions $$\pi_k$$ are tables here. The customers are the atoms that first choose a parameter like their food (or their location in random measure example) and then choose their table based on their preferential attachment model. For example $$\pi={{1,2,5}, {6}, {3,4}}$$ means that 3 customers joint table 1 , one customer table 2 , etc. 

- Therefore, the chinese restuarant probability according to preferential attachment rule above is the product of probabilities of all customers and all tables. For the example above it's the product of probability $$1=\frac{\alpha}{\alpha}$$ for first customer (1) at the first table; $$\frac{1}{\alpha+1}$$ for second customer (2) that joins table 1; $$\frac{\alpha}{\alpha+2}$$ for the third customer (3) that starts a new table and so on. If we work out the general rule of this product, it comes down to  $$p(\pi_[N])= \frac{\alpha^k}{(\alpha+n)!} * \prod (c-1)!$$ where c is the cardinality (number of customers) of a table. Note that the ordering of customers is not important in the probability of a chinese restaurant process. 


