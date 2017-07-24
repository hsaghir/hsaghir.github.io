---
layout: article
title: Information Geometry
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

- Information geometry applies the techniques of differential geometry (Riemannian geometrey) to probability theory. It studies the properties of manifolds of probability density functions. Goes back to 1945 when Fisher proposed that a statistical model can be considered as a manifold with Fisher information as the metric. In 1970s notions of statistical curvature, and flat connection were introduced and it was shown that a well-defined divergence function induces a Riemannian metric and a pair of dual connections. Later, alpha-connections and dually-flat manifolds were studied.

## Differential Geometry
- Main theme in differential geometrey is to characterize the global properties of manifolds.

- A differentiable manifold is a generalization of a smooth curve or surface to n-dimensional space.

## Riemannian geometry





## Information geometry in ML
1. starting point is to consider a family of probability distributions (parameterized by $$\theta$$) as a smooth manifold. each distribution in the family is a point on this manifold.

2. Then the tools of differential geometry (Riemannian geometry in particular) can be used to study geometrical properties of this manifold. The Riemannian geometric analysis makes better sense than the trivial Euclidean geometry, as some interesting and desirable properties of distribution families cannot be derived from Euclidean geometry.

3. we need to introduce some structures to the manifold, most importantly the Riemannian metric tensor and affine connections. these structures are induced by properly defined divergence functions between distributions (in other words, a divergence function determines the geometry of a statistical manifold). Different divergences define different Riemannian geometries.


- For example, first on distribution families side, some families like the exponential and mixture families, as simple geometric structures(making the analysis easier). Second we choose to use Riemannian geometry for its desirable properties, and third we structure the manifold with some divergences like f-­divergence and Bregman divergence, as they can help derive some nice and desirable geometric properties of distribution families.

### Divergence functions and geometry

- a divergence function between two distributions has to satisfy three properties: 1. it has to always be positive 2. be zero only when the distributions are the same and 3. Its Hessian be positive definite. 

- Divergences are like distance measures with the difference that they don't need to satisfy symmetry and triangle inequality properties. A divergence can be regarded as a local distance measure on the manifold.

- Although many divergence functions have been proposed in the literature, most of them are special cases of two general classes of divergences, namely the f-divergence and Bregman divergence.

- f-divergence always induces a Fisher information metric and an α­-connection manifold structure. The choice of f only affects the scale of the metric and the value of α. It is the only metric and connection that is invariant with respect to an invertible mapping of the variables.

- Bregman divergence relies heavily on convex analysis since Bregman divergence induces dually flat manifolds equipped with dually affine coordinate systems. The two coordinate systems can be transformed to each other via convex conjugate functions, and the function pairs happen to preserve local metrics under different coordinate systems. With Bergman divergence, Riemannian metric is preserved when shifting from one coordinate system to the other. 

- The manifold defined by Bergman divergence is flat with respect to both dual connections since the the corresponding curvature tensor vanishes when the connection is zero. Thanks to the special structure of the dually flat manifold, some interesting geometric properties can be derived, most notably the generalized Pythagorean theorem and the projection theorem.

- KL divergence can be derived from both f-divergence by setting $$f=x\log(x)$$ and Bergman divergence by G be negative entropy. Therefore, the geometry of KL­-divergence is both invariant (from f ­divergence) and dually flat (from Bregman divergence). In fact, for probability measures the KL­divergence is the unique intersection of the f and Bregman divergences. 

- unlike the Bregman divergence, the manifold induced by the f-divergence is in general not flat since the α­-connections is in general not 0. It is only in the case of KL­-divergence that the f -divergence induces flat geometry, borrowing properties from Bregman divergence.

- The conventional gradient descent (GD) assumes the underlying parameter space to be Euclidean, nevertheless this is often not a proper assumption (say when the space is a statistical manifold). By contrast, Natural GD shifts GD to non­Euclidean parameter spaces and takes the Riemannian geometry of the space into consideration. 

- When the objective functions are probabilistic models, the parameters are those that controls distribution families and Riemannian manifold is a good assumption about the geometry of the parameter space.

- To do gradient descent on this manifold, we need to find the steepest descent direction. The steepest descent direction on the manifold is given by transforming the steepest gradient in Euclidean parameter space to manifold space using Riemannian metric G(θ) at point θ. Therefore the (natural gradient) is given by $$G(θ)^−1 ∇L(w)$$,
