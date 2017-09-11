---
layout: article
title:  Missing value imputation with deep generative models
comments: true
categories: data_science
image:
  teaser: practical/pytorch_logo.png
---



## Lit review

- [Semantic Image Inpainting with Deep Generative Models](https://arxiv.org/pdf/1607.07539.pdf)

### Collaborative filtering
Collaborative filtering methods based on Matrix factorization methods are a sort of missing value imputation

### Matrix factorizations:
- Eigen decomposition: is the transformation of a space to canonical (without rotation and dilations) form. 
    + The matrix should be a square positive semidefinite normal matrix.
    + $$A$$ multiplied by a vector $$e$$, results in a product of a scalar (eigenvalue) $$\lambda$$ and the vector (eigen-vector) $$e$$ .
    + If $$A$$ is full-rank, then the matrix can be factorized as $$Q^-1 \Lambda Q$$ where Q is the matrix of all eigen-vectors and $$\Lambda$$ is the diagonal matrix of eigen-values.

- Singular value decomposition: SVD is the generalization of the eigendecomposition to any rectangular mxn matrix.
    + The space goes through a sequence of a rotaion $$U$$ (a square mxm real or complex matrix), a dilation $$\Sigma$$ (mxn rectangular diagonal matrix with non-negative real numbers as singular values) and a rotation $$V$$ (a square nxn real or complex matrix).
    + Original matrix is factorized as$$ U \Sigma V $$.

- Non-Negative Matrix Factorization: NMF seeks to decompose a non-negative rectangular nxp matrix $$A$$ to two rectangular matrices $$UV$$ which are nxr and rxp respectively. The rows of $$V$$ are basis in the r-dim space and the rows of U are non-negative coefficients of linear combinations of the bases.
    + NMF transforms the space to a hyperspace with user-defined dimensions and finds a simplicial cone contained in the positive orthant which contains the transformed space. 
    + NMF creates a user-defined number of features each of which is a combination of the original attributes.
    + NMF approximately factorizes a non-negative matrix $$A$$ into 2 non-negative matrices $$UV$$. 
    + Uses SGD to minimize the Frobenius norm distance between the original and approximate factorization.

- PCA/Probabilistic latent matrix indexing/ SVD++/timeSVD++

- CUR decomposition: CUR is similar to but less accurate than SVD. rows and columns come from the original matrix giving advantages of faser calculation, and more interpretablity; The meanings of rows and columns in the decomposed matrix are essentially the same as their meanings in the original matrix.
    + Given mxn matrix $$A$$, we select $$c$$ columns of from $$A$$, and form a mxc matrix $$C$$. we form a cxc matrix U and Using the same $$c$$ columns, we form a cxn matrix $$R$$.
    + 



## Strategies for handling missing data


Missing data classes:
    - Missing completely at random (CR): The probability of an instance having a missing value doens't depend on the known values or the missing data
    - Missing at semi-random (SR): The probability of an instance having a missing value only depends on known values and not the missing values.
    - not missing at random (NR): when the probability of an instance have a missing value depends on the value of that attribute.

General strategies: 
    - Ignoring and discarding the data (only CR since discarding a non-random missing value biases the results)
        + discarding all instances with missing values
        + assessing the extent of missing data on each instance or attribute and discard instances or attributes with high level of missing data. 
        + if an attribute is relevant to analysis it should be kept even with missing data!
    - Maximum likelihood learning with Expectation Maximization can perform parameter estimation in presence of missing data. 
        + Modeling the supervised problem as the problem of modeling the joint distribution. Given that we have both X and Z, we can estimate parameters using maximum likelihood. Then using an EM scheme we iteratively predict X from Z and Z from X which handles the missing values problem in X. 
        + Treating missing values in data as partially-unobserved random variables in a graphical model?
    - Imputation: estimating (modeling) the missing values from the valid values of the data. 
        + mean / mode
        + Hot deck: clustering the dataset; and replacing each instance of missing value with cluster mean/mode. Cold deck: similar but the data source must be different from the current dataset. 
        + model: predict the attributes with missing values from other parts of the dataset. This is possible since in most cases attributes are not completely independent. A drawback: might introduce bias since predicted missing values might depend more on other parts of the dataset than the actual attribute with missing values.




