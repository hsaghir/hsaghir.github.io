---
layout: article
title: Geometric intuitions of Linear Algebra - Deep Learning Book - Ch 2
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

Think about linear algebra as multi-dimensional space and everything else would be easier to understand. 

Studying linear algebra involves three types of mathematical objects:

1. Scalar: a single number. A point in 1 dimensional space.
2. Vector: Think of it as a vector/point in an N dimensional space. 
3. Matrix: A 2-D array of numbers. Think of the columns of a matrix as vectors defining an N dimentional space (coordinate system) with each vector as a direction. If these vectors are eigen-vectors then the coordinate system is canonical since all directions are orthogonal. 
4. Tensor: An array of numbers with more than 2d. 


- Transpose: mirror the array against the diagonal. Note that a matrix and its transpose will have the same rank and eigenvalues but different eigenvectors (unless it's symmetric). Since there is only N orthogonal vectors in N dimensional space, therefore, for a square matrix, the space defined by a matrix and its transpose should be mere rotations of each other!?  

- Addition: Matrices of the same shape can be added together, meaning that the spaces of the same dimensions can be summed.  

- Matrix product: The first matrix should have the same number of columns as the rows of the second matrix. It's distributive i.e. $A(B+C)=AB+AC$ and associative i.e. $A(BC)=(AB)C$ but not communicative i.e. $AB!=BA$

- Element-wise product: multiplying elements of the two matrices. 

- dot product: the same as matrix product but for two vectors. This is communicative for two vectors since the result is a scalar. Think of dot product as the projection of one vector onto another in N dimensional space. So it can be written as the product of their norms times the cosine of the angle inbetween. So two non-zero vector are orthogonal if their dot product is zero which means that the angle between them is 90. 

- In N dimensional space, at most N non-zero vectors can be orthogonal to each other. If Their norms are also 1 along with orthogonality, they are called orthonormal. An orthogonal matrix is a matrix in which the vectors (either columns or rows) are orthonormal.

- A basis for an n-dimensional space is any set of linearly independent vectors that span the space.

- An orthogonal basis for an n-dimensional space is any set of n pairwise orthogonal vectors in the space. Since in an n-dim space, at most n vectors can be orthogonal to each other, there is only one orthogonal basis for an n-dimensional space. Other variants have to be mere rotations of the same orthogonal bases. 

- System of Linear Equations: is defined as Ax=b. Think of the solution as how far should we travel across the coordinate system directions/vectors to get to point b i.e. a linear combination of distances with direction to get vector/point b.  

- Matrix inverse:  A^-1 inverse has eigenvalues λ^–1 corresponding to the same eigenvectors x. So an inverse matrix defines a space with the same bases but reciprocal magnitude (λ^–1). In practice the combination of matrix inverse and a vector are easier and more precise to calculate numerically. For inverse to exist there should be either one or infinity linear combinations to get to b in A coordinate system since if there is more than one solution any linear combinations of them is also a solution. 

- Linear independance: A set of vectors define distinct directions if they are linearly independant meaning that non of them is a linear combination of the others. 

- Span (range) of a matrix: the set of all points reachable by a linear combination of the direction vectors of the matrix. For the linear equation to have a solution for all b values the matrix that defines the coordinate system needs to have at least as many distinct directions (column vectors) as the number of b elements to be able to get to that point! Additionally to have only one solution for each b the coordinate system needs to have exactly as many directions as the number of elements of b. So it needs to be a square matrix with distinct directions. If not all directions are distinct then it's called a singular matrix!

- The rank of a matrix is defined as (a) the maximum number of linearly independent column vectors in the matrix or (b) the maximum number of linearly independent row vectors in the matrix. Both definitions are equivalent. Therefore, the rank of a matrix and it's transpose are the same. 


- Norm: a norm is used to measure the size of a vector by powering each element by p and sum them up then get the p root (L-p norm). The L2 norm is the Euclidean distance, L2 norm increases slowly near zero so in such cases L1 norm might be more useful. L-infinity norm is the maximum element. Sometimes the size of a matrix is measured by Frobenius norm which is L2 norm of square root of sum of element-wise squares.

- A symmetric matrix arises from a symmetric function for example distance. The distance between i and j is the same as the distance between j and i. 

Any symmetric matrix
1) has only real eigenvalues;
2) is always diagonalizable;
3) has orthogonal eigenvectors.

# Eigen decomposition:
- We can transform the directions of the coordinate system defined by a matrix into a canonical orthogonal coordinate system defined by eigen vectors. Only diagonalizable matrices can be factorized in this way. Eigen values are the normalizing factors of the orthogonal directions, so if all directions are independant, then all eigen values are non-zero. The biggest eigenvalue determines the most important direction defined by a matrix. If all eigenvalues are positive, then the matrix is called positive definite.

- The canonical coordinate system defined by the matrix of eigenvectors can be scaled by the eigenvalues back to obtain the original coordinate system by $A=V diag(\lambda) V^T$. This is called eigen decomposition of A which doesn't always exist but every symmetric real valued matrix can be eigen decomposed. 

- Singular value decomposition (SVD) is very similar to Eigen decomposition but more general since it can decompose to singular vectors and singular values. This means that all matrices have an SVD decomposition. It decomposes a matrix to three matrices, U, D, V where U is mxm orthogonal eigenvector matrix of $AA^T$, D is an mxn diagonal of singular values and V is an nxn orthogonal eigenvector matrix of $A^TA$. For example, a 2x3 matrix is actually the combination of a 2x2 orthogonal coordinate system or space, and a 3x3 orthogonal space. This combination is controlled by the singular values diagonal matrix D. 

- Psuedo-inverse: For non-square matrices, inverse becomes psuedo-inverse. SVD easily defines psuedo inverse as $A^+ = V D^+ U^T$ which means that we reverse the order or transformations on the two orthogonal coordinate systems and scale them by the inverse of the eigenvalues.

- Condition number: The ratio of the largest to smallest singular value in SVD. A matrix with a very large condition number is ill-conditioned meaning that it is characterized by a very large change in a certain direction while only a miniscule change in another. Numerical calculations with ill-conditioned matrices is difficult due to over/under-flow problems. 

- Trace: Trace is the sum of diagonal values of a matrix so it's scalar and invariant to moving around matrices in a product. 

- Determinant: is the product of all eigenvalues of a matrix. Since eigenvalues scale different directions/vectors of a matrix, the determinant provides a measure of how a space is transformed through multiplication by a matrix.

- PCA: Consider having m points/vectors in n dimensional space. How to do dimensionality reduction on these vectors with least reconstruction error in a linear autoencoder scheme? Think of all vectors in an mxn space. The eigen vectors of this space and their corresponding eigenvalues are the solution for best linear transformation with least reconstruction error. So basically find the n eigenvalues and eigenvectors, and choose as as many of them as you like for dimensionality reduction starting from the largest to smallest! 

-LU decomposition factors a a square invertable matrix as the product of a lower triangular matrix and an upper triangular matrix. Computers usually solve square systems of linear equations using the LU decomposition, and it is also a key step when inverting a matrix, or computing the determinant of a matrix.

- Cholesky decomposition is the LU decomposition of a Hermitian (complex symmetric), positive-definite matrix into the product of a lower triangular matrix and its conjugate transpose, which is useful e.g. for efficient numerical solutions. Roughly twice as efficient as the LU decomposition for solving systems of linear equations.