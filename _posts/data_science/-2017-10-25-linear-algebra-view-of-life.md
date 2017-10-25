---
layout: article
title: Linear algebra view of Fourier, Wavelets, Machine Learning (matrix factorizations)
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---


Almost everything we do in signal processing and machine learning using computers (and any other computational field for that matter) is done through matrix operations and linear algebra plus sometimes element-wise nonlinear operations and adding/subtracting matrices of noises generated according to a density function. Why don't we make this view explicit and all these fields first from the linear algebra viewpoint and then develop other mathematical treatments where necessary. This may not be possible in development of new algorithms but in hindsight for established methods, this is the obvious way of learning the subjects. This might as well be the reason that deep learning works so well and that we have no idea why. We have experimentally figured out ways of doing a series of matrix operations on data that just works but the rigorous mathematical understanding is much harder to develop.


- Calculating the Fourier transform of a signal is a single linear operation  $$y = W.x$$ where the matrix $$W$$ is the kernel sine and cosine functions.

- Calculating the wavelet coefficients of a signal is a series of convolutions (linear operations i.e. $$y = W.x$$) and downsampling procedures.

- The question is how to find the wavelet kernels of different scale to use for convolutions. 

