---
layout: article
title: ConvNets Deep Learning book Ch9
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

- ConvNets apply convolutions through tied weights in a layer and can be applied to any grid-like data with similar spacing. Two main ideas are sparse connections where only units are locally connected and parameters sharing through weight tying. Convolutions are invariant to translations of inputs. The main benefit of ConvNets is allowing to run networks on very large inputs without too many computational resources.

- Think of convolution operation as applying a function to a given window of a singal as it evolves through time. For example applying an average function to a given window of values for each measurement of a temperature. convolution in matrix form is a sparse matrix with the same weights shifted to right at every row.

- Convolution effectively slides a kernel along all dimensions of the signal. At each location the kernel and the signal patch are element-wise multiplied and then added up (MUC-multiply accumulate). We usually have multiple filter kernels in each layer learning features (tensor depth). In the backward pass of a convnet, we need to perform deconvolution which is basically multiplication by the transpose of kernel. If input matrix is X and filter kernel matrix is F, then, the convolving the input X with filter F,  X*F, would involve  element-wise multiplication or simply vectorizing both matrices and calculating their dot product $x . f = c$. Deconvolution, would then similarly consist of calculating the dot product of the output c, and the transpose of the vectorized kernel $x=c.f’$. 

- Pooling is a summary statistic of a local region e.g. maximum, average, or L2 norm. Pooling is equivalent to adding a strong prior that the function learned be translation invariant upto the stride of pooling. Invariance useful if we care about whether a feature is present rather than where it exactly is. Key insight is that if a task relies on preserving precise spatial information, then using pooling on all features can increase the training error.

- Both Convolution and pooling are strong priors applied in the network design. A convnet is more efficient than imposing the priors using regulizers, but can also lead to underfitting if the priors don't match the data and problem. Models that do not use convolution would be able to learn even if we permuted all of the pixels in the image. Therefore, there is a clear distinction between models that are permutation invariant and must discover the concept of topology via learning, and models that have the knowledge of spatial relationships hard-coded into them by their designer.

- When working with images, we usually think of the input and output of the convolution as being 3-D tensors, with one index into the different channels and two indices into the spatial coordinates of each channel. Software implementations usually work in batch mode, so they will actually use 4-D tensors, with the fourth axis indexing different examples in the batch. ConvNet libraries should be able to implicitly zero-pad the input before each conv, otherwise the output will shrink by 1-kernel width. 

- Other variants of imposing local structure prior on a NN is locally connected net where weights are not shared but are only locally connected or tiled convolution where weights for locally connected NN in the form of convolutions are partly shared resulting in a rotation between a few kernels. 

- For convolutional layers, it is typical to have one bias per channel of the output and share it across all locations within each convolution map. However, if the input is of known, fixed size, it is also possible to learn a separate bias at each location of the output map. Separating the biases may slightly reduce the statistical efficiency of the model, but also allows the model to correct for differences in the image statistics at different locations. For example, when using implicit zero padding, detector units at the edge of the image receive less total input and may need larger biases.

- ConvNets are well-suited to variable size inputs/outputs. For example, consider a collection of images, where each image has a different width and height. It is unclear how to model such inputs with a weight matrix of fixed size. Convolution is straightforward to apply; the kernel is simply applied a different number of times depending on the size of the input, and the output of the convolution operation scales accordingly.

- Convolution is equivalent to converting both the input and the kernel to the frequency domain using a Fourier transform, performing point-wise multiplication of the two signals, and converting back to the time domain using an inverse Fourier transform. For some problem sizes, this can be faster than the naive implementation of discrete convolution. When a d-dimensional kernel can be expressed as the outer product of d vectors, the kernel is called separable (orthogonal basis?). When the kernel is separable, naive convolution is inefficient. It is equivalent to compose d one dimensional convolutions with each of these vectors. The composed approach is significantly faster than performing one d-dimensional convolution with their outer product.

- There are three basic strategies for obtaining convolution kernels without supervised training (computationally expensive). One is to simply initialize them randomly. Another is to design them by hand, for example by setting each kernel to detect edges at a certain orientation or scale. Finally, one can learn the kernels with an unsupervised criterion. For example, Coates et al. (2011) apply k-means clustering to small image patches, then use each learned centroid as a convolution kernel.

- Reverse correlation (measuring electrical activity of neuron and estimating weights linearly) shows us that most V1 simple cells have weights that are described by Gabor functions (edge detectors). It has two important factors: one is a Gaussian function and the other is a cosine function. The Gaussian factor ensures the simple cell will only respond to values near the center of the cell’s receptive field and the cosine factor controls change in radial basis. The simple cell will therefore respond to image features centered at the point (x0, y0), and it will respond to changes in brightness as we move along a line rotated τ radians from the horizontal. Altogether, a simple cell responds to a specific spatial frequency of brightness in a specific direction at a specific location. A complex cell computes the L2 norm of the 2-D vector containing two simple cells’ responses. The same kind of features are usually learned by many machine learning algorithms making it hard to say which algorithms is more similar to brain but if an algo doesn't learn similar feature that's a sign it's not like brain. 

- ConvNets are inspired by visual cortex. 1) V1 is arranged in a spatial map, and ConvNets capture this property by having their features defined in terms of two dimensional maps. 2)V1 contains many simple cells with localized receptive field, Local connections and detector units in a ConvNet emulate this. 3)V1 also contains many complex cells that are invariant to some changes in lighting and spatial locations. Pooling in ConvNet emulate this. 

- There are super-complex cells in brain (medial temporal lobe) called grandmother cells pointing to the fact that get activated by a super complex pattern, for example the image of grand mother. 

- The human eye is mostly very low resolution, except for a tiny patch called the fovea. The fovea only observes an area about the size of a thumbnail held at arms length. This has inspired building attention mechanisms into NN.


## Separable Convolutions
- In a separable convolution, we can split the kernel operation into multiple steps. Assuming the kernel can be expressed as $$k = k1.dot(k2)$$, instead of doing a 2D convolution with k, we could get to the same result by doing 2 1D convolutions with k1 and k2.
- **depthwise separable convolution**. This will perform a spatial convolution while keeping the channels separate and then follow with a depthwise convolution. We do this because of the hypothesis that spatial and depthwise information can be decoupled and reduces the number of kernels/parameters in the network. 
    + Let’s say we have a 3x3 convolutional layer on 16 input channels and 32 output channels. What happens in detail is that every of the 16 channels is traversed by 32 3x3 kernels resulting in 512 (16x32) feature maps. Adding up features along output channels give us 32 output channels we wanted.
    + For a depthwise separable convolution on the same example, we traverse the 16 channels with 1 3x3 kernel each, giving us 16 feature maps. Now, before merging anything, we traverse these 16 feature maps with 32 1x1 convolutions each (depth multiplier of 1) and only then start to them add together. This results in 656 (16x3x3 + 16x32x1x1) parameters opposed to the 4608 (16x32x3x3) parameters from above


