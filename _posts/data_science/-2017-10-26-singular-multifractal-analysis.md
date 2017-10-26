---
layout: article
title: Quantifying singularity and fractal dimensions of their support set
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---



## [Singularity in time series](https://physionet.org/tutorials/multifractal/behavior.htm)

The functions f(t) typically studied in mathematical analysis are continuous and have continuous derivatives. Hence, they can be approximated in the vicinity of some time ti by a so-called Taylor series or power series

For small regions around ti, just a few terms of the expansion (eqn. 1) are necessary to approximate the function f(t). In contrast, most time series f(t) found in "real-life" applications appear quite noisy (Fig. 1). Therefore, at almost every point in time, they cannot be approximated either by Taylor series (or by Fourier series) of just a few terms. Moreover, many experimental or empirical time series have fractal features--i.e., for some times ti, the series f(t) displays singular behavior. By this, we mean that at those times ti, the signal has components with non-integer powers of time which appear as step-like or cusp-like features, the so-called singularities, in the signal (see Figs. 1b,c).

## Holder exponents as an index of singularity in signals

Holder exponents describe the local regularity of a function or distribution and index its local singularity strength. For a real, Holder continuous function $f(t)$, the Holder exponent $h(t_0)$ at point $t_0$ is defined as the supremum of all values of $h>0$ such that there exists a polynomial $P_n$ of degree $n<h$ that satisfies,
\begin{equation}
\label{eq:holder}
|f(t)-P_n(t-t_0)| \leq C|t-t_0|^h, \quad C>0 \  \textrm{is a positive constant}
\end{equation}


## Holder exponent estimastion - DFA
Holder exponents can be estimated from the local scaling behavior around each sample of time-series $X =\{x_1, \ldots, x_N\}$ . The range of scales used in the local detrending are determined by the minimum, $s_{min}$, and maximum scales, $s_{max}$. In particular, for a scale $s\subset(s_{min}, s_{max})$ and the corresponding time series interval centered around point $x_i$, i.e. $[x_i-s/2,x_i+s/2]$,  the root-mean-square deviation, $r_i(s)$, of the time-series, $X$, from a local detrending polynomial, $P_n$, of degree $n$ is calculated as,
\begin{equation}
\label{eq:deviation}
r_i(s)= \sqrt{\frac{1}{s+1} \sum_{k=-s/2}^{s/2}\Big[x_{i+k} - P_n^{i,s}(k)\Big]^2}
\end{equation}
where $\frac{s}{2}+1 \leq i \leq N-\frac{s}{2}$, and the superscript on $P_n$ explicitly indicates that the local polynomial fit depends on the time index $i$ and the chosen scale $s$.
\noindent The slope of a linear fit of log deviations versus log scales then estimates the Holder exponents,
\begin{equation}
\label{eq:logfit}
\log{r_i(s)}=h_i(s)\log{s}+ const
\end{equation}

In this setting, Holder exponent estimation is unstable and prone to numerical errors. Struzik \cite{Struzik2000} presents a methodology for a more stable estimation of scale- and time- dependent Holder exponents through the paradigm of the multiplicative cascade model. The local Holder exponent $h_i(s)$ at a given scale $s$ is stably estimated from the following equation \cite{Ihlen2012,Struzik2000} .
\begin{equation}
\label{eq:localh}
h_i(s)= \frac{\log{r_i(s)}-(H\log(s)+C)}{\log(s)-\log(N)}+H
\end{equation}

\noindent where $H$ represents the global Hurst exponent calculated according to conventional DFA \cite{Kantelhardt2001}, $H\log(s)+C$ represents the corresponding regression line in $\log$ scales and $N$ is the length of the time-series.


## Holder exponent estimation using Wavelets

In a signal with fractal features, an immediate question one faces is "how to quantify the fractal properties of such a signal?'' The first problem is to find the set of locations of the singularities {ti}, and to estimate the value of h for each ti. 


In contrast to the Fourier transform, which assumes that the signal is stationary at the time scales of interest, the wavelet transform instead determines the frequency content of a signal as a function of time (1). In the Fourier transform, one determines the coefficients that best approximate a function f(t) as a sum of sines and cosines. Similarly, in the wavelet transform, one approximates a function f(t) as a sum of properly weighted basis functions. The basis in the wavelet transform are functions that, like the sines or cosines, can be considered at different frequency but, unlike the sines or cosines, are localized in time and hence have to be translated along the signal. An example of a wavelet basis is the set of functions.


where G'(t,a,b) is the first derivative of the Gaussian function, a is an inverse frequency and b is the time location. One determines the coefficients of the wavelet transform by convolving f(t) with G'(t,a,b).

Besides being naturally suited to handle nonstationary signals, the wavelet transform easily removes polynomial contributions that would otherwise mask singular (fractal) behavior. To illustrate this fact, consider a signal f(t) that one can expand for t close to ti as a series of the form of Eqn. (2). In a fractal analysis, one wants to measure hi, but for small values of t - ti, the "trends'' (t - ti) k with k < hi will dominate the sum. Hence, one ideally wants to remove all terms (t - ti) k for which k < hi. By convolving f(t) with an appropriate wavelet function, one can put to zero all coefficients that would arise from such polynomial contributions. For instance, the derivative of order k of the Gaussian convolves to zero all polynomial terms up to order k - 1.

Figure 3 shows the wavelet decomposition of a heart rate signal for a healthy subject. The self-similar "arch''-like structures in the figure indicate maxima of the modulus of the wavelet transform. They indicate the time locations (at each scale) of the singularities in the signal. The figure helps illustrate two points. First, the singularities are not present for all times. Second, the location of the singularities, as a function of scale and time, have a fractal structure. 

Another interesting property of the wavelet transform is that the coefficients at these maxima---which are a small fraction of the total number of coefficients---are enough to encode the information contained in the signal (3). Moreover, as one follows a maxima line from the lowest scale to higher and higher scales, one is following the same singularity. This fact allows for the calculation of hi by a power law fit to the coefficients of the wavelet transform along the maxima line (5).

We ask the question "what comes out of our analysis of the signal?" The first possibility is that we find a single value hi = H for all singularities ti, the signal is then said to be monofractal (6, 7). The second, more complex, possibility is that we find several distinct values for h, the signal is then said to be multifractal (8, 9). 


## Fractal dimension of the support set of singularities

The next problem is to quantify the "frequency" in the signal of a particular value h of the singularity exponents hi. Let us first assume that our signal is monofractal. Different possibilities can be considered. For example, the set of times with singular behavior {ti} may be a finite fraction of the time series and homogeneously distributed over the signal. But {ti} may also be an asymptotically infinitesimal fraction of the entire signal and have a very heterogeneous structure. That is, the set {ti} may be a fractal itself. In either case, it is useful to quantify the properties of the sets of singularities in the signal by calculating their fractal dimensions (8).

Consider the signal in Fig. 4. This type of signal is usually called a Devil's staircase because it takes constant values except at a subset of points where it changes discontinuously (2, 4). At those points, the function f(t) has singularities. Moreover, all singularities are of the same type --i.e., the signal is monofractal. 

Because the signal of Fig. 4 is deterministic, we can easily identify the position of the singularities. Their positions are shown in the top panel of Fig. 4. One can see that the singularity points arise from the iteration of a Cantor set rule. The signal in the bottom panel of Fig. 4 arises from integrating the "dust'' generated by the Cantor rule (8).

One can easily calculate the fractal dimension of the Cantor set of singularities by using box counting methods. The fractal dimension is, as usual

In the multifractal formalism, one says that the signal of Fig. 4 has a single type of singularity hi = 1/2 and that the support of that singularity has fractal dimension D(h=1/2) = 1/2. The curve D(h) is called the singularity spectrum of the time series, which for this case is zero everywhere except at a single point h = 1/2.

The signals in Figs. 1a,c are also monofractal. They are usually called fractional Brownian motion. For the signal in Fig. 1a we have h = -0.8 while for the signal in Fig. 1c we have h = 0.2. But in contrast with the devil's staircase of Fig. 4, for which singularities appear only for a very small and heterogeneous set of times, singularities appear uniformly throughout the signals in Figs. 1a,c. Hence, the fractal dimension of the set of singularities is one, the dimension of a line. 

## Singularity/Multifractality specrum

Our analysis becomes more complex if instead of a single type of singularity, the signal of interest has multiple types of singularities. As an example, consider the signal in Fig. 5 which is also a Devil's staircase (i.e., Fig. 4) because of its many singularities. But in contrast to the signal of Fig. 4, the types of singularities vary considerably. The reason for this variation is made clear by the top panel in Fig. 5. The type of fluctuations in local increments vary considerably even for the fourth iteration. 

 To quantify the variation in the local singularities of the signal of Fig. 5, we calculate the value of h at every singularity. Figure 6 shows the signal again and also, by a color coding, the value of h. Clearly hi can take many different values. Moreover, by focusing on a single color, i.e., a single value of h, one can uncover the fractal structure of the corresponding set of singularities. 

The singularity spectrum D(h) quantifies the degree of nonlinearity in the processes generating the output f(t) in a very compact way (see Fig. 7). For a linear fractal process the output of a system will have the same fractal properties (i.e., the same type of singularities) regardless of initial conditions or of driving forces. In contrast, nonlinear fractal processes will generate outputs with different fractal properties that depend on the input conditions or the history of the system. That is, the output of the system over extended periods of time will display different types of singularities. 

A classical example from physics is the Navier-Stokes equation for fluid dynamics (10). In the turbulent regime, this nonlinear equation generates a multifractal output with a characteristic singularity spectrum D(h) similar, for some types of turbulence, to D(h) for the binomial multiplicative process.

Multifractality has been uncovered in a number of fundamental physical and chemical processes (9). Recently, it was also reported that heart rate fluctuations of healthy individuals are multifractal (11). This finding posed new challenges to our understanding of heart rate regulation as most modeling of heart rate fluctuations over long time scales had concerned itself only with monofractal properties (12). For example, it appears that a major life-threatening condition, congestive heart failure, leads to a loss of multifractality (Fig. 8). 

More importantly, neither monofractal nor multifractal behaviors are accounted for by current understanding of physiological regulation based on homeostasis. Hence it would be beneficial, perhaps, to uncover how multifractality in the healthy heart dynamics arises. Two distinct possibilities can be considered. The first is that the observed multifractality is primarily a consequence of the response of neuroautonomic control mechanisms to activity-related fractal stimuli. If this were the case, then in the absence of such correlated inputs the heartbeat dynamics would not generate such a heterogeneous multifractal output. The second is that the neuroautonomic control mechanisms---in the presence of even weak external noise---endogenously generate multifractal dynamics.


## Multifractal processes

Fractional Brownian motion (fBm), also called a fractal Brownian motion, is a generalization of Brownian motion. Unlike classical Brownian motion, the increments of fBm need not be independent. fBm is a continuous-time Gaussian process BH(t) on [0, T], which starts at zero, has expectation zero for all t in [0, T], and has the following covariance function:

$$E[B_H(t) B_H (s)] = \frac{1}{2}  (|t|^{2H}+|s|^{2H}-|t-s|^{2H})$$

where H is a real number in (0, 1), called the Hurst index or Hurst parameter associated with the fractional Brownian motion. The Hurst exponent describes the raggedness of the resultant motion, with a higher value leading to a smoother motion. 

Therefore, Brownian motion (White noise) is uncorrelated due to the constant and independent covariance function for the points while fBm has long/short range correlations due to the above covariance function. 

The process is self-similar, since in terms of probability distributions:

$$ B_H (at) \sim |a|^{H}B_H (t).$$

This property is due to the fact that the covariance function is homogeneous of order 2H and can be considered as a fractal property. Fractional Brownian motion is the only self-similar Gaussian process. For H > .5 the process exhibits long-range dependence,
