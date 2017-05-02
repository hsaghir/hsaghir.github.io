
## [Business of Fraud Detection]()
- Fraud detection is a very big problem. For an organization like RBC that's in the order of 100s of millions per year. 

- Currently RBC does not use any machine learning in their fraud detection unit. 

- There are tons of startups providing analytics/ML solutions, but the main concern of the bank is data privacy. They are not comfortable giving customer data to third parties, their brand will be at risk if they do, and there might be some potential regulatory risks. 

## [Techniques]()
- Unsupervised: 

-- Autoencoder, moving average, KL divergence (reconstruction cost, what's input?)

-- Autoencoder, dimensionality reduction anomaly detection. 

-- Maybe, anomalous events occur rarely in the training data, preventing the autoencoder from producing a good reconstruction at those events!? Therefore, the KL distance can be thresholded to find anomalous events? [cf](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3193936/)


-- change point detection, 



- Supervised: 
-- RNN learns from labeled transactions points in a time series (predict anomaly occurance)
-- ?



[Orange tests DL to identify fraud](https://blogs.wsj.com/cio/2016/03/14/orange-tests-deep-learning-software-to-identify-fraud/)
[DL infrastructure in production](https://insights.ubuntu.com/2016/04/25/making-deep-learning-accessible-on-openstack/)


[How PayPal beats the bad guys with machine learning](http://www.infoworld.com/article/2907877/machine-learning/how-paypal-reduces-fraud-with-machine-learning.html)

- PayPal uses three types of machine learning algorithms for risk management: linear, neural network, and deep learning. Experience has shown PayPal that in many cases, the most effective approach is to use all three at once. we “ensemble” them together!

- detect fraud: "we can separate the good from the bad with one straight line". When linear fails, neural network. Also,tree-based, mimicking a human being when we have to make a judgment -- for example, we would say if it’s raining, I’ll take an umbrella.


[Ubalanced dataset for Fraud](http://rvc.eng.miami.edu/Paper/2015/Yan_ISM2015.pdf)

- The minority class is actually class of interest which is the fraud instances.



[Classification pipeline]()

Neural net workflow for "Large-scale malware classification using random projections and neural networks":

1. Feature extraction
2. Feature selection

3. Dimensionality reduction using random projections: Even after feature selection, the input dimensionality is still quite large (179 thousand), although there is a lot of sparsity. Naive neural net training on such high dimensional input is too expensive. To make the problem more manageable, we used the very sparse random projections. 

We project each input vector into a much lower dimensional space (a few thousand dimensions) using a sparse projection matrix R with entries sampled iid from a distribution over {0, 1, −1}. Entries of 1 and -1 are equiprobable and $$P(Rij = 0) = 1− \frac{1}{\sqrt(d)}$$, where $$d$$ is the original input dimensionality. Another way to view the random projection in the context of neural network training is that the random projection step forms a layer with linear hidden units in which the weight matrix is not learned and is instead simply set to R.

4. Classifier (ensemble NN!?)



