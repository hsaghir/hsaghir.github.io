## Literature Synthesis
"Anomaly detection involves identifying the events which do not conform to an expected pattern in data."


### Supervised: 
- treats the problem as a binary classification problem. Classification operates based on the assumption that the a classifier can distrintuish between the normal/anomalous classes from the designated feature space. It requires accurately labeled data. The main challenge here is that the data is inherently unbalanced and needs appropriate methods to tackle that. Popular classifiers have been used in the literature for this purpose. 

---
#### Strategies for unbalanced classification problem 

- Changing the performance metric:
    + Use the confusio nmatrix to calculate Precision, Recall
    + F1score (weighted average of precision recall)
    + Use Kappa - which is a classification accuracy normalized by the imbalance of the classes in the data
    + ROC curves - calculates sensitivity/specificity ratio.
- Resampling the dataset
    + Essentially this is a method that will process the data to have an approximate 50-50 ratio. 
    + One way to achieve this is by OVER-sampling, which is adding copies of the under-represented class (better when you have little data)
    + Another is UNDER-sampling, which deletes instances from the over-represented class (better when he have lot's of data)

---

### Semi-Supervised: 

- Assumes that the data for only one class is available. The approach is to model only one class of the data and identify the other class based on their unconformity to the model. Most semi-supervised models can be used in an unsupervised way where the normal and anomalous data are not labeled. The assumption then would be that normal data is much more frequent than anomalous data. 

#### Point Anomalies
##### K-Nearest Neighbor
- Key assumption is that normal data instances occur in dense neighborhoods while anomalies occur far from closest neighbors. 
- Two main categories exits. techniques that use the distance of an instance to its k-th nearest neighbor as the anomaly score. 2) Techniques that compute relative density of each data point to compute its anomaly score. 

##### Clustering
- Based on the assumption that normal instances belong to a cluster in the data while anomalies do not belong to any cluster.

##### Statistical Anomaly detection
- These models fit a stat model to the data and then perform inference for any new point to see whether that data point belongs to this model or not. The underlying intuition is that an anomaly is data point that is unlikely under the generative process of the model. Assumptions are that normal data points occur in high probability regions while anomalies occur in low probability regions. Two main categories are,  Parametric techniques that assume a certain generative model for data generation such as Gaussian model based techniques, and Regression models for time series. Non-parametric techniques don't assume a generative process. Examples are Histogram-based, and Kernel-based methods. 

##### Information theoretic anomaly detection
- Based on the assumption that anomalies in data induce irregularities in the information content of the dataset as measured by entropy/complexity. 

##### Spectral anomaly detection 
- Based on the assumption that the data can be embedded into a lower dimension space where normal/anomalous points are significantly different. 

- Autoencoder, moving average, KL divergence (reconstruction cost). anomalous events occur rarely in the training data, preventing the autoencoder from producing a good reconstruction at those events!? Therefore, the KL distance can be thresholded to find anomalous events? [cf](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3193936/)

#### Contextual anomalies
- requires defining a context for each data point so that it can be evaluated in that context. The approach usually used is to use point anomaly detection techniques in the subset of data marked as context for that data point.  

Some ways contextual attributes have been defined are:

##### Spatial neighborhood
- Given the location of each data point, the spatial neighbors can be used as context, for example the neighboring points in a sequence. 

##### Graphs 
- edges between nodes define the neighbors which are used as context. 

##### Profile
- The idea is that the data forms sub-clusters or sub-groups and these sub-groups can be used as context for finding anomalies. 

##### change point detection method:
- These methods can be used for defining contexts in which anomalies are assessed. The assumption would be that a change point occurs when the context changes and the anomaly has to be assessed based on the new dynamic regime (context). An alternative view is to use change point detection methods for detecting collective anomalies.


---

## Chosen Methodology

### Internal Use Fraud detector
- Model normal class using an autoencoder (i.e. VAE) 
- The reconstruction error for new data points show the anomaly score of that point.

- The uncertainty of a model about a data point might be useful as fraud score. For example, using a Gaussian process, when a point is outside the uncertainty bounds, it is an anomaly. This lends itself well to active learning as well.

## Per customer fraud detection (ANGEL - ANomaly Guard Event Luncher!?)
- Customer downloads their transaction data into the ANGEL app. ANGEL trains a model on customer data, warns user about anomalies in their transactions, and gets them to label the transaction as safe or not w/ a short description (Active Learning). An example scenario is when customer phone location indicates they are in Toronto while a transaction is reported on their card in EU. ANGEL might mark that as a very highly anomalous transaction and will prompts the user to label it as either good or bad. 
- Bayesian optimization, Gaussian Processes, SVMs, uncertainty score based on a Bayesian approach (or using noise on hidden params) and other similar techniques can tell when uncertainties are high and can query the prompt the user to label the data point. 



## Literature Survey

"What is Anomaly? Anomaly is an observation that doesn’t conform with prediction, which is highly subjective for operators or analysts who use this model as a tool to detect anomalies. For this reason, we first generate an anomaly score by applying the current observation to the learned predictive model and decide the current observation as an anomaly only if its anomaly score exceeds a certain threshold"

---

### Papers

0. [A Comprehensive Survey of Data Mining-based Fraud Detection Research]()

- related adversarial data mining fields/applications such as epidemic/outbreak detection, insider trading, intrusion detection, money laundering, spam detection, and terrorist detection

- In fraud detection, misclassification cost is not balanced. For example, a false negative error is usually more costly than a false positive error. Therefore, the analysis of the ROC curve (plot of true positive vs. false positive) and maximizing the area under this curve (AUC) is required as opposed to simply improving accuracy.

---


1. [A Data-driven Health Monitoring Method for Satellite Housekeeping Data based on Probabilistic Clustering and Dimensionality Reduction]()

- data is high dimensional, distance between typical and anomalous data hard to assess in high dim so we do dim reduction.

- multiple modes of operation in the distribution! Therefore, needs a mixture distribution that can bear multiple modes. A multivariate Gaussian is unimodal!

- Preprocessing: Removing trivial outliers if there are any. 

- dim reduction: probabilistic PCA with mode-specific params assuming iid samples in the time series, $$X=Wh+b+noise$$ meaning that mode-specific $$W_k, \mu_k, \sigma_k$$, are used for each of the k modes. The noise is in the form of $$\sigma_k I$$. The prior on the latents is assumed to be in the form of $$p(h)=N(0,I)$$ for each mode and a switching categorical distribution on modes as a generalized Bernouli dist consisting of $$k$$ probability values for the $$k$$ modes, $p(k)=Cat(\pi)$. This translates to a mixture model of a type $$p(x_t;\theta)=\sum_k \pi_k . N(\mu_k, w_k w_k^T+\sigma_k I)$$.

---

- Anomaly detection: After learning params, the likelihood of a sample given the learned model indicates the probability of the sample being anomalous. Alternatively the log-likelihood can be used as an anomaly score, $$score(a_t)=-\log p(a_t|learned_model)$$ when the learned parameters are used in the model $$p(x_t;\theta)=\sum_k \pi_k . N(\mu_k, w_k w_k^T+\sigma_k I)$$.

- Labeled data: If there is labeled data of anomalies, We can either determine an optimal threshold by considering a trade-off between false positives and false negatives. Or we can model the joint probability of the unlabeled and labeled data 


- Analysis: We can analyze the contribution of each of the dimensions of the data point $$x_t$$ to the anomaly score by reconstruction. We first encode the data $$x_t$$ to the hidden representation and then reconstruct it from the hidden representation. The reconstruction error can be calculated element-wise and the element-wise error shows the contribution of each element to the anomaly score. 

-- We can use time windows instead of every time point $$x_t$$. This way, the element-wise reconstruction error will indicate which time point in that window contributed more to the anomaly score. 

---

2. [Anomaly Detection from Multivariate Time-Series with Sparse Representation]()

- Feature extraction with sparse coding. (make the features with sparse coding and then make a bag of features matrix with feature in row and a sebsequence windows of time series in column. )

- Performing matrix decomposition (SVD, LSA, PCA, etc), reduce dimension of the bag-of-features matix. 

- Reconstruct the signal. Element-wise reconstruction error is the equivalent of anomaly score for each window. The window contributing most to the error or whose error are above a threshold are somewhat anomalous. 

---

3. [Structured Denoising Autoencoder for Fault Detection and Analysis]()
Two steps, 

- dimensionality reduction and reconstruction:  Structured denoising autoencoder

- Structured denoising autoencoder: modified objective function of the denoising autoencoder to include an $$MxM$$ matrix of prior knowledge relationships between the elements of the input $$x_t$$. This matrix works as a regularizer and somehow constrains the manifold that the DA is able to learn. Difference with the Bayesian approach is that this method adds priors to the reconstruction errors while the Bayesian approach applies the priors to model parameters 



- Contribution Analysis (CA): looking at the elements of the reconstruction error to find the one with most contribution to error. 

---
4. [Ubalanced dataset for Fraud](http://rvc.eng.miami.edu/Paper/2015/Yan_ISM2015.pdf)

- The minority class is actually class of interest which is the fraud instances.

---
## Strategies for handling missing data'


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


