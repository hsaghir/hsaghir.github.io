# Fraud Detection


## Business Problem: Fraud Detection
- Fraud detection is a very big problem. For an organization like RBC that's in the order of 100s of millions per year. 

- Currently RBC does not use any machine learning in their fraud detection unit. 

- There are tons of startups providing analytics/ML solutions, but the main concern of the bank is data privacy. They are not comfortable giving customer data to third parties, their brand will be at risk if they do, and there might be some potential regulatory risks. 

---


## In the news

[Orange tests DL to identify fraud](https://blogs.wsj.com/cio/2016/03/14/orange-tests-deep-learning-software-to-identify-fraud/)

[DL infrastructure in production](https://insights.ubuntu.com/2016/04/25/making-deep-learning-accessible-on-openstack/)


[How PayPal beats the bad guys with machine learning](http://www.infoworld.com/article/2907877/machine-learning/how-paypal-reduces-fraud-with-machine-learning.html)

- PayPal uses three types of machine learning algorithms for risk management: linear, neural network, and deep learning. Experience has shown PayPal that in many cases, the most effective approach is to use all three at once. we “ensemble” them together!

- detect fraud: "we can separate the good from the bad with one straight line". When linear fails, neural network. Also,tree-based, mimicking a human being when we have to make a judgment -- for example, we would say if it’s raining, I’ll take an umbrella.

---

## Data

- Almost no publicly available datasets for fraud detection (legal/competition)

- Previous studies (pre 2010)

<img src="/images/fraud_detection/Fraud_data_size_pre2010.png" alt="Scatter plot of the data size from 40 unique and published fraud detection papers (pre-2010) within common fraud types." width="350" height="350"> | <img src="/images/fraud_detection/dataset_balance.png" alt="Scatter plot of the percentage of fraud and percentage of test of entire data set. 19 unique and published fraud detection papers within common fraud types were used." width="350" height="350">

---
### RBC Data

- About 2 million transactions a day. 
- 2 years of transaction data, 200GB of data. 

- Columns in the database in the order of 1000: every possible credit card event has columns associated to it (transaction types,  activating cards, - location, date, time, aggregate data, last time client made payment, system oriented, apple pay, etc). Therefore, the database is very spare since not all events happen at the same time. 

--- 

#### Fraud labeling process:

1. client calls or fraud detection system flags.
2. Fraud analyst confirms it. 

- Fraud detection system is called Prism and is basically a rule base

- If I want to shadow fraud analysts should coordinate with -> Mike Attey

---

## Emprical work
5. [Classification pipeline](George Dahl)

- Neural net workflow for "Large-scale malware classification using random projections and neural networks":

- Feature extraction
- Feature selection

- Dimensionality reduction using random projections: Even after feature selection, the input dimensionality is still quite large (179 thousand), although there is a lot of sparsity. Naive neural net training on such high dimensional input is too expensive. To make the problem more manageable, we used the very sparse random projections.

- We project each input vector into a much lower dimensional space (a few thousand dimensions) using a sparse projection matrix R with entries sampled iid from a distribution over {0, 1, −1}. Entries of 1 and -1 are equiprobable and $$P(Rij = 0) = 1− \frac{1}{\sqrt(d)}$$, where $$d$$ is the original input dimensionality. Another way to view the random projection in the context of neural network training is that the random projection step forms a layer with linear hidden units in which the weight matrix is not learned and is instead simply set to R.

- Classifier (ensemble NN!?)

---




### TSNE Exploration
 - I have now stored both Auth and Fraud data (all columns) into proper csv files on our local server. I have  0) downloaded data to local server (I have currently downloaded only 2 month of transactions due to space limitations on our local server, @jateeq any solutions? 1) selected columns that make sense 2) explored different encoding methods for categorical variables and chose and performed binary encoding 3) cleaned the data to remove entries that don't make sense  4) randomly subsampled of month 6 of 2016 of transactions to make the good transactions class and used the fraud instances for month 6 of 2016 as fraud class. In this dataset that I will call dataset6, there are about 65k transactions in the good transactions class and about 8k fraudulent transactions in the fraud class. There are a few further processing I need to do to make this dataset in the proper shape for supervised learning 5) I have performed an exploratory data analysis using tsne dimensionality reduction method on my dataset6. Here is how it looks like:


-  The blue dots are good transactions while the red are fraudulent transactions. There are clearly some structure in the data which is a very good sign for our supervised learning efforts. My next step will be to 1) make dataset6 ready for a supervised learning (minor processing) 2) Perform some supervised learning methods on dataset6 to get a baseline 3)  work on semi-supervised learning methods

---

## Classification

So I am finished with cleaning up the data so I ran a classifier to get a baseline. Here are my results up to now: 
I classified the data using random forests which resulted in perfect classification metrics!

accuracy: 1.0
('f1:', 1.0)
('precision:', 1.0)
('recall:', 1.0)

So I trained the model with only 10% of data (90% of data held out for test). and then ran a single tree instead of the random forests with the 10% of data. Still results are near perfect. Confusion matrix and feature importance figures are below. 



So I ran a couple of experiments to see why: 

- checked for no overlap between training and test sets (there is no overlap)
- the classification with even less than 10% of the data (90% test) is still perfect!
- seem like a couple of features are highly correlated with fraud.
- In the extreme case, feature "LG3_ADS_EXP_DATE_MATCH_FLAG" [Compares the expiration date received in the authorization with the expiration date on file.] alone can predict fraud with precision 87% and recall 92% !
- It might be that some features like the above mis-match between expiration date on card and on file is highly indicative of fraud in the dataset that we have. However, it is highly unlikely that the problem of credit card fraud can be almost solved by simply checking if the expiration date reported in transaction matches the one on file!

- Given that I had previously manually selected the feature set I am working with to make sure I am not including any features that don't make sense, I believe the next step would be to ask experts of data on the business side to annotate the feature and tell us what columns are good to use.

- The other possibility is that this dataset is not the proper dataset for fraud detection task.

---

- Linear classifiers like logistic regression and linear SVM are not able to classify the data at all. They put all the data into the normal class.

- While the tree based classifiers can perfectly classify the data. This is probably due to the binary encoding of categorical variable that makes the problem completely nonlinear and tough for linear classifiers. However, the results of the tree-based algorithms show that there are some columns in the data that are highly correlated with the class. I suspect that if such columns are one-hot encoded instead of binary encoding, linear classifiers will also return perfect classification score.

These experiments further strengthen my hypothesis that there are some categorical columns in the data that are highly correlated with fraud. The next step for me would be to see exactly what columns and values correlate with fraud to further understand the correlation.

---
### Data leakage solved

- So I performed joins on the hadoop dataset again and downloaded the data again with the columns that I had chosen (117/964 columns). I removed columns with high percentage of missing values and encoded the data again. In the process of encoding I found the reason for label leakage.

- the label leakage issue was caused by the two classes being encoded to binary dummy variables separately in my code resulting in same categories being encoded to two separate codes. This resulted in the classes having  different encoding and thus the classifier was able to tell all instances apart. After fixing this issue, removing missing-value-heavy columns and a using fresh new set of data, the label leakage problem was solved. The classification results are very good as follows with about 100k training samples in a balanced dataset (i.e. 50% fraud + 50% good transactions). Test and validation sets each have 50k balanced samples. 
 
accuracy: 0.953424228791
('f1:', 0.95309762560583666)
('precision:', 0.95356122669158261)
('recall:', 0.95306584292927599)
('TPR' = 0.9683)
('TNR' = 0.9387)

- I performed a feature importance analysis and the features that come up as important make a lot of sense. The top 10 features are as follows:

[('lg3_transaction_amount', 0.048593633073846253),
('lg3_pos_entry_mode_1', 0.046413134597429372),
('lg3_chip_contactless_flag_0', 0.033354790771903599),
('lg3_prior_available_money', 0.029687734257986521),
('lg3_source_amount', 0.02944274734365614),
('lg3_outstanding_auth', 0.028880774759200225),
('lg3_dh_tran_type_0', 0.028122791104801533),
('lg3_cash_limit', 0.025143854777954174),
('lg3_cvv_cvc_response_2', 0.024062528825336153),
('lg3_tmp_merch_1', 0.021094125403098629),
('lg3_current_balance', 0.019684078659600555),

- Althought this classification accuracy seems very good with very simple feature engineering, it is important to note that the dataset is balanced and not really reflective of reality. In real life the dataset is more like (i.e. fraud 0.1% - 99.9% good transactions ). My next step will be to get a supervised baseline for the unbalanced dataset, after which I will start implementing semi-supervised methods to see if they can improve the performance from baseline. 

- Regarding vetting the features, it will be important to know what features are present when the transaction data comes in real-time since only these features can be used in predicting a fraud score for each transaction. 

---
## Unbalanced data:
- In fraud detection, the metrics of importance are actually first how many of the fraud instances the model is able to correctly label as fraud i.e. specificity or 'True negative rate' and second how many of good transactions is the model able to label as good i.e. sensitivity or 'True positive rate'. 

- While in the balanced data problem the model performance is very good at $$TPR = 96.83% $$ and $$TNR = 93.87%$$, in the unbalanced classification problem where the negative to positive class ration is $$\frac{fraud}{good} = 0.001$$, the model's performance is detecting fraud samples is underwhelming at $$TNR = 48.44 % $$ (as follows). 

('f1:', 0.8203)
('precision:', 0.9730)
('recall:', 0.7422)
('TPR:', 0.9999)
('TNR:', 0.4844)

- I believe this concludes the first phase of the fraud project by establishing a supervised baseline of 50% (unbalanced) fraud detection rate. I am now going to switch my attention to strategies for improving this rate using oversampling (supervised) and anomaly-detection (semi-supervised methods.)

---
# Metric area under the ROC curve
- The area measures discrimination, that is, the ability of the test to correctly classify those with and without the disease. Consider the situation in which patients are already correctly classified into two groups. You randomly pick on from the disease group and one from the no-disease group and do the test on both. The patient with the more abnormal test result should be the one from the disease group. The area under the curve is the percentage of randomly drawn pairs for which this is true (that is, the test correctly classifies the two patients in the random pair).

