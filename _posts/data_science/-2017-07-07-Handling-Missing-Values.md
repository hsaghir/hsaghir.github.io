---
layout: article
title: NLP
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
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

[IDEA: use GANs/VAEs to learn the learn the distribution of the column with missing value from other columns? i.e. learn a generative model for the column with missing value and generate data for the missing value conditioned on other attributes (i.e. infoGAN)]

