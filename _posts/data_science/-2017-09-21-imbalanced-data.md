---
layout: article
title: Dealing with imbalanced data
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---


Re-sampling techniques are divided in two categories:

1.  Under-sampling the majority class(es).
2.  Over-sampling the minority class.
3.  Combining over- and under-sampling.
4.  Create ensemble balanced sets.

Below is a list of the methods currently implemented in this module.

-   Under-sampling

    1.  Random majority under-sampling with replacement
    2.  Extraction of majority-minority Tomek links [[1]](https://github.com/scikit-learn-contrib/imbalanced-learn#id20)
    3.  Under-sampling with Cluster Centroids
    4.  NearMiss-(1 & 2 & 3) [[2]](https://github.com/scikit-learn-contrib/imbalanced-learn#id21)
    5.  Condensend Nearest Neighbour [[3]](https://github.com/scikit-learn-contrib/imbalanced-learn#id22)
    6.  One-Sided Selection [[4]](https://github.com/scikit-learn-contrib/imbalanced-learn#id23)
    7.  Neighboorhood Cleaning Rule [[5]](https://github.com/scikit-learn-contrib/imbalanced-learn#id24)
    8.  Edited Nearest Neighbours [[6]](https://github.com/scikit-learn-contrib/imbalanced-learn#id25)
    9.  Instance Hardness Threshold [[7]](https://github.com/scikit-learn-contrib/imbalanced-learn#id26)
    10. Repeated Edited Nearest Neighbours [[14]](https://github.com/scikit-learn-contrib/imbalanced-learn#id33)
    11. AllKNN [[14]](https://github.com/scikit-learn-contrib/imbalanced-learn#id33)

-   Over-sampling

    1.  Random minority over-sampling with replacement
    2.  SMOTE - Synthetic Minority Over-sampling Technique [[8]](https://github.com/scikit-learn-contrib/imbalanced-learn#id27)
    3.  bSMOTE(1 & 2) - Borderline SMOTE of types 1 and 2 [[9]](https://github.com/scikit-learn-contrib/imbalanced-learn#id28)
    4.  SVM SMOTE - Support Vectors SMOTE [[10]](https://github.com/scikit-learn-contrib/imbalanced-learn#id29)
    5.  ADASYN - Adaptive synthetic sampling approach for imbalanced learning [[15]](https://github.com/scikit-learn-contrib/imbalanced-learn#id34)

-   Over-sampling followed by under-sampling

    1.  SMOTE + Tomek links [[12]](https://github.com/scikit-learn-contrib/imbalanced-learn#id31)
    2.  SMOTE + ENN [[11]](https://github.com/scikit-learn-contrib/imbalanced-learn#id30)

-   Ensemble sampling

    1.  EasyEnsemble [[13]](https://github.com/scikit-learn-contrib/imbalanced-learn#id32)
    2.  BalanceCascade [[13]](https://github.com/scikit-learn-contrib/imbalanced-learn#id32)

The different algorithms are presented in the [sphinx-gallery](http://contrib.scikit-learn.org/imbalanced-learn/auto_examples).

### [](https://github.com/scikit-learn-contrib/imbalanced-learn#references)References:

| [[1]](https://github.com/scikit-learn-contrib/imbalanced-learn#id3) | : I. Tomek, "Two modifications of CNN," In Systems, Man, and Cybernetics, IEEE Transactions on, vol. 6, pp 769-772, 2010. |

| [[2]](https://github.com/scikit-learn-contrib/imbalanced-learn#id4) | : I. Mani, I. Zhang. "kNN approach to unbalanced data distributions: a case study involving information extraction," In Proceedings of workshop on learning from imbalanced datasets, 2003. |

| [[3]](https://github.com/scikit-learn-contrib/imbalanced-learn#id5) | : P. Hart, "The condensed nearest neighbor rule," In Information Theory, IEEE Transactions on, vol. 14(3), pp. 515-516, 1968. |

| [[4]](https://github.com/scikit-learn-contrib/imbalanced-learn#id6) | : M. Kubat, S. Matwin, "Addressing the curse of imbalanced training sets: one-sided selection," In ICML, vol. 97, pp. 179-186, 1997. |

| [[5]](https://github.com/scikit-learn-contrib/imbalanced-learn#id7) | : J. Laurikkala, "Improving identification of difficult small classes by balancing class distribution," Springer Berlin Heidelberg, 2001. |

| [[6]](https://github.com/scikit-learn-contrib/imbalanced-learn#id8) | : D. Wilson, "Asymptotic Properties of Nearest Neighbor Rules Using Edited Data," In IEEE Transactions on Systems, Man, and Cybernetrics, vol. 2 (3), pp. 408-421, 1972. |

| [[7]](https://github.com/scikit-learn-contrib/imbalanced-learn#id9) | : D. Smith, Michael R., Tony Martinez, and Christophe Giraud-Carrier. "An instance level analysis of data complexity." Machine learning 95.2 (2014): 225-256. |

| [[8]](https://github.com/scikit-learn-contrib/imbalanced-learn#id12) | : N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE: synthetic minority over-sampling technique," Journal of artificial intelligence research, 321-357, 2002. |

| [[9]](https://github.com/scikit-learn-contrib/imbalanced-learn#id13) | : H. Han, W. Wen-Yuan, M. Bing-Huan, "Borderline-SMOTE: a new over-sampling method in imbalanced data sets learning," Advances in intelligent computing, 878-887, 2005. |

| [[10]](https://github.com/scikit-learn-contrib/imbalanced-learn#id14) | : H. M. Nguyen, E. W. Cooper, K. Kamei, "Borderline over-sampling for imbalanced data classification," International Journal of Knowledge Engineering and Soft Data Paradigms, 3(1), pp.4-21, 2001. |

| [[11]](https://github.com/scikit-learn-contrib/imbalanced-learn#id17) | : G. Batista, R. C. Prati, M. C. Monard. "A study of the behavior of several methods for balancing machine learning training data," ACM Sigkdd Explorations Newsletter 6 (1), 20-29, 2004. |

| [[12]](https://github.com/scikit-learn-contrib/imbalanced-learn#id16) | : G. Batista, B. Bazzan, M. Monard, ["Balancing Training Data for Automated Annotation of Keywords: a Case Study," In WOB, 10-18, 2003. |

| [13] | *([1](https://github.com/scikit-learn-contrib/imbalanced-learn#id18), [2](https://github.com/scikit-learn-contrib/imbalanced-learn#id19))* : X. Y. Liu, J. Wu and Z. H. Zhou, "Exploratory Undersampling for Class-Imbalance Learning," in IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), vol. 39, no. 2, pp. 539-550, April 2009. |

| [14] | *([1](https://github.com/scikit-learn-contrib/imbalanced-learn#id10), [2](https://github.com/scikit-learn-contrib/imbalanced-learn#id11))* : I. Tomek, "An Experiment with the Edited Nearest-Neighbor Rule," IEEE Transactions on Systems, Man, and Cybernetics, vol. 6(6), pp. 448-452, June 1976. |

| [[15]](https://github.com/scikit-learn-contrib/imbalanced-learn#id15) | : He, Haibo, Yang Bai, Edwardo A. Garcia, and Shutao Li. "ADASYN: Adaptive synthetic sampling approach for imbalanced learning," In IEEE International Joint Conference on Neural Networks (IEEE World Congress on Computational Intelligence), pp. 1322-1328, 2008. |