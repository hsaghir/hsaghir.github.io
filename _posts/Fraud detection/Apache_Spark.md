


# Apache Spark

- started in 2009 at Berkeley, it is a general-purpose data processing engine across a cluster, and an API-powered toolkit which data scientists and application developers incorporate into their applications to rapidly query, analyze and transform data at scale. Spark sends the code to the data and executes it where the data lives.

- Spark is capable of handling several petabytes of data at a time, distributed across a cluster of thousands of cooperating physical or virtual servers. It is faster than alternative approaches like Hadoop's MapReduce, which tends to write data to and from computer hard drives between each stage of processing. 

---

- Spark is good at:
    +   Stream processing: Streams of data related to financial transactions, for example, can be processed in real time to identify--and refuse--potentially fraudulent transactions.
    +   Machine learning: Spark's ability to store data in memory and rapidly run repeated queries makes it well-suited to training machine learning algorithms. Running broadly similar queries again and again, at scale, significantly reduces the time required to iterate through a set of possible solutions in order to find the most efficient algorithms.
    +   Interactive ML: running models, view/visualize, modify data/model, iterate
    +   Data Integration: Extract, transform, and load (ETL) processes are often used to pull data from different systems, clean and standardize it, and then load it into a separate system for analysis. Spark (and Hadoop) are increasingly being used to reduce the cost and time required for this ETL process.
    +   Although often closely associated with Hadoop's underlying storage system, HDFS, Spark includes native support for tight integration with a number of leading storage solutions including: MapR (file system and database), Google Cloud Amazon S3, Apache Cassandra, Apache Hadoop (HDFS), Apache HBase, Apache Hive, Berkeley's Tachyon project. 

---

- Spark normally runs on an existing big data cluster. These data clusters are managed through a resource manager software layer which assigns taks. These clusters are often also used for Hadoop jobs, and Hadoop's YARN resource manager will generally be used to manage that Hadoop cluster. Spark can also run just as easily on alternative cluster resource managers like Apache Mesos.

- MapReduce is really a programming model. Hadoop MapReduce was an early cluster manager which strung multiple MapReduce jobs together to create a data pipeline. MapReduce reads and writes data to disk inbetween task which is inefficient. Spark does all activities in memory (RAM) which is much faster. Hadoop now has YARN cluster manager and isn't dependent upon Hadoop MapReduce for cluster managment anymore. 

- Spark is not a replacement for Hadoop. Nor is MapReduce dead. Spark can run on top of Hadoop, benefiting from Hadoop's cluster manager (YARN) and underlying storage (HDFS, HBase, etc.). Spark can also run completely separately from Hadoop, integrating with alternative cluster managers like Mesos and alternative storage platforms like Cassandra and Amazon S3.

---

- Spark project stack is comprised of Spark Core and four optimized libraries i.e. 
    + SparkSQL: Spark's module for working with structured data. It supports the open source Hive project, and its SQL-like HiveQL query syntax. Spark SQL also supports JDBC and ODBC connections, enabling a degree of integration with existing databases, data warehouses and business intelligence tools. JDBC connectors can also be used to integrate with Apache Drill, opening up access to an even broader range of data sources.
    + Spark Streaming: Supports scalable and fault-tolerant processing of streaming data, and can integrate with established sources of data streams like Flume (optimized for data logs) and Kafka (optimized for distributed messaging). 
    + MLLib: Machine learning library for Spark that builds on top of Spark resilient distributed databases (RDD).
    + GraphX: Supports analysis and computation over graphs of data, and supports a version of graph processing's Pregel API. GraphX includes a number of widely understood graph algorithms, including PageRank.
    + Spark for DL: SparkNet: Integrates running Caffe with Spark. Sparkling Water: Integrates H2O with Spark. DeepLearning4J: Built on top of Spark. TensorFlow on Spark (experimental)

---

## Resilient Distributed Dataset (RRD)
- RRD is a at the heart of Spark. It is designed to support in-memory data storage, distributed across a cluster in a manner that is demonstrably both fault-tolerant and efficient. Where possible, these RDDs remain in memory, greatly increasing the performance of the cluster.

- Fault-tolerance is achieved, in part, by tracking the lineage of transformations applied to coarse-grained sets of data. Efficiency is achieved through parallelization of processing across multiple nodes in the cluster, and minimization of data replication between those nodes.

- Once data is loaded into an RDD, two basic operations can be carried out (that are logged and can be reversed):
    + Transformations, which create a new RDD by changing the original through processes such as mapping, filtering, and more;
    + Actions, such as counts, which measure but do not change the original data.

- An additional DataFrames API was added to Spark in 2015. A dataFrame is a distributed collection of data organized into named columns. It is conceptually equivalent to a table in a relational database or a data frame in R/Python, but with richer optimizations under the hood. DataFrames can be constructed from a wide array of sources such as: structured data files, tables in Hive, external databases, or existing RDDs.

---

## Hadoop offers Spark:

- YARN resource manager, which takes responsibility for scheduling tasks across available nodes in the cluster;

- Distributed File System, which stores data when the cluster runs out of free memory;

- Disaster Recovery capabilities, inherent to Hadoop, which enable recovery of data when individual nodes fail.

- Data Security, which becomes increasingly important as Spark tackles production workloads in regulated industries such as healthcare and financial services. 


## Understanding SparkContext, SQLContext, and HiveContext
- The first step for any spark app (e.g. an interactive spark shell) is to create a  project manager (i.e. an SparkContext object). The project manager will hire a bunch of parallel workers (the cluster) through a worker resource manager (cluster resource manager). The app will ask SparkContext to do a set of jobs, SparkContext will breakdown the jobs and asks the worker resource manager for workers. If resources are available, the resource manager will allocate workers to that SparkContext instance with the specified cores and memory. SparkContext will then break down a job into stages and tasks, and schedules and assigns them to the allocated workers.

- SparkSQL is one of Spark's modules that is used to work with structured data (i.e. tables, dataframes) and query them. SparkSQL has a SQLContext and its super-set HiveContext which allow one to run SQL queries and Hive commands

- When you run spark shell, it automatically creates a SparkContext as 'sc' and a HiveContext as 'sqlContext'



- To create a SparkContext, we need to create a SparkConf object that stores config params like number/memory/etc for executors. This object will be passed to SparkContext. 

- SparkConf() has methods likke setAppName() or set() to define parameters.  

```
import org.apache.spark.SparkConf as SConf
conf= SConf.setAppName('myApp').set('spark.executor.memory','2g')...
```

- We pass this conf object to SparkContext so that the driver knows how to access the cluster. 

```
import org.apache.spark.SparkContext as SC
sc=SC(conf)
```


## Strategies for unbalanced classification problem 

- Changing the performance metric:
    + Use the confusio nmatrix to calculate Precision, Recall
    + F1score (weighted average of precision recall)
    + Use Kappa - which is a classification accuracy normalized by the imbalance of the classes in the data
    + ROC curves - calculates sensitivity/specificity ratio.
- Resampling the dataset
    + Essentially this is a method that will process the data to have an approximate 50-50 ratio. 
    + One way to achieve this is by OVER-sampling, which is adding copies of the under-represented class (better when you have little data)
    + Another is UNDER-sampling, which deletes instances from the over-represented class (better when he have lot's of data)

## blob
The question I am trying to answer is whether a new transaction is a good transaction and should go through or is it a fradulent one, in which case it has to be stopped. Transaction data is fundamentally an unbalanced data problem with many good transactions in one class and a small number of fradulent ones in the other. So it is important to combine supervised and unsupervised learning methods to solve this unbalanced data problem. My work is focused on semi-supervised anomaly detection methods based on deep generative models to detect fradulent transactions. 

## Per customer fraud detection (ANGEL - ANomaly Gaurde Event Luncher!?)
- You basically download your transaction data into your Angel app and it gaurds you against fraud by looking for anomallies in your transaction data. e.g. your phone location indicates you are in Toronto while a transaction is reported on your card in EU. Angel will mark that as a very highly anormal transaction and will tell you about it. A Gaussian Process can tell when uncertainties are high and can query the user about it. 



