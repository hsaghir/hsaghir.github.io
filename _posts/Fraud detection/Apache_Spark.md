

# Distributed Computing


- Distributed computing consists of having multiple executors cooperate to do a job. Each executor has its own allocated cpu, memory and disk space. A resource manager breaks down a job and assigns responsibilities (functions) to each executor. Each executor applies its assigned function to its own data. This phase is called a "map". Then the resource manager combines the results from all executors. This is called a "Reduce" operation. 

- Map/Reduce structure of working with big data usually admits only linear operations; nonlinear models don't really work well with the Map/Reduce workflow. In other words, there is a kind of tradeoff between amount of data and model complexity. A good way of getting passed this problem is to train complex models with more managable data sizes.

<img src="/Users/hamid/Documents/hsaghir.github.io/images/fraud_detection/big_data_model_complexity.png" alt="Data vs. Model Complexity " width="350" height="350">

---

## Apache Spark

- started in 2009 at Berkeley, it is a general-purpose data processing engine across a cluster, and an API-powered toolkit which data scientists and application developers incorporate into their applications to rapidly query, analyze and transform data at scale. Spark sends the code to the data and executes it where the data lives.

- Spark is capable of handling several petabytes of data at a time, distributed across a cluster of thousands of cooperating physical or virtual servers. It is faster than alternative approaches like Hadoop's MapReduce, which tends to write data to and from computer hard drives between each stage of processing. 

---

- Spark normally runs on an existing big data cluster. These data clusters are managed through a resource manager software layer which assigns taks. These clusters are often also used for Hadoop jobs, and Hadoop's YARN resource manager will generally be used to manage that Hadoop cluster. Spark can also run just as easily on alternative cluster resource managers like Apache Mesos.

- Hadoop MapReduce was an early cluster manager which strung multiple MapReduce jobs together to create a data pipeline. MapReduce reads and writes data to disk in-between task which is inefficient. Spark does all activities in memory (RAM) which is much faster. Hadoop now has YARN cluster manager and isn't dependent upon Hadoop MapReduce for cluster managment anymore. 

- Spark is not a replacement for Hadoop. Nor is MapReduce dead. Spark can run on top of Hadoop, benefiting from Hadoop's cluster manager (YARN) and underlying storage (HDFS, HBase, etc.). Spark can also run completely separately from Hadoop, integrating with alternative cluster managers like Mesos and alternative storage platforms like Cassandra and Amazon S3.

---
## Hadoop/Hive vs. Spark

- Hadoop includes a distributed file system (HDFS) and a parallel processing engine called "MapReduce". Hive and Pig make Map and Reduce jobs much easier to write using a SQL-based access method for working with structured data within Hadoop. 

- Spark is also a parallel processing engine but requires a cluster manager and a distributed storage system. For cluster management, Spark supports standalone (native Spark cluster), Hadoop YARN, or Apache Mesos. For distributed storage, Spark can interface with a wide variety, including Hadoop Distributed File System (HDFS), MapR File System (MapR-FS), Cassandra, OpenStack Swift, Amazon S3, Kudu, or a custom solution can be implemented.

---

## Hadoop offers Spark:

- YARN resource manager, which takes responsibility for scheduling tasks across available nodes in the cluster;

- Distributed File System, which stores data when the cluster runs out of free memory;

- Disaster Recovery capabilities, inherent to Hadoop, which enable recovery of data when individual nodes fail.

- Data Security, which becomes increasingly important as Spark tackles production workloads in regulated industries such as healthcare and financial services. 


---
## Spark Components

- Apache Spark project stack comprises of Spark Core and four optimized libraries i.e. 
    + SparkSQL: Spark's module for working with structured data. It supports the open source Hive project, and its SQL-like HiveQL query syntax. Spark SQL also supports JDBC and ODBC connections, enabling a degree of integration with existing databases, data warehouses and business intelligence tools. JDBC connectors can also be used to integrate with Apache Drill, opening up access to an even broader range of data sources.
    + Spark Streaming: Supports scalable and fault-tolerant processing of streaming data, and can integrate with established sources of data streams like Flume (optimized for data logs) and Kafka (optimized for distributed messaging). 
    + MLLib: Machine learning library for Spark that builds on top of Spark resilient distributed databases (RDD).
    + GraphX: Supports analysis and computation over graphs of data, and supports a version of graph processing's Pregel API. GraphX includes a number of widely understood graph algorithms, including PageRank.
    + Spark for DL: SparkNet: Integrates running Caffe with Spark. Sparkling Water: Integrates H2O with Spark. DeepLearning4J: Built on top of Spark. TensorFlow on Spark (experimental)

---

## Understanding SparkContext, SQLContext, and HiveContext
- The first step for any spark app (e.g. an interactive spark shell) is to create a  project manager (i.e. an SparkContext object). The project manager will hire a bunch of parallel workers (the cluster) through a worker resource manager (cluster manager). The app will ask SparkContext to do a set of jobs, SparkContext will ask the resource manager for workers. If resources are available, the resource manager will allocate workers to that SparkContext instance with the specified cores and memory. SparkContext will then break down the job into stages and tasks, and schedules and assigns them to the allocated workers.

- SparkSQL is one of Spark's modules that is used to work with structured data (i.e. tables, dataframes) and query them. SparkSQL has a SQLContext and its super-set HiveContext which allow one to run SQL queries and Hive commands.

---

- When you run spark shell, it automatically creates a SparkContext as 'sc' and a HiveContext as 'sqlContext'.

- To create a SparkContext, we need to create a SparkConf object that stores config params like number/memory/etc for executors. This object will be passed to SparkContext. 

- SparkConf() has methods like setAppName() or set() to define parameters.  

```
import org.apache.spark.SparkConf as SConf
conf= SConf.setAppName('myApp').set('spark.executor.memory','2g')...
```

---

- We pass this conf object to SparkContext so that the driver knows how to access the cluster. 

```
import org.apache.spark.SparkContext as SC
sc=SC(conf)
```

---

## Spark basic object 
- RDD is the most basic data structure in Spark. It is a fault tolerant collection of elements which support in-memory data storage. When possible, RDDs remain in memory, otherwise they are moved to disk. RRD can be created in two ways. 1) parallelizing an existing collection 2)referencing a data structure in an external storage system (HDFS). 

- a DataFrame is a restricted RRD in tabular format, which allows Spark to run certain optimizations on the finalized query. One can go from a DataFrame to an RDD via its rdd method, and you can go from an RDD to a DataFrame (if the RDD is in a tabular format) via the toDF method. In general it's better to use dataframe if possible. 

- Once data is loaded into an RDD, two basic operations can be carried out (that are logged and can be reversed):

    + Transformations, which creates a new RDD by changing the original through a map, filter, etc. Transformations are lazy meaning that they are remembered but not applied until the results are needed. By default each transformed RDD is recomputed each time we call an action on it. However, we can **persist** it for faster access if required. 
    + Actions, a computation on the RDD (e.g reduce, count, etc), that doesn't change the original data.

---


## Map and Reduce operations

- Map is a transformation operation which applies a function to an RDD and creates a new one from the result. A map is only applied when the result is needed. 
- Reduce is an action. When run, it breaks down the map computation into tasks to run on separate machines. Each machine runs both its part of the map on its local data and a local reduction, returning the answer to the driver program. 


## SQLContext
- the entry point to all Spark SQL functionality is the SQLContext class. It needs to be instantiated like this if not already defined:

```
from pyspark.sql import SQLContext
sqlContext=SQLContext(sc)
```


- SparkSQL can convert an RDD of Row objects to a dataframe. To create Row object we use the Row class which gets (column name, value) pairs and creates a row object. For example:

```
row_data= rdd_data.map(lambda r: Row(column0=r[0], column1=r[1]))
```

- We can convert this Row rdd to a dataframe and then register it as a sql table with name "table1" to be able to run sql queries

```
df=sqlContext.createDataFrame(row_data)
df.registerTempTable('table1')
```

- A sample sql query:
```
query_df= sqlContext.sql("""SELECT column0 FROM table1 WHERE column0>5""")
```

- the result of a sql query is an RDD. 

---

## Using other python files with PySpark
- you can simply import other python files in pyspark using following:
```
SparkContext.addPyFile("module.py/.zip")
```



## Spark Out Of Memory remedies:
-   If your nodes are configured to have 6g maximum for Spark (and are leaving a little for other processes), then use 6g rather than 4g, `spark.executor.memory=6g`. Make sure **you're using as much memory as possible** by checking the UI (it will say how much mem you're using)
-   Try using more partitions, you should have 2 - 4 per CPU. IME increasing the number of partitions is often the easiest way to make a program more stable (and often faster). For huge amounts of data you may need way more than 4 per CPU, I've had to use 8000 partitions in some cases!
-   Decrease the **fraction of memory reserved for caching**, using `spark.storage.memoryFraction`. If you don't use `cache()` or `persist` in your code, this might as well be 0. It's default is 0.6, which means you only get 0.4 * 4g memory for your heap. IME reducing the mem frac often makes OOMs go away. **UPDATE:** From spark 1.6 apparently we will no longer need to play with these values, spark will determine them automatically.
-   Similar to above but **shuffle memory fraction**. If your job doesn't need much shuffle memory then set it to a lower value (this might cause your shuffles to spill to disk which can have catastrophic impact on speed). Sometimes when it's a shuffle operation that's OOMing you need to do the opposite i.e. set it to something large, like 0.8, or make sure you allow your shuffles to spill to disk (it's the default since 1.0.0).
-   Watch out for **memory leaks**, these are often caused by accidentally closing over objects you don't need in your lambdas. The way to diagnose is to look out for the "task serialized as XXX bytes" in the logs, if XXX is larger than a few k or more than an MB, you may have a memory leak. See <https://stackoverflow.com/a/25270600/1586965>
-   Related to above; use **broadcast variables** if you really do need large objects.
-   If you are caching large RDDs and can sacrifice some access time consider serialising the RDD <http://spark.apache.org/docs/latest/tuning.html#serialized-rdd-storage>. Or even caching them on disk (which sometimes isn't that bad if using SSDs).
-   (**Advanced**) Related to above, avoid `String` and heavily nested structures (like `Map` and nested case classes). If possible try to only use primitive types and index all non-primitives especially if you expect a lot of duplicates. Choose `WrappedArray` over nested structures whenever possible. Or even roll out your own serialisation - YOU will have the most information regarding how to efficiently back your data into bytes, **USE IT**!
-   (**bit hacky**) Again when caching, consider using a `Dataset` to cache your structure as it will use more efficient serialisation. This should be regarded as a hack when compared to the previous bullet point. Building your domain knowledge into your algo/serialisation can minimise memory/cache-space by 100x or 1000x, whereas all a `Dataset` will likely give is 2x - 5x in memory and 10x compressed (parquet) on disk.
 
<http://spark.apache.org/docs/1.2.1/configuration.html>
 
EDIT: (So I can google myself easier) The following is also indicative of this problem:
 
```
java.lang.OutOfMemoryError : GC overhead limit exceeded
```








---
Big Data Engineer role:

Main qualifications: 

- Excellent knowledge of distributed systems and big data infrastructure including Hadoop, Spark, HBase, and Hive/Pig/SQL
- Strong coding skills preferably in Python
- Good knowledge of Python scientific stack. SciPy, Numpy, Pandas, jupyter, matplotlib
- Good data modeling skills
- Ability to quickly build a web app to collect and visualize data with Flask and D3 or an alternative workflow. 

a plus if :
- Knows Machine learning/deep learning
- is familiar with a deep learning library i.e. keras/tensorflow/theano/pytorch/etc



---
I have been trying to work with a large dataset stored on Haddop/Spark to work with batches so that we can do stochastic learning with our favorite machine learning algorithms but it turns out that spark is not designed for stochastic processing meaning that one can not iterate through data in batches to feed to ML algorithm 

a summary of my venture into the world of distributed processing. As most of you know we have been having some trouble with Spark-Tensorflow work flow. The problem stems from the differences in design paradigm of the two systems. 

Spark is designed to efficiently perform operations on all of a large dataset in parallel. For example, when one wants to perform gradient descent using the whole dataset, one can parallelize it by calculating the gradient for partitions of the dataset, adding them all up and do an update calculated from the whole dataset. However, Spark doesn't work well if one wants to do sequential operations on batches of a dataset. For example, most modern machine learning algorithm use Stochastic Gradient Descent (SGD) as the optimization procedure for the learning. SGD takes in batches of data, calculates the gradient and perform update for the batch. 

Stochastic optimization procedure is very important for training Deep Learning and Reinforcement Learning models. 


"Ray" is a python library for distributed computing as a replacement for "Spark" that can handle batch/stochastic processing for modern DL/RL algorithms. 

Ray on github (currently on pre-Alpha):
https://github.com/ray-project/ray

An article on Ray:
https://www.datanami.com/2017/03/28/meet-ray-real-time-machine-learning-replacement-spark/

Michael Jordan (Berkeley) explains this new paradigm very briefly here:
https://youtu.be/bIfB1fj8xGQ?t=28m38s

---

As Machine learning-assisted decision making is becoming commonplace (for example fraud detection), the ability to generate simple explanations for black-box systems becomes very important 

https://arxiv.org/pdf/1606.09517.pdf
https://arxiv.org/pdf/1704.03296.pdf

https://blog.acolyer.org/2016/09/22/why-should-i-trust-you-explaining-the-predictions-of-any-classifier/
