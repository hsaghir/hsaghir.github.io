

# Distributed Computing


- Distributed computing consists of having multiple executors cooperate to do a job. Each executor has its own allocated cpu, memory and disk space. A resource manager breaks down a job and assigns responsibilities (functions) to each executor. Each executor applies its assigned function to its own data. This phase is called a "map". Then the resource manager combines the results from all executors. This is called a "Reduce" operation. 

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










