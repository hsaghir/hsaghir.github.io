

# Tensorflow with Spark

- Tensorflow is a library for facilitating two things, 1) defining a symblic computational graph for various mathematical statements 2) an optimized computation engine that can do computations on GPUs (similar to numpy for cpu). 

- You define nodes of the computational graph by defining a placeholders, its name and its type like this:

```
x=tf.placeholder(tf.int32, name='x')
y=tf.placeholder(tf.int32, name='y')
```

- At this point no computation is done, you only manually defined two nodes x and y. By writing a mathematical statement between these two manually defined nodes, tensorflow automatically generates the rest of the computational graph for the statement. For example:

```
output=tf.add(x, y*3, name='z')
```

- The above statement automatically generates the rest of nodes required to make the computational graph for calculating the statement. At this point still no computation has been done. We have only defined the graph. If we want to do computations on this graph we have to define a session and send values through this computational graphs. This is done like this:

```
session= tf.Session()
output_value=session.run(output, {x:3, y:4})
```

## Running Tensorflow with distributed file systems
- If our data sits in Hadoop and we want to use Tensorflow, there are two ways, 1) we can use PySpark and run Tensorflow in a reduce operation, or better yet, 2) iterate over the resulting query and pass it into the local model as a placeholder. This requires the conversion of the Spark dataframe (distributed) to pandas dataframe (local). Pandas is not a distributed computing library and it runs locally on local memory. So everything after coversion to pandas is completely local including tensorflow runs. 

- The above requires conversion of data types from Hadoop HDFS, to Spark Tungesten binary, to Spark Java, to Python Pickle, to Tensorflow C++ object which is very costly. Tensorframe library does this data type conversion in one fell swoop.

### Reading data from Hadoop directly to Tensorflow
- This is a new feature in tensorflow explained here: [TensorFlow with HDFS](https://www.tensorflow.org/deploy/hadoop) 


### TensorFrame
- As mentioned, the computational graph in tensorflow is decoupled from the optimized low-level GPU computation library. Therefore, in theory we can use only one of these two parts. Spark's TensorFrame library does exactly this by using the computational graph defined in tensorflow and run it on a cluster with Spark dataframes. Here is an example:

```
# tensorflow computational graph defined above
import tensorflow as tf
x=tf.placeholder(tf.int32, name='x')
y=tf.placeholder(tf.int32, name='y')
output=tf.add(x, y*3, name='z')


# an spark dataframe
df=sqlContext.creatDataFrame(..)


# using tensorframe library to run the computational graph 
# on each row of the dataframe. 

import tensorframe as tfs
output_df=tfs.map_rows(output, df)


# The computation is done only when the data needs to be used
output_df.collect()

```



