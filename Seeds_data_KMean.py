# Databricks notebook source
# DBTITLE 1,Analysis Wheat Seeds X-ray (Dimensions)data set from University California ML Repo with KMeans Clustering
import pyspark,pyspark.sql
from pyspark.sql import SparkSession

# COMMAND ----------

spark = SparkSession.builder.appName('Seeds').getOrCreate()

# COMMAND ----------

df = spark.read.csv('/FileStore/tables/seeds_dataset.csv',inferSchema ='True', header ='True')

# COMMAND ----------

df.show()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.head(1)

# COMMAND ----------

from pyspark.ml.clustering import KMeans

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

df.columns

# COMMAND ----------

assembler = VectorAssembler(inputCols = df.columns,
                           outputCol ='features')

# COMMAND ----------

final_data = assembler.transform(df)

# COMMAND ----------

final_data.printSchema()

# COMMAND ----------

#scaling Data:
from pyspark.ml.feature import StandardScaler

# COMMAND ----------

scaler = StandardScaler(inputCol='features',
                        outputCol='scaledfeatures')

# COMMAND ----------

#not so requre the as the vlues in df in rage 0 to 25 not so varried but for learing purpose i did scaling 
scaler_model = scaler.fit(final_data)

# COMMAND ----------

final_data = scaler_model.transform(final_data)

# COMMAND ----------

final_data.printSchema()

# COMMAND ----------

final_data.head(1)

# COMMAND ----------

Kmeans = KMeans(featuresCol='scaledfeatures',k=3)

# COMMAND ----------

model = Kmeans.fit(final_data)

# COMMAND ----------

#not so helpful as we scaled the data
#Silhouette with squared euclidean distance
print('Silhouette with squared euclidean distance:')
pdt = model.transform(final_data)
from pyspark.ml.evaluation import ClusteringEvaluator
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(pdt)
print(silhouette)

#with in set sum of squre errors(wssse)
#print(model.computeCost(final_data)) # from spark 2 to spark 3.0.0 

# COMMAND ----------

centers = model.clusterCenters()

# COMMAND ----------

# as k =3 there are 3 centers
print(centers)

# COMMAND ----------

model.transform(final_data).show()

# COMMAND ----------

model.transform(final_data).select('prediction').show()

# COMMAND ----------

print('In above predition is  eithorv0, 1 or 2')
