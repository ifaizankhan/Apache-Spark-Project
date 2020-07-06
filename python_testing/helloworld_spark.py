from __future__ import print_function
import pyspark
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import desc
from pyspark.sql import SQLContext

from pyspark.sql.types import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from functools import reduce

sc = pyspark.SparkContext('local')
spark = SparkSession(sc)

sql = SQLContext(sc)
sc.setLogLevel("ERROR")


csv_data = spark.read.format('csv').options(header='true').load('time_series_19-covid-Confirmed_archived_0325.csv')

df_Australia = csv_data.na.fill(0)

country = ['Australia']
df_Australia = df_Australia.filter(df_Australia['Country/Region'].isin(country))


columns_to_drop = ['Province/State', 'Lat', 'Long']
df_Australia = df_Australia.drop(*columns_to_drop)

df_Australia = df_Australia.select(df_Australia.columns[0:1] + df_Australia.columns[11:])



newNames = []
for count,ele in enumerate(df_Australia.columns,-1):
    if(count==-1):
        newNames.append('Index')
        continue
    newNames.append(str(count))


oldColumns = df_Australia.schema.names
# print(oldColumns)
# print("!!!!!!!!!!!!")
# print(newNames)

df = reduce(lambda df_Australia, idx: df_Australia.withColumnRenamed(oldColumns[idx], newNames[idx]), range(len(oldColumns)), df_Australia)
# df.printSchema()
#df.show()

# df_Australia.show()
# df_Australia = df_Australia.drop('sum(Country/Region)')


#df.groupby('Country/Region').sum().show()


exprs = {x: "sum" for x in df.columns}
#print(exprs)
df = df.groupBy('Index').agg(exprs)
df_Australia = df
#print(type(df))
df.show(vertical=True)


# for c, n in zip(df_Australia.columns[:], newNames):
#     df_Australia=df_Australia.withColumnRenamed(c, str(n))


# df_Australia = df_Australia.toPandas().set_index("Index").transpose() 
# df_Australia['Dates'] = df_Australia.index # New column for Dates
# df_Australia = df_Australia.rename(columns={"Australia": "Cases"})




# # making schema for converting back to pyspark dataframe
# mySchema = StructType([ StructField("Cases", DoubleType(), True)\
#                         ,StructField("Dates", StringType(), True)\
#                        ])
# # converting back to pyspark dataframe
# df_Australia = sql.createDataFrame(df_Australia,mySchema)
# df_Australia = df_Australia.withColumn("Dates", df_Australia["Dates"].cast(IntegerType()))



# from pyspark.ml.feature import VectorAssembler

# vectorAssembler = VectorAssembler(inputCols = ['Dates'], outputCol = 'features')
# vdf_Australia = vectorAssembler.transform(df_Australia)
# vdf_Australia = vdf_Australia.select(['features', 'Cases'])
# #vdf_Australia.show()



# splits = vdf_Australia.randomSplit([0.965, 0.035])
# train_df = splits[0]
# #test_df = splits[1]

# # train_df = vdf_Australia.limit(50)
# test_df = vdf_Australia.orderBy(desc("features")).limit(2)


# # train_df.show()
# # print('---------------------------------------------------------')
# # test_df.show()

# from pyspark.ml.regression import LinearRegression
# lr = LinearRegression(featuresCol = 'features', labelCol='Cases', maxIter=2000, regParam=0.3, elasticNetParam=0.8)
# lr_model = lr.fit(train_df)
# print("Coefficients: " + str(lr_model.coefficients))
# print("Intercept: " + str(lr_model.intercept))



# trainingSummary = lr_model.summary
# print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
# print("r2: %f" % trainingSummary.r2)

# train_df.describe().show()


# lr_predictions = lr_model.transform(test_df)
# lr_predictions.select("prediction","Cases","features").show()

# from pyspark.ml.evaluation import RegressionEvaluator
# lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
#                  labelCol="Cases",metricName="r2")

# print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))



# test_result = lr_model.evaluate(test_df)

# print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)


# print("numIterations: %d" % trainingSummary.totalIterations)
# print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
# trainingSummary.residuals.show()

# predictions = lr_model.transform(test_df)
# predictions.select("prediction","Cases","features").show()

