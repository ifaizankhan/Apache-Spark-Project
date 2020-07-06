
# MATRICULATION NUMBERS

# Daniyal Saleem (219203408)
# Rubhan Azeem (219203211)
# Faizan khan (219203213)	
# Syed Saad Ahmed (219203029)
# Usama Mazhar (219203368)


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
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import length

#Creating a pysaprk local session using context
sc = pyspark.SparkContext('local')
spark = SparkSession(sc)

#for working on with structured data, using SQLCOntext
sql = SQLContext(sc)
sc.setLogLevel("ERROR")

#loading the csv file
csv_data = spark.read.format('csv').options(header='true').load('time_series_19-covid-Confirmed_archived_0325.csv')

#removing null values
df_Australia = csv_data.na.fill(0)

#Grabbing the records for country Australia only
country = ['Australia']
df_Australia = df_Australia.filter(df_Australia['Country/Region'].isin(country))

#Dropping columns that are not neccessary
columns_to_drop = ['Province/State', 'Lat', 'Long']
df_Australia = df_Australia.drop(*columns_to_drop)

#Selecting and Merging data in range from 1st Feb to 23rd March 
df_Australia = df_Australia.select(df_Australia.columns[0:1] + df_Australia.columns[11:])

#Renaming and preparing the data-frame with respect to index values (0,1,2....)
newNames = []
for count,ele in enumerate(df_Australia.columns,-1):
    if(count==-1):
        newNames.append('Index')
        continue
    newNames.append(str(count))
oldColumns = df_Australia.schema.names

#Encoding the dates into 0,1,2,3....
df_Australia = reduce(lambda df_Australia, idx: df_Australia.withColumnRenamed(oldColumns[idx], newNames[idx]), range(len(oldColumns)), df_Australia)

#Using the sum function and applying aggregate to the sum function for summation of value
exprs = {x: "sum" for x in df_Australia.columns if x is not df_Australia.columns[0]}
df_Australia=df_Australia.groupBy("Index").agg(exprs)

#Renaming columns (dates) as (0,1,2,3....)
for i in df_Australia.columns[1:]:
    df_Australia=df_Australia.withColumnRenamed(i, (i[4:-1]).zfill(2))

#Taking transpose and renaming the columns for processing
df_Australia=df_Australia.toPandas().set_index("Index").transpose() 
df_Australia['Dates'] = df_Australia.index
df_Australia=df_Australia.rename(columns={"Australia": "Cases"})

#making mySchema for converting things back to pyspark dataframe
mySchema = StructType([ StructField("Cases", DoubleType(), True)\
                        ,StructField("Dates", StringType(), True)\
                       ])
df_Australia = sql.createDataFrame(df_Australia,mySchema)

#Type-casting the Dates column in dataframe
df_Australia = df_Australia.withColumn("Dates", df_Australia["Dates"].cast(IntegerType()))

#Calculation of feautre vector for the column with feature variables
vectorAssembler = VectorAssembler(inputCols = ['Dates'], outputCol = 'features')
vdf_Australia = vectorAssembler.transform(df_Australia)
vdf_Australia = vdf_Australia.select(['features', 'Cases'])

#Splitting data to Training data and Test data
splits = vdf_Australia.randomSplit([0.965, 0.035])
train_df = splits[0]
test_df = vdf_Australia.orderBy(desc("features")).limit(2)

#Applying LinearRegression method  
lr = LinearRegression(featuresCol = 'features', labelCol='Cases', maxIter=2000, regParam=0.3, elasticNetParam=0.8)
lr_model = lr.fit(train_df)
print("Coefficients: " + str(lr_model.coefficients))
print("Intercept: " + str(lr_model.intercept))

#Getting the paramters resulting from training summary
trainingSummary = lr_model.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
train_df.describe().show()

#Calculating the Predictions params from the regression evaluator function 
lr_predictions = lr_model.transform(test_df)
lr_predictions.select("prediction","Cases","features").show()
lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                 labelCol="Cases",metricName="r2")
print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
test_result = lr_model.evaluate(test_df)
print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)
print("numIterations: %d" % trainingSummary.totalIterations)
print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
trainingSummary.residuals.show()
predictions = lr_model.transform(test_df)
predictions.select("prediction","Cases","features").show()

#Converting to pandas in order to plot scatter graph
df_plot=df_Australia.toPandas()
df_pred=predictions.toPandas()

#Visualization using graph and plot
plt.scatter(df_plot["Dates"], df_plot["Cases"],  color='black')
plt.scatter([50,51], df_pred["prediction"],  color='red')
plt.title('Covid19 Dataset Prediction')
plt.xlabel('Dates')
plt.ylabel('Cases')
plt.show()

