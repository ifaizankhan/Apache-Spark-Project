import pyspark
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import desc


sc = pyspark.SparkContext('local')
spark = SparkSession(sc)


csv_data = spark.read.format('csv').options(header='true').load('time_series_19-covid-Confirmed_archived_0325.csv')

#dataa.printSchema()
#csv_data.show(3,vertical=True)

df_Australia = csv_data.na.fill(0)

country = ['Australia']
df_Australia = df_Australia.filter(df_Australia['Country/Region'].isin(country))


columns_to_drop = ['Province/State', 'Lat', 'Long']
df_Australia = df_Australia.drop(*columns_to_drop)



exprs = {x: "sum" for x in df_Australia.columns}
df_Australia = df_Australia.groupBy("Country/Region").agg(exprs)


df_Australia.show(3,vertical=True)
