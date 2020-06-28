from pyspark import SparkContext

sc = SparkContext('local', 'hello world')
sc.setLogLevel("ERROR")

print("Hello World!")
