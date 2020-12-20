import pandas as pd
from fbprophet import Prophet
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.functions import current_date


if __name__ == "__main__" :
 spark = SparkSession.builder.appName("TimeSeries").config("spark.hadoop.fs.s3a.access.key", "XX").config("spark.hadoop.fs.s3a.secret.key", "XX").config("fs.s3a.endpoint", "s3-us-east-2.amazonaws.com").getOrCreate()
 
 sc=spark.sparkContext
 sc.setSystemProperty("com.amazonaws.services.s3.enableV4", "true")


 print("Start of Code")
 df = pd.read_csv('s3a://bucket/input/weekly_sales_data.csv')
 df.info()
 df['date'] = pd.to_datetime(df['date'], infer_datetime_format=True)
 print ("\nFeatures : \n" ,df.columns.tolist())
 print ("\nMissing values :  ", df.isnull().any())
 print ("\nUnique values :  \n",df.nunique())
 
 df_base = df.copy()
 df['store_id'].value_counts()
 item_df = df.set_index('date')
 
 sdf = spark.createDataFrame(df)
 sdf.show(5)
 sdf.printSchema()
 sdf.count()
 sdf.select(['store_id']).groupBy('store_id').agg({'store_id':'count'}).show()


 sdf.createOrReplaceTempView("sales")
 spark.sql("select store_id, count(*) from sales group by store_id order by store_id").show()
 sql = 'SELECT store_id, date as ds, sum(sales) as y FROM sales GROUP BY store_id, ds ORDER BY store_id, ds'
 spark.sql(sql).show()

 sdf.explain()
 store_part = (spark.sql( sql ).repartition(spark.sparkContext.defaultParallelism, ['store_id'])).cache()
 store_part.explain()

 result_schema =StructType([
  StructField('ds',TimestampType()),
  StructField('store_id',IntegerType()),
  StructField('y',DoubleType()),
  StructField('yhat',DoubleType()),
  StructField('yhat_upper',DoubleType()),
  StructField('yhat_lower',DoubleType())
  ])

 @pandas_udf( result_schema, PandasUDFType.GROUPED_MAP )
 def forecast_sales( store_pd ):
  model = Prophet(interval_width=0.95,seasonality_mode = 'multiplicative', weekly_seasonality=True, yearly_seasonality=True)
  model.fit( store_pd )
  future_pd = model.make_future_dataframe(
    periods=5, 
    freq='w'
    )
  forecast_pd = model.predict( future_pd )  
  f_pd = forecast_pd[ ['ds','yhat', 'yhat_upper', 'yhat_lower'] ].set_index('ds')
  st_pd = store_pd[['ds','store_id','y']].set_index('ds')
  results_pd = f_pd.join( st_pd, how='left' )
  results_pd.reset_index(level=0, inplace=True)
  results_pd['store_id'] = store_pd['store_id'].iloc[0]
  return results_pd[ ['ds', 'store_id','y', 'yhat', 'yhat_upper', 'yhat_lower'] ]

 results = (store_part.groupBy('store_id').apply(forecast_sales).withColumn('training_date', current_date()))
 results.cache()
 results.explain()
 results.coalesce(1)

 results.count()

 results.createOrReplaceTempView('forecasted')
 spark.sql("select store_id, count(*) from forecasted group by store_id").show()
 final_df = results.toPandas()
 final_df.to_csv("s3a://bucket/output/weekly_sales_forecast.csv")


 print("End of Code")
 spark.stop()
