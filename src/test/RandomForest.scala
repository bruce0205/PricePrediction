package test

import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.sql.{DataFrame, SQLContext}
import org.apache.spark.{SparkContext, SparkConf}

/**
  * Created by bruce on 2016/5/22.
  */
object RandomForest {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("appName").setMaster("local")
    val sc = new SparkContext(conf)

    val sqlContext = new SQLContext(sc)
    val sourceFile = "C://Spark//PricePrediction//result//LibSVM//part-00000"
    val data: DataFrame = sqlContext.read.format("libsvm").load(sourceFile)
    data.show(20)

    val rf = new RandomForestRegressor()
    .setMaxBins(80)
    .setMaxDepth(4)
    .setNumTrees(100)
    .setImpurity("variance")
    .setFeatureSubsetStrategy("auto")


  }
}
