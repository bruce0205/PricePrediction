package test

import org.apache.spark.ml.feature.{Normalizer, MinMaxScaler, StandardScaler}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkContext, SparkConf}

/**
  * Created by bruce on 2016/4/24.
  */
object Normalizer {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("appName").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val sourceFile = "C:\\Spark\\PricePrediction\\result\\LibSVM\\part-00000"

    val dataFrame = sqlContext.read.format("libsvm").load(sourceFile)

    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normFeatures")
      .setP(1.0)

    val l1NormData = normalizer.transform(dataFrame)
    l1NormData.show()

    // Normalize each Vector using $L^\infty$ norm.
    val lInfNormData = normalizer.transform(dataFrame, normalizer.p -> Double.PositiveInfinity)
    lInfNormData.show()
  }
}
