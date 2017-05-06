package test

import java.io.File

import org.apache.commons.io.FileUtils
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by bruce on 2016/4/25.
  */
object MinMaxScaler {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("appName").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    //val sourceFile = "C:\\Spark\\spark-1.6.1-bin-hadoop2.6\\data\\mllib\\sample_libsvm_data.txt"
    val sourceFile = "C://Spark//PricePrediction//result//LibSVM//part-00000"
    val resultFolder = "C://Spark//PricePrediction//result//Scaling_Test"

    FileUtils.deleteDirectory(new File(resultFolder))

    val dataFrame = sqlContext.read.format("libsvm").load(sourceFile)

    val scaler = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")

    // Compute summary statistics and generate MinMaxScalerModel
    val scalerModel = scaler.fit(dataFrame)

    // rescale each feature to range [min, max].
    val scaledData: DataFrame = scalerModel.transform(dataFrame)
    scaledData.registerTempTable("test")
    val result: DataFrame = sqlContext.sql("select label,features, scaledFeatures from test")

    result.foreach {
      r: Row =>
        val features = r(2).asInstanceOf[DenseVector]
        var result = r(0) + ""
        for (f <- 0 to features.size - 1) {
          println(" " + (f + 1) + ":" + features(f));
          result = result.concat(" " + (f + 1) + ":" + features(f))
        }
        result
    }

    result.map {
      r: Row =>
        val features = r(2).asInstanceOf[DenseVector]
        var result = r(0) + ""
        for (f <- 0 to features.size - 1) {
          println(" " + (f + 1) + ":" + features(f));
          result = result.concat(" " + (f + 1) + ":" + features(f))
        }
        result
    }.repartition(1)
      .saveAsTextFile(resultFolder)

  }
}
