package preprocessing

import java.io.File

import org.apache.commons.io.FileUtils
import org.apache.spark.ml.feature.{MinMaxScaler, MinMaxScalerModel, StringIndexer}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.linalg.DenseVector
import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by bruce on 2016/4/5.
  */
object LibSVM_Scaling {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("appName").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val sourceFile = "C://Spark//PricePrediction//result//ny//LibSVM_Derivative//part-00000"
    val resultFolder = "C://Spark//PricePrediction//result//ny//LibSVM_Scaling"

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
    val resultDataFrame: DataFrame = sqlContext.sql("select label,features, scaledFeatures from test")

    resultDataFrame.foreach {
      r: Row =>
        val features = r(2).asInstanceOf[DenseVector]
        var result = r(0) + ""
        for (f <- 0 to features.size - 1) {
          println(" " + (f + 1) + ":" + features(f));
          result = result.concat(" " + (f + 1) + ":" + features(f))
        }
        result
    }

    resultDataFrame.map {
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
