package building

import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by bruce on 2016/4/5.
  */
object Lasso {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("appName").setMaster("local")
    val sc = new SparkContext(conf)
    build(sc)
  }

  def build(sc: SparkContext): Unit = {
    val sourceFile = "C://Spark//PricePrediction//result//ny//LibSVM_Scaling//part-00000"

    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, sourceFile)

    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    val numIterations = 100 // Set the number of iterations for SGD. Default 100.
    val stepSize = 1 // Set the initial step size of SGD for the first step. Default 1.0. In subsequent steps, the step size will decrease with stepSize/sqrt(t)
    val regParam = 0.01 // Set the regularization parameter. Default 0.01
    val minBatchFraction = 1 // Set fraction of data to be used for each SGD iteration. Default 1.0
    val initialWeights = 11 // Initial set of weights to be used. Array should be equal in size to the number of features in the data.

    // parameter = trainingData, numIterations, stepSize, regParam, miniBatchFraction, initialWeights
    // default parameters: {stepSize: 1.0, numIterations: 100, regParam: 0.01, miniBatchFraction: 1.0}.
    val model: LassoModel = LassoWithSGD.train(trainingData, numIterations, stepSize, regParam, minBatchFraction) // infinite if without feature scaling

    println("model.toString : " + model.toString())
    println("model.weights : " + model.weights)
    println("model.intercept : " + model.intercept)

    val labelsAndPredictions = testData.map { point: LabeledPoint =>
      val prediction: Double = model.predict(point.features)
      (point.label, prediction)
    }

    val regressionMetrics = new RegressionMetrics(labelsAndPredictions)
    val rmse = regressionMetrics.rootMeanSquaredError
    println("Lasso ===> rmse : ")
    println(rmse)
  }
}
