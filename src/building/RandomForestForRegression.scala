package building

import java.util.Calendar

import org.apache.commons.lang3.time.DurationFormatUtils
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.ListBuffer

/**
  * Created by bruce on 2016/4/5.
  */
object RandomForestForRegression {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("appName").setMaster("local")
    val sc = new SparkContext(conf)

    val numTreesArray: Array[Int] = Array(3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300)
    //val numTreesArray: Array[Int] = Array(5000)
    val maxDepth: Int = 4
    val list = new ListBuffer[(Int, Double, String)]()

    numTreesArray.foreach { f: Int =>
      println("numTrees : " + f)
      val begin: Calendar = Calendar.getInstance()
      val testRMSE = build(sc, f, maxDepth);
      val end: Calendar = Calendar.getInstance()
      val timeDiff = DurationFormatUtils.formatDuration(Math.abs(begin.getTimeInMillis - end.getTimeInMillis), "ss.SS")
      list += Tuple3(f, testRMSE, timeDiff)
    }

    list.foreach { f => println("numTrees/maxDepth/RMSE/timeDiff : " + f._1 + "/" + maxDepth + "/" + f._2 + "/" + f._3) }

  }

  def build(sc: SparkContext, numTrees: Int, maxDepth: Int): Double = {
    val sourceFile = "C://Spark//PricePrediction//result//LibSVM//part-00000"

    val data = MLUtils.loadLibSVMFile(sc, sourceFile)

    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a RandomForest model.
    val categoricalFeaturesInfo_2015 = Map[Int, Int]((0, 6), (1, 3), (2, 80), (3, 48), (4, 20), (5, 3), (6, 5))
    val categoricalFeaturesInfo_201505 = Map[Int, Int]((0, 5), (1, 2), (2, 80), (3, 43), (4, 19), (5, 3), (6, 5))
    val categoricalFeaturesInfo_201511 = Map[Int, Int]((0, 6), (1, 3), (2, 80), (3, 48), (4, 18), (5, 3), (6, 5))

    val featureSubsetStrategy = "auto" // Number of features to consider for splits at each node. Supported: "auto", "all", "sqrt", "log2", "onethird". If "auto" is set, this parameter is set based on numTrees: if numTrees == 1, set to "all"; if numTrees > 1 (forest) set to "onethird".
    val impurity = "variance" // Criterion used for information gain calculation. Supported values: "variance".
    //val maxDepth = 10 // Maximum depth of the tree. E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. (suggested value: 4)
    val maxBins = 128 //maximum number of bins used for splitting features (suggested value: 100)
    //val seed = 12345 // Random seed for bootstrapping and choosing feature subsets.

    val model: RandomForestModel = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo_2015, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins) // Since( "1.2.0" )
    // RandomForestRegressor // @Since( "1.4.0" )

    // Evaluate model on test instances and compute test error
    val labelsAndPredictions = testData.map { point: LabeledPoint =>
      val prediction: Double = model.predict(point.features)
      (point.label, prediction)
    }

    val regressionMetrics = new RegressionMetrics(labelsAndPredictions)
    val rmse = regressionMetrics.rootMeanSquaredError
    rmse

    // Evaluate model on training instances and compute test error
    //    val labelsAndPredictions = trainingData   .map { point: LabeledPoint =>
    //      val prediction: Double = model.predict(point.features)
    //      (point.label, prediction)
    //    }
    //
    //    val regressionMetrics = new RegressionMetrics(labelsAndPredictions)
    //    val rmse = regressionMetrics.rootMeanSquaredError
    //    rmse
  }

}
