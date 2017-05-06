package building

import java.util
import java.util.Calendar

import common.Utility
import org.apache.commons.lang3.time.DurationFormatUtils
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.loss.AbsoluteError
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.tree.impurity.Variance

import scala.collection.mutable.ListBuffer

/**
  * Created by bruce on 2016/4/6.
  */
object GradientBoostedTree {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("appName").setMaster("local")
    val sc = new SparkContext(conf)

    val numIterationsArray: Array[Int] = Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 60, 70, 80, 90, 100, 150)
    //val numIterationsArray: Array[Int] = Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    val maxDepth: Int = 4
    val list = new ListBuffer[(Int, Double, String)]()

    numIterationsArray.foreach { f: Int =>
      println("numIterations : " + f)
      val begin: Calendar = Calendar.getInstance()
      val testRMSE = build(sc, f, maxDepth);
      val end: Calendar = Calendar.getInstance()
      val timeDiff = DurationFormatUtils.formatDuration(Math.abs(begin.getTimeInMillis - end.getTimeInMillis), "ss.SS")
      list += Tuple3(f, testRMSE, timeDiff)
    }

    list.foreach { f => println("numIterations/RMSE/timeDiff : " + f._1 + "/" + f._2 + "/" + f._3) }

  }

  def build(sc: SparkContext, numIterations: Int, maxDepth: Int): Double = {
    val sourceFile = "C://Spark//PricePrediction//result//LibSVM_Derivative//part-00000"

    // Load and parse the data file.
    val data = MLUtils.loadLibSVMFile(sc, sourceFile)

    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a GradientBoostedTrees model.
    // The defaultParams for Regression use SquaredError by default.
    val boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.setNumIterations(numIterations) // Note: Use more iterations in practice.
    boostingStrategy.getTreeStrategy().setMaxDepth(maxDepth)
    boostingStrategy.getTreeStrategy().setMaxBins(200)
    boostingStrategy.getTreeStrategy().setImpurity(Variance)
    boostingStrategy.setLoss(AbsoluteError) // more robust than SquaredError
    boostingStrategy.setLearningRate(0.03)


    //    val categoricalFeaturesInfo_2015 = Map[Int, Int]((0, 6), (1, 3), (2, 80), (3, 48), (4, 20), (5, 3), (6, 5))
    //    val categoricalFeaturesInfo_201505 = Map[Int, Int]((0, 5), (1, 2), (2, 80), (3, 43), (4, 19), (5, 3), (6, 5))
    //    val categoricalFeaturesInfo_201511 = Map[Int, Int]((0, 6), (1, 3), (2, 80), (3, 48), (4, 18), (5, 3), (6, 5))
    //    boostingStrategy.treeStrategy.categoricalFeaturesInfo = categoricalFeaturesInfo_2015

    val categoricalFeaturesInfo = new util.HashMap[Integer, Integer]();
    categoricalFeaturesInfo.put(0, 6)
    categoricalFeaturesInfo.put(1, 3)
    categoricalFeaturesInfo.put(2, 80)
    categoricalFeaturesInfo.put(3, 48)
    categoricalFeaturesInfo.put(4, 20)
    categoricalFeaturesInfo.put(5, 3)
    categoricalFeaturesInfo.put(6, 5)
    boostingStrategy.getTreeStrategy.setCategoricalFeaturesInfo(categoricalFeaturesInfo)

    val model:GradientBoostedTreesModel = GradientBoostedTrees.train(trainingData, boostingStrategy)

    // Evaluate model on training instance and compute test error
    val labelsAndPredictions = testData.map { point: LabeledPoint =>
      val prediction: Double = model.predict(point.features)
      (point.label, prediction)
    }

    //    val testMSE: Double = labelsAndPredictions.map { case (actualValue: Double, predictValue: Double) => math.pow((actualValue - predictValue), 2) }.mean()
    //    Utility.decimalRoundOff(math.sqrt(testMSE), 2)

    val regressionMetrics = new RegressionMetrics(labelsAndPredictions)
    val rmse = regressionMetrics.rootMeanSquaredError
    rmse
  }
}
