package building

import java.util.Calendar

import common.Utility
import org.apache.commons.lang3.time.DurationFormatUtils
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

import scala.collection.mutable.ListBuffer

/**
  * Created by bruce on 2016/4/5.
  */
object RegressionTree {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("appName").setMaster("local")
    val sc = new SparkContext(conf)

    val maxDepthArray: Array[Int] = Array(2, 3, 4, 5, 6, 7, 8, 9, 10)
    //val maxDepthArray: Array[Int] = Array(4)
    val list = new ListBuffer[(Int, Double, String)]()

    maxDepthArray.foreach { f: Int =>
      println("treeDepth : " + f)
      val begin: Calendar = Calendar.getInstance()
      val testRMSE = build(sc, f);
      val end: Calendar = Calendar.getInstance()
      val timeDiff = DurationFormatUtils.formatDuration(Math.abs(begin.getTimeInMillis - end.getTimeInMillis), "ss.SS")
      list += Tuple3(f, testRMSE, timeDiff)
    }

    list.foreach { f => println("treeDepth/RMSE/timeDiff : " + f._1 + "/" + f._2 + "/" + f._3) }
  }

  def build(sc: SparkContext, maxDepth: Int): Double = {
    val sourceFile = "C://Spark//PricePrediction//result//ny//LibSVM_Scaling//part-00000"

    // data structure = label index1:value1 index2:value2
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, sourceFile)

    // feature scaling
    val featuresData = data.map(labelpoint => labelpoint.features)
    val stdScaler = new StandardScaler(withMean = true, withStd = true).fit(featuresData)
    val scaledRDD: RDD[LabeledPoint] = data.map(labelpoint => LabeledPoint(labelpoint.label, stdScaler.transform(labelpoint.features)))


    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a DecisionTree model.
    val categoricalFeaturesInfo_2015 = Map[Int, Int]((0, 6), (1, 3), (2, 80), (3, 48), (4, 20), (5, 3), (6, 5))
    val categoricalFeaturesInfo_201505 = Map[Int, Int]((0, 5), (1, 2), (2, 80), (3, 43), (4, 19), (5, 3), (6, 5))
    val categoricalFeaturesInfo_201511 = Map[Int, Int]((0, 6), (1, 3), (2, 80), (3, 48), (4, 18), (5, 3), (6, 5))

    val impurity = "variance"
    val maxBins = 100000 // num of bins used when discretizing continuous features, increasing maxbins allow the algorithm to consider more split candidates and make fine-grained split decision

    val model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo_2015, impurity, maxDepth, maxBins)
    // Evaluate model on test instances and compute test error
    val labelsAndPredictions = testData.map { point: LabeledPoint =>
      val prediction: Double = model.predict(point.features)
      (point.label, prediction)
    }
    //println("Learned regression tree model:\n" + model.toDebugString)
    val testMSE: Double = labelsAndPredictions.map { case (actualValue: Double, predictValue: Double) => math.pow((actualValue - predictValue), 2) }.mean()

    var mape: Double = labelsAndPredictions.map { case (actualValue: Double, predictValue: Double) =>
      // println("actualValue : " + actualValue)
      // println("predictValue : " + predictValue)
      // println("errorRate : " + (actualValue - predictValue) / actualValue)
      math.abs((actualValue - predictValue) / actualValue) * 100
    }.mean()
    mape = math.round(mape)

    Utility.decimalRoundOff(math.sqrt(testMSE), 2)

    val regressionMetrics = new RegressionMetrics(labelsAndPredictions)
    val rmse = regressionMetrics.rootMeanSquaredError
    val r2 = regressionMetrics.r2
    val mae = regressionMetrics.meanAbsoluteError
    val explainVariance = regressionMetrics.explainedVariance

    println("*****************************************************")
    println("mape : " + mape);
    println("mae : " + mae)
    println("rmse : " + rmse)
    println("r2 : " + r2)
    println("explainVariance : " + explainVariance)

    rmse
  }
}
