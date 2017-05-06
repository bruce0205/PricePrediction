package building.adjust_mape

import java.util

import common.Utility
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.GradientBoostedTrees
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.impurity.Variance
import org.apache.spark.mllib.tree.loss.{AbsoluteError, SquaredError}
import org.apache.spark.mllib.tree.model.GradientBoostedTreesModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.joda.time._

object RunGBT_mape {

  def main(args: Array[String]): Unit = {

    SetLogger()
    val sc = new SparkContext(new SparkConf().setAppName("RDF").setMaster("local[4]"))
    println("Run GBT start....")
    val times = 10
    for (a <- 1 to times) {
      println("")
      println("Iteration : " + a);

      println("==========資料準備階段===============")
      val (trainData, validationData, testData, predictData) = PrepareData(sc)
      trainData.persist();
      validationData.persist();
      testData.persist();
      predictData.persist();

      println()
      println("==========訓練評估階段===============")
      val model = parametersTuning(trainData, validationData)
      // println("best model's structure : " + model.toDebugString)
      println()
      println("==========測試階段===============")
      val result = evaluateModel(model, testData)
      val RMSE = Utility.decimalRoundOff(result._1, 2)
      val MAE = Utility.decimalRoundOff(result._2, 2)
      val MAPE = result._3
      println("使用testData,共計" + testData.count() + "筆,測試結果RMSE:" + RMSE + ",MAE:" + MAE + ",MAPE:" + MAPE + "%")
      println()
      println("==========預測階段===============")
      PredictData(sc, model, predictData, 50)

      //取消暫存在記憶體中
      trainData.unpersist();
      validationData.unpersist();
      testData.unpersist();
      predictData.unpersist();
    }

    println("Run GBT end....")
  }

  def PrepareData(sc: SparkContext): (RDD[LabeledPoint], RDD[LabeledPoint], RDD[LabeledPoint], RDD[LabeledPoint]) = {
    //----------------------1.匯入轉換資料-------------
    print("開始匯入資料...")

    val sourceFile = "C://Spark//PricePrediction//result//ny//LibSVM_Scaling//part-00000"
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, sourceFile)
    val Array(trainData, cvData, testData, predictData) = data.randomSplit(Array(0.6, 0.2, 0.1, 0.1))
    println("將資料分為 trainData:" + trainData.count() + "   cvData:" + cvData.count() + "   testData:" + testData.count() + "   predictData:" + predictData.count())
    return (trainData, cvData, testData, predictData)
  }


  def PredictData(sc: SparkContext, model: GradientBoostedTreesModel, predictData: RDD[LabeledPoint], predictCnt: Int): Unit = {
    println("predictData共計 : " + predictData.count() + "筆,隨機取出" + predictCnt + "筆,進行預測")

    predictData.take(predictCnt).map { labeledPoint: LabeledPoint =>
      val label: Double = labeledPoint.label
      val features = labeledPoint.features
      val predict: Double = model.predict(features)
      val error: Double = math.abs(label.toInt - predict.toInt)
      println(" ==> 預測結果 : " + predict.toInt + "    實際:" + label.toInt + "  誤差:" + error.toInt)
    }
  }

  def trainModel(trainData: RDD[LabeledPoint], numIterations: Int, loss: String, impurity: String, maxDepth: Int, maxBins: Int, learningRate: Double): (GradientBoostedTreesModel, Double) = {
    val startTime = new DateTime()

    val boostingStrategy = BoostingStrategy.defaultParams("Regression")
    boostingStrategy.setNumIterations(numIterations) // Note: Use more iterations in practice.
    boostingStrategy.getTreeStrategy().setMaxDepth(maxDepth)
    boostingStrategy.getTreeStrategy().setMaxBins(maxBins)
    boostingStrategy.setLearningRate(learningRate)

    if ("Variance" == impurity) {
      boostingStrategy.getTreeStrategy().setImpurity(Variance)
    }
    if ("AbsoluteError" == loss) {
      boostingStrategy.setLoss(AbsoluteError) // more robust than SquaredError
    }
    if ("SquaredError" == loss) {
      boostingStrategy.setLoss(SquaredError) // more robust than SquaredError
    }

    val categoricalFeaturesInfo = new util.HashMap[Integer, Integer]();
    categoricalFeaturesInfo.put(0, 5)
    categoricalFeaturesInfo.put(1, 2)
    categoricalFeaturesInfo.put(2, 199)
    categoricalFeaturesInfo.put(3, 188)
    categoricalFeaturesInfo.put(4, 21)
    categoricalFeaturesInfo.put(5, 3)
    categoricalFeaturesInfo.put(6, 5)
    boostingStrategy.getTreeStrategy.setCategoricalFeaturesInfo(categoricalFeaturesInfo)

    val model: GradientBoostedTreesModel = GradientBoostedTrees.train(trainData, boostingStrategy)
    val endTime = new DateTime()
    val duration = new Duration(startTime, endTime)
    (model, duration.getMillis())
  }

  def evaluateModel(model: GradientBoostedTreesModel, validationData: RDD[LabeledPoint]): (Double, Double, Double) = {
    val scoreAndLabels = validationData.map { data =>
      val predict = model.predict(data.features)
      (predict, data.label)
    }
    val Metrics = new RegressionMetrics(scoreAndLabels)
    val RMSE = Metrics.rootMeanSquaredError
    val MAE = Metrics.meanAbsoluteError

    var MAPE: Double = scoreAndLabels.map { case (predictValue: Double, actualValue: Double) =>
      math.abs((actualValue - predictValue) / actualValue) * 100
    }.mean()
    MAPE = math.round(MAPE)

    (RMSE, MAE, MAPE)
  }

  def parametersTuning(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): GradientBoostedTreesModel = {

    // numIterations: Int, loss: String, impurity: String, maxDepth: Int, maxBins: Int, learningRate: Double

    // val numIterationsArray: Array[Int] = Array(80, 100, 120, 150, 200, 250)
    val numIterationsArray: Array[Int] = Array(4, 5, 6, 7, 8, 9, 10, 15, 20)
    val lossArray: Array[String] = Array("AbsoluteError", "SquaredError") // AbsoluteError, SquaredError
    val impurityArray: Array[String] = Array("Variance")
    val maxDepthArray: Array[Int] = Array(3, 4, 5, 6, 7)
    val maxBinsArray: Array[Int] = Array(200, 250)
    val learningRateArray = Array[Double](1, 0.1, 0.01)

    /*
    println("-----評估NumIterations參數使用 (" + numIterationsArray.mkString(" , ") + ")---------")
    println("-----評估Loss參數使用 (" + lossArray.mkString(" , ") + ")---------")
    println("-----評估Impurity參數使用 (" + impurityArray.mkString(" , ") + ")---------")
    println("-----評估MaxDepth參數使用 (" + maxDepthArray.mkString(" , ") + ")---------")
    println("-----評估MaxBins參數使用 (" + maxBinsArray.mkString(" , ") + ")---------")
    println("-----評估LearningRate參數使用 (" + learningRateArray.mkString(" , ") + ")---------")
    println("-----所有參數交叉評估找出最好的參數組合---------")
    println("")
    */
    val bestModel = evaluateAllParameter(trainData, validationData, numIterationsArray, lossArray, impurityArray, maxDepthArray, maxBinsArray, learningRateArray)

    return (bestModel)
  }

  def evaluateAllParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint], numIterationsArray: Array[Int], lossArray: Array[String], impurityArray: Array[String], maxDepthArray: Array[Int], maxBinsArray: Array[Int], learningRateArray: Array[Double]): GradientBoostedTreesModel = {
    val evaluationsArray =
      for (numIterations <- numIterationsArray; loss <- lossArray; impurity <- impurityArray; maxDepth <- maxDepthArray; maxBins <- maxBinsArray; learningRate <- learningRateArray) yield {
        val (model, time) = trainModel(trainData, numIterations, loss, impurity, maxDepth, maxBins, learningRate)
        val evaluateResult = evaluateModel(model, validationData)
        val RMSE = evaluateResult._1
        val MAE = evaluateResult._2
        val MAPE = evaluateResult._3
        println("參數組合：numIterations:" + numIterations + ", loss:" + loss + ", impurity:" + impurity + ", maxDepth:" + maxDepth + ", maxBins:" + maxBins + ", learningRate:" + learningRate + ", RMSE:" + Utility.decimalRoundOff(RMSE, 2) + ", timeSpent:" + time + ", MAE:" + MAE + ", MAPE:" + MAPE)
        (numIterations, loss, impurity, maxDepth, maxBins, learningRate, MAPE, time)
      }
    val evaluationsArraySortedAsc = (evaluationsArray.sortBy(_._7))
    val BestEval = evaluationsArraySortedAsc(0)
    println("")
    println("使用validationData進行評估")
    println("調校後最佳參數：numIterations:" + BestEval._1 + ", loss:" + BestEval._2 + ", impurity:" + BestEval._3 + ", maxDepth:" + BestEval._4 + ", maxBins:" + BestEval._5 + ", learningRate:" + BestEval._6 + ", MAPE:" + Utility.decimalRoundOff(BestEval._7, 2) + ", timeSpent:" + BestEval._8)

    val (bestModel, time) = trainModel(trainData.union(validationData), BestEval._1, BestEval._2, BestEval._3, BestEval._4, BestEval._5, BestEval._6)
    return bestModel
  }

  def SetLogger() = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }
}