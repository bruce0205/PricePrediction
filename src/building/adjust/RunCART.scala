package building.adjust

import common.Utility
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.joda.time._

object RunCART {

  def main(args: Array[String]): Unit = {

    SetLogger()
    val sc = new SparkContext(new SparkConf().setAppName("RDF").setMaster("local[4]"))
    println("Run CART start....")
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
      println("best model's structure : " + model.toDebugString)
      println()
      println("==========測試階段===============")
      val RMSE = Utility.decimalRoundOff(evaluateModel(model, testData)._1, 2)
      val MAE = Utility.decimalRoundOff(evaluateModel(model, testData)._2, 2)
      println("使用testData,共計" + testData.count() + "筆,測試結果RMSE:" + RMSE + ",MAE:" + MAE)
      println()
      println("==========預測階段===============")
      PredictData(sc, model, predictData, 50)

      //取消暫存在記憶體中
      trainData.unpersist();
      validationData.unpersist();
      testData.unpersist();
      predictData.unpersist();
    }

    println("Run CART end....")
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


  def PredictData(sc: SparkContext, model: DecisionTreeModel, predictData: RDD[LabeledPoint], predictCnt: Int): Unit = {
    println("predictData共計 : " + predictData.count() + "筆,隨機取出" + predictCnt + "筆,進行預測")

    predictData.take(predictCnt).map { labeledPoint: LabeledPoint =>
      val label: Double = labeledPoint.label
      val features = labeledPoint.features
      val predict: Double = model.predict(features)
      val error: Double = math.abs(label.toInt - predict.toInt)
      println(" ==> 預測結果 : " + predict.toInt + "    實際:" + label.toInt + "  誤差:" + error.toInt)
    }
  }

  def trainModel(trainData: RDD[LabeledPoint], impurity: String, maxDepth: Int, maxBins: Int): (DecisionTreeModel, Double) = {
    val startTime = new DateTime()
    val categoricalFeaturesInfo = Map[Int, Int]((0, 5), (1, 2), (2, 199), (3, 188), (4, 21), (5, 3), (6, 5))
    val model: DecisionTreeModel = DecisionTree.trainRegressor(trainData, categoricalFeaturesInfo, impurity, maxDepth, maxBins)
    val endTime = new DateTime()
    val duration = new Duration(startTime, endTime)
    (model, duration.getMillis())
  }

  def evaluateModel(model: DecisionTreeModel, validationData: RDD[LabeledPoint]): (Double, Double) = {
    val scoreAndLabels = validationData.map { data =>
      val predict = model.predict(data.features)
      (predict, data.label)
    }
    val Metrics = new RegressionMetrics(scoreAndLabels)
    val RMSE = Metrics.rootMeanSquaredError
    val MAE = Metrics.meanAbsoluteError
    (RMSE, MAE)
  }

  def parametersTuning(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): DecisionTreeModel = {

    val impurityArray: Array[String] = Array("variance")
    val maxDepthArray: Array[Int] = Array(4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 25, 30)
    val maxBinsArray: Array[Int] = Array(200, 250, 300)

    /*
    println("-----評估Impurity參數使用 (" + impurityArray.mkString(" , ") + ")---------")
    println("-----評估MaxDepth參數使用 (" + maxDepthArray.mkString(" , ") + ")---------")
    println("-----評估MaxBins參數使用 (" + maxBinsArray.mkString(" , ") + ")---------")
    println("-----所有參數交叉評估找出最好的參數組合---------")
    println("")
    */
    val bestModel = evaluateAllParameter(trainData, validationData, impurityArray, maxDepthArray, maxBinsArray)

    return (bestModel)
  }

  def evaluateAllParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint], impurityArray: Array[String], maxDepthArray: Array[Int], maxBinsArray: Array[Int]): DecisionTreeModel = {
    val evaluationsArray =
      for (impurity <- impurityArray; maxDepth <- maxDepthArray; maxBins <- maxBinsArray) yield {
        val (model, time) = trainModel(trainData, impurity, maxDepth, maxBins)
        val RMSE = evaluateModel(model, validationData)._1
        // println("參數組合：impurity:" + impurity + ", maxDepth:" + maxDepth + ", maxBins:" + maxBins + ", RMSE:" + Utility.decimalRoundOff(RMSE, 2) + ", timeSpent:" + time)
        (impurity, maxDepth, maxBins, RMSE, time)
      }
    val evaluationsArraySortedAsc = (evaluationsArray.sortBy(_._4))
    val BestEval = evaluationsArraySortedAsc(0)
    println("")
    println("使用validationData進行評估")
    println("調校後最佳參數：impurity:" + BestEval._1 + ", maxDepth:" + BestEval._2 + ", maxBins:" + BestEval._3 + ", RMSE:" + Utility.decimalRoundOff(BestEval._4, 2) + ", timeSpent:" + BestEval._5)

    val (bestModel, time) = trainModel(trainData.union(validationData), BestEval._1, BestEval._2, BestEval._3)
    return bestModel
  }

  def SetLogger() = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }
}