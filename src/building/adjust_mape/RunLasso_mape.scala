package building.adjust_mape

import common.Utility
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.evaluation._
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.joda.time._

object RunLasso_mape {

  def main(args: Array[String]): Unit = {

    SetLogger()
    val sc = new SparkContext(new SparkConf().setAppName("RDF").setMaster("local[4]"))
    println("Run Lasso start....")
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
      println("best model's weights : " + model.weights)
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

    println("Run Lasso end....")
  }

  def PrepareData(sc: SparkContext): (RDD[LabeledPoint], RDD[LabeledPoint], RDD[LabeledPoint], RDD[LabeledPoint]) = {
    //----------------------1.匯入轉換資料-------------
    print("開始匯入資料...")

    val sourceFile = "C://Spark//PricePrediction//result//ny//LibSVM_Scaling//part-00000"
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, sourceFile)
    val Array(trainData, cvData, testData, predictData) = data.randomSplit(Array(0.6, 0.1, 0.2, 0.1))
    println("將資料分為 trainData:" + trainData.count() + "   cvData:" + cvData.count() + "   testData:" + testData.count() + "   predictData:" + predictData.count())
    return (trainData, cvData, testData, predictData)
  }


  def PredictData(sc: SparkContext, model: LassoModel, predictData: RDD[LabeledPoint], predictCnt: Int): Unit = {
    println("predictData共計 : " + predictData.count() + "筆,隨機取出" + predictCnt + "筆,進行預測")

    predictData.take(predictCnt).map { labeledPoint: LabeledPoint =>
      val label: Double = labeledPoint.label
      val features = labeledPoint.features
      val predict: Double = model.predict(features)
      val error: Double = math.abs(label.toInt - predict.toInt)
      println(" ==> 預測結果 : " + predict.toInt + "    實際:" + label.toInt + "  誤差:" + error.toInt)
    }
  }

  def trainModel(trainData: RDD[LabeledPoint], numIterations: Int, stepSize: Double, regParam: Double, minBatchFraction: Double): (LassoModel, Double) = {
    val startTime = new DateTime()
    val model: LassoModel = LassoWithSGD.train(trainData, numIterations, stepSize, regParam, minBatchFraction)
    val endTime = new DateTime()
    val duration = new Duration(startTime, endTime)
    (model, duration.getMillis())
  }

  def evaluateModel(model: LassoModel, validationData: RDD[LabeledPoint]): (Double, Double, Double) = {
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

  def parametersTuning(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint]): LassoModel = {

    val numIterationsArray: Array[Int] = Array(10, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400)
    val stepSizeArray: Array[Double] = Array(1, 0.1, 0.01)
    val regParamArray: Array[Double] = Array(0.01, 0.001, 0.0001)
    val minBatchFractionArray: Array[Double] = Array(1, 0.1, 0.01)

    /*
    println("-----評估NumIterations參數使用 (" + numIterationsArray.mkString(" , ") + ")---------")
    println("-----評估StepSize參數使用 (" + stepSizeArray.mkString(" , ") + ")---------")
    println("-----評估RegParam參數使用 (" + regParamArray.mkString(" , ") + ")---------")
    println("-----評估MinBatchFraction參數使用 (" + minBatchFractionArray.mkString(" , ") + ")---------")
    println("-----所有參數交叉評估找出最好的參數組合---------")
    println("")
    */
    val bestModel = evaluateAllParameter(trainData, validationData, numIterationsArray, stepSizeArray, regParamArray, minBatchFractionArray)

    return (bestModel)
  }


  def evaluateAllParameter(trainData: RDD[LabeledPoint], validationData: RDD[LabeledPoint], numIterationsArray: Array[Int], stepSizeArray: Array[Double], regParamArray: Array[Double], minBatchFractionArray: Array[Double]): LassoModel = {
    val evaluationsArray =
      for (numIterations <- numIterationsArray; stepSize <- stepSizeArray; regParam <- regParamArray; minBatchFraction <- minBatchFractionArray) yield {
        val (model, time) = trainModel(trainData, numIterations, stepSize, regParam, minBatchFraction)
        val evaluateResult = evaluateModel(model, validationData)
        val RMSE = evaluateResult._1
        val MAE = evaluateResult._2
        val MAPE = evaluateResult._3
        // println("參數組合：numIterations:" + numIterations + ", stepSize:" + stepSize + ", regParam:" + regParam + ", minBatchFraction:" + minBatchFraction + ", RMSE:" + Utility.decimalRoundOff(RMSE, 2) + ", timeSpent:" + time + ", MAE:" + MAE + ", MAPE:" + MAPE)
        (numIterations, stepSize, regParam, minBatchFraction, MAPE, time)
      }
    val evaluationsArraySortedAsc = (evaluationsArray.sortBy(_._5))
    val BestEval = evaluationsArraySortedAsc(0)
    println("")
    println("使用validationData進行評估")
    println("調校後最佳參數：numIterations:" + BestEval._1 + ", stepSize:" + BestEval._2 + ", regParam:" + BestEval._3 + ", minBatchFraction:" + BestEval._4 + ", MAPE:" + Utility.decimalRoundOff(BestEval._5, 2) + ", timeSpent:" + BestEval._6)

    val (bestModel, time) = trainModel(trainData.union(validationData), BestEval._1, BestEval._2, BestEval._3, BestEval._4)
    return bestModel
  }

  def SetLogger() = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("com").setLevel(Level.OFF)
    System.setProperty("spark.ui.showConsoleProgress", "false")
    Logger.getRootLogger().setLevel(Level.OFF);
  }
}