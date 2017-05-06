package building

import java.io.File
import java.util.Calendar

import common.Utility
import org.apache.commons.io.FileUtils
import org.apache.commons.lang3.time.DurationFormatUtils
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable.{ArrayBuffer, ListBuffer}

/**
  * Created by bruce on 2016/4/5.
  */
object KNN {
  def main(args: Array[String]): Unit = {
    val begin: Calendar = Calendar.getInstance()
    cosineSimilarity()
    val end: Calendar = Calendar.getInstance()
    println("time difference : " + DurationFormatUtils.formatDurationHMS(Math.abs(begin.getTimeInMillis - end.getTimeInMillis)));
  }

  def cosineSimilarity(): Unit = {
    val sourceFile = "C://Spark//PricePrediction//source//listings_201505.csv"
    val resultFolder = "C://Spark//PricePrediction//result//CosineSimilarity"

    FileUtils.deleteDirectory(new File(resultFolder))

    val conf = new SparkConf().setAppName("appName").setMaster("local")
    val sc = new SparkContext(conf)

    // --------------------- check each tuple's # of features
    var lineNum = 1
    val lineRdd = sc.textFile(sourceFile).map { line => line.split(",", -1) }
    lineRdd.foreach { f: Array[String] =>
      println("lineNum : " + lineNum + " , length : " + f.length)
      lineNum = lineNum + 1
      f.foreach { a: String =>
        //print(a + ",")
      }
      //println("")
    }

    // --------------------- gen cosine similarity matrix
    var resultListBuffer = new ListBuffer[Array[String]]()
    lineRdd.collect().drop(1).foreach { baseArray: Array[String] =>
      var baseSimilarityArrayBuffer = new ArrayBuffer[String]()

      lineRdd.collect().drop(1).foreach { compareArray: Array[String] =>
        baseSimilarityArrayBuffer += calCosine(baseArray, compareArray).toString()
      }

      resultListBuffer += baseSimilarityArrayBuffer.toArray
    }

    sc.parallelize(resultListBuffer.toList).map { x => x.mkString(",") }.saveAsTextFile(resultFolder)

  }

  def calCosine(aArray: Array[String], bArray: Array[String]): Double = {
    var sum: Double = 0
    var aSqrtSum: Double = 0
    var bSqrtSum: Double = 0

    for (i <- 0 to (aArray.length - 1)) {
      //println("aArray(i) : " + aArray(i))
      //println("bArray(i) : " + bArray(i))
      var a = 1
      var b = 1
      if (aArray(i) != bArray(i)) {
        b = 0
      }
      sum = sum + (a * b)
      aSqrtSum += math.pow(a, 2)
      bSqrtSum += math.pow(b, 2)
    }
    Utility.decimalRoundOff(sum / (math.sqrt(aSqrtSum) * math.sqrt(bSqrtSum)), 3)
  }

}
