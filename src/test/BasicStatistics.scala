package test

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by bruce on 2016/5/19.
  */
object BasicStatistics {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("appName").setMaster("local")
    val sc = new SparkContext(conf)
    val observations: RDD[Vector] = sc.textFile("C:\\Spark\\spark-1.6.1-bin-hadoop2.6\\data\\mllib\\sample_lda_data.txt").map(s => Vectors.dense(s.split(" ").map(_.toDouble)))


    // Compute column summary statistics.
    val summary: MultivariateStatisticalSummary = Statistics.colStats(observations)
    println(summary.mean) // a dense vector containing the mean value for each column
    println(summary.variance) // column-wise variance
    println(summary.numNonzeros) // number of nonzeros in each column
    println(summary.max)
    println(summary.min)
    println(summary.count)
  }
}
