package preprocessing

import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, SQLContext}

import scala.collection.mutable.ListBuffer

/**
  * Created by bruce on 2016/4/5.
  */
object DistinctFeature {
  def main(args: Array[String]): Unit = {
    val filePath = "C://Spark//PricePrediction//source//ny//listings_201605.csv";
    // _2015()
    // _201511()
    // _201505()
    doAction(filePath)
  }

  def doAction(filePath: String): Unit = {
    val conf = new SparkConf().setAppName("appName").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val listings = sc.textFile(filePath)
    val schemaString = "host_response_time,host_is_superhost,neighbourhood,zipcode,property_type,room_type,accommodates,bathrooms,bedrooms,beds,bed_type,square_feet,price,weekly_price,monthly_price,guests_included,extra_people,minimum_nights,number_of_reviews,reviews_per_month"

    val schema = StructType(schemaString.split(",", -1).map(f => StructField(f, StringType, true)))
    val rowRDD = sc.parallelize(listings.collect().drop(1)).map(_.split(",", -1)).map(p => Row(p(0), p(1), p(2), p(3), p(4), p(5), p(6), p(7), p(8), p(9), p(10), p(11), p(12), p(13), p(14), p(15), p(16), p(17), p(18), p(19)))

    val listingDataFrame = sqlContext.createDataFrame(rowRDD, schema)
    listingDataFrame.registerTempTable("listing")

    sqlContext.sql("select distinct property_type from listing")
      .map { r: Row => r.get(0) }
      .collect().foreach(println)

    val list = new ListBuffer[String]()

    sqlContext.sql("select count(*) from (select distinct host_response_time from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("host_response_time : " + f) }
    sqlContext.sql("select count(*) from (select distinct host_is_superhost from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("host_is_superhost : " + f) }
    sqlContext.sql("select count(*) from (select distinct neighbourhood from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("neighbourhood : " + f) }
    sqlContext.sql("select count(*) from (select distinct zipcode from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("zipcode : " + f) }
    sqlContext.sql("select count(*) from (select distinct property_type from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("property_type : " + f) }
    sqlContext.sql("select count(*) from (select distinct room_type from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("room_type : " + f) }
    sqlContext.sql("select count(*) from (select distinct bed_type from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("bed_type : " + f) }

    list.foreach(println)
  }

  def _2015(): Unit = {
    val conf = new SparkConf().setAppName("appName").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val listings = sc.textFile("C:\\Spark\\PricePrediction\\source\\listings_2015.csv")
    val schemaString = "host_response_time,host_is_superhost,neighbourhood,zipcode,property_type,room_type,accommodates,bathrooms,bedrooms,beds,bed_type,square_feet,price,weekly_price,monthly_price,guests_included,extra_people,minimum_nights,number_of_reviews,reviews_per_month"

    val schema = StructType(schemaString.split(",", -1).map(f => StructField(f, StringType, true)))
    val rowRDD = sc.parallelize(listings.collect().drop(1)).map(_.split(",", -1)).map(p => Row(p(0), p(1), p(2), p(3), p(4), p(5), p(6), p(7), p(8), p(9), p(10), p(11), p(12), p(13), p(14), p(15), p(16), p(17), p(18), p(19)))

    val listingDataFrame = sqlContext.createDataFrame(rowRDD, schema)
    listingDataFrame.registerTempTable("listing")

    sqlContext.sql("select distinct property_type from listing")
      .map { r: Row => r.get(0) }
      .collect().foreach(println)

    val list = new ListBuffer[String]()

    sqlContext.sql("select count(*) from (select distinct host_response_time from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("host_response_time : " + f) }
    sqlContext.sql("select count(*) from (select distinct host_is_superhost from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("host_is_superhost : " + f) }
    sqlContext.sql("select count(*) from (select distinct neighbourhood from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("neighbourhood : " + f) }
    sqlContext.sql("select count(*) from (select distinct zipcode from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("zipcode : " + f) }
    sqlContext.sql("select count(*) from (select distinct property_type from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("property_type : " + f) }
    sqlContext.sql("select count(*) from (select distinct room_type from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("room_type : " + f) }
    sqlContext.sql("select count(*) from (select distinct bed_type from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("bed_type : " + f) }

    list.foreach(println)
  }


  def _201511(): Unit = {
    val conf = new SparkConf().setAppName("appName").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val listings = sc.textFile("C:\\Spark\\PricePrediction\\source\\listings_201511.csv")
    val schemaString = "host_response_time,host_is_superhost,neighbourhood,zipcode,property_type,room_type,accommodates,bathrooms,bedrooms,beds,bed_type,square_feet,price,weekly_price,monthly_price,guests_included,extra_people,minimum_nights,number_of_reviews,reviews_per_month"

    val schema = StructType(schemaString.split(",", -1).map(f => StructField(f, StringType, true)))
    val rowRDD = sc.parallelize(listings.collect().drop(1)).map(_.split(",", -1)).map(p => Row(p(0), p(1), p(2), p(3), p(4), p(5), p(6), p(7), p(8), p(9), p(10), p(11), p(12), p(13), p(14), p(15), p(16), p(17), p(18), p(19)))

    val listingDataFrame = sqlContext.createDataFrame(rowRDD, schema)
    listingDataFrame.registerTempTable("listing")

    sqlContext.sql("select distinct property_type from listing")
      .map { r: Row => r.get(0) }
      .collect().foreach(println)

    val list = new ListBuffer[String]()

    sqlContext.sql("select count(*) from (select distinct host_response_time from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("host_response_time : " + f) }
    sqlContext.sql("select count(*) from (select distinct host_is_superhost from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("host_is_superhost : " + f) }
    sqlContext.sql("select count(*) from (select distinct neighbourhood from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("neighbourhood : " + f) }
    sqlContext.sql("select count(*) from (select distinct zipcode from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("zipcode : " + f) }
    sqlContext.sql("select count(*) from (select distinct property_type from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("property_type : " + f) }
    sqlContext.sql("select count(*) from (select distinct room_type from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("room_type : " + f) }
    sqlContext.sql("select count(*) from (select distinct bed_type from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("bed_type : " + f) }

    list.foreach(println)
  }

  def _201505(): Unit = {
    val conf = new SparkConf().setAppName("appName").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val listings = sc.textFile("C:\\Spark\\PricePrediction\\source\\listings_201505.csv")
    val schemaString = "host_response_time,host_is_superhost,neighbourhood,zipcode,property_type,room_type,accommodates,bathrooms,bedrooms,beds,bed_type,square_feet,price,weekly_price,monthly_price,guests_included,extra_people,minimum_nights,number_of_reviews,reviews_per_month"

    val schema = StructType(schemaString.split(",", -1).map(f => StructField(f, StringType, true)))
    val rowRDD = sc.parallelize(listings.collect().drop(1)).map(_.split(",", -1)).map(p => Row(p(0), p(1), p(2), p(3), p(4), p(5), p(6), p(7), p(8), p(9), p(10), p(11), p(12), p(13), p(14), p(15), p(16), p(17), p(18), p(19)))

    val listingDataFrame = sqlContext.createDataFrame(rowRDD, schema)
    listingDataFrame.registerTempTable("listing")

    val list = new ListBuffer[String]()

    sqlContext.sql("select count(*) from (select distinct host_response_time from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("host_response_time : " + f) }
    sqlContext.sql("select count(*) from (select distinct host_is_superhost from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("host_is_superhost : " + f) }
    sqlContext.sql("select count(*) from (select distinct neighbourhood from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("neighbourhood : " + f) }
    sqlContext.sql("select count(*) from (select distinct zipcode from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("zipcode : " + f) }
    sqlContext.sql("select count(*) from (select distinct property_type from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("property_type : " + f) }
    sqlContext.sql("select count(*) from (select distinct room_type from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("room_type : " + f) }
    sqlContext.sql("select count(*) from (select distinct bed_type from listing) a")
      .map { r: Row => r.get(0) }
      .collect().foreach { f => list += ("bed_type : " + f) }

    list.foreach(println)
  }

}
