package preprocessing

import java.io.File

import org.apache.commons.io.FileUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.{PipelineModel, Pipeline}
import org.apache.spark.ml.feature.{MinMaxScalerModel, MinMaxScaler, StringIndexer}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.sql.types.{StringType, StructField, StructType}

/**
  * Created by bruce on 2016/4/5.
  */
object LibSVM {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("appName").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    val sourceFile = "C://Spark//PricePrediction//source//ny//listings_201605.csv";
    val resultFolder = "C://Spark//PricePrediction//result//ny//LibSVM"

    // val filePath = "C://Spark//PricePrediction//source//listings_2015.csv"
    // val resultFolder = "C://Spark//PricePrediction//result//ny//LibSVM"

    build(sc, sqlContext, sourceFile, resultFolder)
  }

  def build(sc: SparkContext, sqlContext: SQLContext, sourceFile: String, resultFolder: String): Unit = {
    FileUtils.deleteDirectory(new File(resultFolder))

    val listings = sc.textFile(sourceFile)
    val schemaString = "host_response_time,host_is_superhost,neighbourhood,zipcode,property_type,room_type,accommodates,bathrooms,bedrooms,beds,bed_type,price,guests_included,extra_people,minimum_nights,number_of_reviews,review_scores_rating,review_scores_accuracy,review_scores_cleanliness,review_scores_checkin,review_scores_communication,review_scores_location,review_scores_value,reviews_per_month"

    val schema = StructType(schemaString.split(",", -1).map(f => StructField(f, StringType, true)))
    val rowRDD = sc.parallelize(listings.collect().drop(1)).map(_.split(",", -1)).map(p => Row(p(0), p(1), p(2), p(3), p(4), p(5), p(6), p(7), p(8), p(9), p(10), p(11), p(12), p(13), p(14), p(15), p(16), p(17), p(18), p(19), p(20), p(21), p(22), p(23)))
    val listingDataFrame: DataFrame = sqlContext.createDataFrame(rowRDD, schema)
    listingDataFrame.registerTempTable("listing")

    val hostResponseTimeIndexer = new StringIndexer()
      .setInputCol("host_response_time")
      .setOutputCol("host_response_time_index")

    val hostIsSuperHostIndexer = new StringIndexer()
      .setInputCol("host_is_superhost")
      .setOutputCol("host_is_superhost_index")

    val neighbourhoodIndexer = new StringIndexer()
      .setInputCol("neighbourhood")
      .setOutputCol("neighbourhood_index")

    val zipcodeIndexer = new StringIndexer()
      .setInputCol("zipcode")
      .setOutputCol("zipcode_index")

    val propertyTypeIndexer = new StringIndexer()
      .setInputCol("property_type")
      .setOutputCol("property_type_index")

    val roomTypeIndexer = new StringIndexer()
      .setInputCol("room_type")
      .setOutputCol("room_type_index")

    val bedTypeIndexer = new StringIndexer()
      .setInputCol("bed_type")
      .setOutputCol("bed_type_index")

    val pipeline = new Pipeline().setStages(Array(hostResponseTimeIndexer, hostIsSuperHostIndexer, neighbourhoodIndexer, zipcodeIndexer, propertyTypeIndexer, roomTypeIndexer, bedTypeIndexer))
    val model: PipelineModel = pipeline.fit(listingDataFrame)

    val finalListingDataFrame: DataFrame = model.transform(listingDataFrame)
    finalListingDataFrame.registerTempTable("listing_final")

    finalListingDataFrame
      .select("price", "host_response_time", "host_response_time_index", "neighbourhood", "neighbourhood_index", "zipcode", "zipcode_index", "property_type", "property_type_index", "room_type", "room_type_index", "bed_type", "bed_type_index")
    //.show(20)

    val result: DataFrame = sqlContext.sql("select * from listing_final where review_scores_rating !='' ")
    println("result.count() : " + result.count());

    result.map { r: Row => r.getAs[String]("price")
      .concat(" 1:" + r.getAs[Double]("host_response_time_index"))
      .concat(" 2:" + r.getAs[Double]("host_is_superhost_index"))
      .concat(" 3:" + r.getAs[Double]("neighbourhood_index"))
      .concat(" 4:" + r.getAs[Double]("zipcode_index"))
      .concat(" 5:" + r.getAs[Double]("property_type_index"))
      .concat(" 6:" + r.getAs[Double]("room_type_index"))
      .concat(" 7:" + r.getAs[Double]("bed_type_index"))
      .concat(if (r.getAs[String]("accommodates") == "") "" else " 8:" + r.getAs[String]("accommodates"))
      .concat(if (r.getAs[String]("bathrooms") == "") "" else " 9:" + r.getAs[String]("bathrooms"))
      .concat(if (r.getAs[String]("bedrooms") == "") "" else " 10:" + r.getAs[String]("bedrooms"))
      .concat(if (r.getAs[String]("beds") == "") "" else " 11:" + r.getAs[String]("beds"))
      .concat(if (r.getAs[String]("guests_included") == "") "" else " 12:" + r.getAs[String]("guests_included"))
      .concat(if (r.getAs[String]("extra_people") == "") "" else " 13:" + r.getAs[String]("extra_people"))
      .concat(if (r.getAs[String]("minimum_nights") == "") "" else " 14:" + r.getAs[String]("minimum_nights"))
      .concat(if (r.getAs[String]("number_of_reviews") == "") "" else " 15:" + r.getAs[String]("number_of_reviews"))
      .concat(if (r.getAs[String]("review_scores_rating") == "") "" else " 16:" + r.getAs[String]("review_scores_rating"))
      .concat(if (r.getAs[String]("review_scores_accuracy") == "") "" else " 17:" + r.getAs[String]("review_scores_accuracy"))
      .concat(if (r.getAs[String]("review_scores_cleanliness") == "") "" else " 18:" + r.getAs[String]("review_scores_cleanliness"))
      .concat(if (r.getAs[String]("review_scores_checkin") == "") "" else " 19:" + r.getAs[String]("review_scores_checkin"))
      .concat(if (r.getAs[String]("review_scores_communication") == "") "" else " 20:" + r.getAs[String]("review_scores_communication"))
      .concat(if (r.getAs[String]("review_scores_location") == "") "" else " 21:" + r.getAs[String]("review_scores_location"))
      .concat(if (r.getAs[String]("review_scores_value") == "") "" else " 22:" + r.getAs[String]("review_scores_value"))
      .concat(if (r.getAs[String]("reviews_per_month") == "") "" else " 23:" + r.getAs[String]("reviews_per_month"))
    }.repartition(1).saveAsTextFile(resultFolder)

    buildDerivative(sqlContext)

  }

  def buildDerivative(sqlContext: SQLContext): Unit = {
    val resultFolder = "C://Spark//PricePrediction//result//ny//LibSVM_Derivative"
    FileUtils.deleteDirectory(new File(resultFolder))



    // step 1 : neighbourhood : price average
    var neighborPriceCntMap = scala.collection.mutable.Map[String, Int]()
    var neighborPriceMap = scala.collection.mutable.Map[String, Int]()
    val neighborPriceData: DataFrame = sqlContext.sql("select neighbourhood, price from listing_final where neighbourhood != '' and review_scores_rating !='' order by neighbourhood ")

    neighborPriceData.map { r: Row =>
      (r.getAs[String]("neighbourhood"), 1)
    }.reduceByKey(_ + _).collect().foreach { f: ((String, Int)) =>
      neighborPriceCntMap += f._1 -> f._2
    }

    neighborPriceData.map { r: Row =>
      (r.getAs[String]("neighbourhood"), r.getAs[String]("price").toInt)
    }.reduceByKey(_ + _).collect().foreach { f: ((String, Int)) =>
      neighborPriceMap += f._1 -> (f._2 / neighborPriceCntMap(f._1))
    }



    // step 2 : neighbourhood + roomType : price average
    var neighborRoomPriceCntMap = scala.collection.mutable.Map[String, Int]()
    var neighborRoomPriceMap = scala.collection.mutable.Map[String, Int]()
    val neighborRoomPriceData: DataFrame = sqlContext.sql("select neighbourhood, room_type, price from listing_final where neighbourhood != '' and room_type !='' and review_scores_rating !='' order by neighbourhood ")

    neighborRoomPriceData.map { r: Row =>
      (r.getAs[String]("neighbourhood") + "_" + r.getAs[String]("room_type"), 1)
    }.reduceByKey(_ + _).collect().foreach { f: ((String, Int)) =>
      neighborRoomPriceCntMap += f._1 -> f._2
    }

    neighborRoomPriceData.map { r: Row =>
      (r.getAs[String]("neighbourhood") + "_" + r.getAs[String]("room_type"), r.getAs[String]("price").toInt)
    }.reduceByKey(_ + _).collect().foreach { f: ((String, Int)) =>
      neighborRoomPriceMap += f._1 -> (f._2 / neighborRoomPriceCntMap(f._1))
    }



    // step 3 : neighbour + property_type + room_type : average price
    var neighborPropertyRoomPriceCntMap = scala.collection.mutable.Map[String, Int]()
    var neighborPropertyRoomPriceMap = scala.collection.mutable.Map[String, Int]()
    val neighborPropertyRoomPriceData: DataFrame = sqlContext.sql("select neighbourhood, property_type, room_type, price from listing_final where neighbourhood != '' and property_type !='' and room_type !='' and review_scores_rating !='' order by neighbourhood ")

    neighborPropertyRoomPriceData.map { r: Row =>
      (r.getAs[String]("neighbourhood") + "_" + r.getAs[String]("property_type") + "_" + r.getAs[String]("room_type"), 1)
    }.reduceByKey(_ + _).collect().foreach { f: ((String, Int)) =>
      neighborPropertyRoomPriceCntMap += f._1 -> f._2
    }


    neighborPropertyRoomPriceData.map { r: Row =>
      (r.getAs[String]("neighbourhood") + "_" + r.getAs[String]("property_type") + "_" + r.getAs[String]("room_type"), r.getAs[String]("price").toInt)
    }.reduceByKey(_ + _).collect().foreach { f: ((String, Int)) =>
      neighborPropertyRoomPriceMap += f._1 -> (f._2 / neighborPropertyRoomPriceCntMap(f._1))
    }

    // step 4 : neighbour + property + room_type + accommodate : average price
    var neighborPropertyRoomAccommodatePriceCntMap = scala.collection.mutable.Map[String, Int]()
    var neighborPropertyRoomAccommodatePriceMap = scala.collection.mutable.Map[String, Int]()
    val neighborPropertyRoomAccommodatePriceData: DataFrame = sqlContext.sql("select neighbourhood, property_type, room_type, accommodates, price from listing_final where neighbourhood != '' and property_type !='' and room_type !='' and accommodates!= '' and review_scores_rating !='' order by neighbourhood ")

    neighborPropertyRoomAccommodatePriceData.map { r: Row =>
      (r.getAs[String]("neighbourhood") + "_" + r.getAs[String]("property_type") + "_" + r.getAs[String]("room_type") + "_" + r.getAs[String]("accommodates"), 1)
    }.reduceByKey(_ + _).collect().foreach { f: ((String, Int)) =>
      neighborPropertyRoomAccommodatePriceCntMap += f._1 -> f._2
    }


    println("show neighborPropertyRoomAccommodatePriceCntMap")
    neighborPropertyRoomAccommodatePriceCntMap.foreach { case (key, value) => println(key + "-->" + value) }
    println("neighborPropertyRoomAccommodatePriceCntMap.size : " + neighborPropertyRoomAccommodatePriceCntMap.size)


    neighborPropertyRoomAccommodatePriceData.map { r: Row =>
      (r.getAs[String]("neighbourhood") + "_" + r.getAs[String]("property_type") + "_" + r.getAs[String]("room_type") + "_" + r.getAs[String]("accommodates"), r.getAs[String]("price").toInt)
    }.reduceByKey(_ + _).collect().foreach { f: ((String, Int)) =>
      println(f._1)
      println(f._2)
      println(neighborPropertyRoomAccommodatePriceCntMap(f._1))
      println(f._2 / neighborPropertyRoomAccommodatePriceCntMap(f._1))
      neighborPropertyRoomAccommodatePriceMap += f._1 -> (f._2 / neighborPropertyRoomAccommodatePriceCntMap(f._1))
    }

    // step 5 : gen libsvm
    val result: DataFrame = sqlContext.sql("select * from listing_final where review_scores_rating !=''")
    result.map { r: Row =>
      r.getAs[String]("price")
        .concat(" 1:" + r.getAs[Double]("host_response_time_index"))
        .concat(" 2:" + r.getAs[Double]("host_is_superhost_index"))
        .concat(" 3:" + r.getAs[Double]("neighbourhood_index"))
        .concat(" 4:" + r.getAs[Double]("zipcode_index"))
        .concat(" 5:" + r.getAs[Double]("property_type_index"))
        .concat(" 6:" + r.getAs[Double]("room_type_index"))
        .concat(" 7:" + r.getAs[Double]("bed_type_index"))
        .concat(if (r.getAs[String]("accommodates") == "") "" else " 8:" + r.getAs[String]("accommodates"))
        .concat(if (r.getAs[String]("bathrooms") == "") "" else " 9:" + r.getAs[String]("bathrooms"))
        .concat(if (r.getAs[String]("bedrooms") == "") "" else " 10:" + r.getAs[String]("bedrooms"))
        .concat(if (r.getAs[String]("beds") == "") "" else " 11:" + r.getAs[String]("beds"))
        .concat(if (r.getAs[String]("guests_included") == "") "" else " 12:" + r.getAs[String]("guests_included"))
        .concat(if (r.getAs[String]("extra_people") == "") "" else " 13:" + r.getAs[String]("extra_people"))
        .concat(if (r.getAs[String]("minimum_nights") == "") "" else " 14:" + r.getAs[String]("minimum_nights"))
        .concat(if (r.getAs[String]("number_of_reviews") == "") "" else " 15:" + r.getAs[String]("number_of_reviews"))
        .concat(if (r.getAs[String]("review_scores_rating") == "") "" else " 16:" + r.getAs[String]("review_scores_rating"))
        .concat(if (r.getAs[String]("review_scores_accuracy") == "") "" else " 17:" + r.getAs[String]("review_scores_accuracy"))
        .concat(if (r.getAs[String]("review_scores_cleanliness") == "") "" else " 18:" + r.getAs[String]("review_scores_cleanliness"))
        .concat(if (r.getAs[String]("review_scores_checkin") == "") "" else " 19:" + r.getAs[String]("review_scores_checkin"))
        .concat(if (r.getAs[String]("review_scores_communication") == "") "" else " 20:" + r.getAs[String]("review_scores_communication"))
        .concat(if (r.getAs[String]("review_scores_location") == "") "" else " 21:" + r.getAs[String]("review_scores_location"))
        .concat(if (r.getAs[String]("review_scores_value") == "") "" else " 22:" + r.getAs[String]("review_scores_value"))
        .concat(if (r.getAs[String]("reviews_per_month") == "") "" else " 23:" + r.getAs[String]("reviews_per_month"))
        .concat(if (r.getAs[String]("neighbourhood") == "") "" else " 24:" + neighborPriceMap(r.getAs[String]("neighbourhood")))
        .concat(if (r.getAs[String]("neighbourhood") == "" || r.getAs[String]("room_type") == "") "" else " 25:" + neighborRoomPriceMap(r.getAs[String]("neighbourhood") + "_" + r.getAs[String]("room_type")))
        .concat(if (r.getAs[String]("neighbourhood") == "" || r.getAs[String]("property_type") == "" || r.getAs[String]("room_type") == "") "" else " 26:" + neighborPropertyRoomPriceMap(r.getAs[String]("neighbourhood") + "_" + r.getAs[String]("property_type") + "_" + r.getAs[String]("room_type")))
        .concat(if (r.getAs[String]("neighbourhood") == "" || r.getAs[String]("property_type") == "" || r.getAs[String]("room_type") == "" || r.getAs[String]("accommodates") == "") "" else " 27:" + neighborPropertyRoomAccommodatePriceMap(r.getAs[String]("neighbourhood") + "_" + r.getAs[String]("property_type") + "_" + r.getAs[String]("room_type") + "_" + r.getAs[String]("accommodates")))
    }.repartition(1).saveAsTextFile(resultFolder)


  }

}
