package test

import java.util.Calendar

import org.apache.commons.lang3.time.DurationFormatUtils
import org.joda.time.{Duration, DateTime}

/**
  * Created by bruce on 2016/4/25.
  */
object MainApp {
  def main(args: Array[String]): Unit = {
    val startTime = new DateTime()
    val endTime = new DateTime()
    val duration = new Duration(startTime, endTime)
    println(duration.getMillis)
    println(duration.getStandardSeconds)

    val begin: Calendar = Calendar.getInstance()
    val end: Calendar = Calendar.getInstance()
    val timeDiff = DurationFormatUtils.formatDuration(Math.abs(begin.getTimeInMillis - end.getTimeInMillis), "ss.SS")
    println(timeDiff)

    println(math.round(0.345))

  }
}
