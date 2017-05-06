package common

/**
  * Created by bruce on 2016/4/5.
  */
object Utility {
  def decimalRoundOff(x: Double, digit: Int): Double = {
    val base: Double = math.pow(10, digit)
    math.round(x * base) / base
  }

}
