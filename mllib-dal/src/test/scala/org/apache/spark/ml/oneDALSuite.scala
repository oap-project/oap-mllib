package org.apache.spark.ml

import com.intel.oap.mllib.OneDAL
import org.apache.spark.internal.Logging
import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.sql.Row

class oneDALSuite extends FunctionsSuite with Logging {

  import testImplicits._

  test("test sparse vector to CSRNumericTable") {
    val data = Seq(
      Vectors.sparse(3, Seq((0, 1.0), (1, 2.0), (2, 3.0))),
      Vectors.sparse(3, Seq((0, 10.0), (1, 20.0), (2, 30.0))),
      Vectors.sparse(3, Seq.empty),
      Vectors.sparse(3, Seq.empty),
      Vectors.sparse(3, Seq((0, 1.0), (1, 2.0))),
      Vectors.sparse(3, Seq((0, 10.0), (2, 20.0))),
    )
    val df = data.map(Tuple1.apply).toDF("features")
    df.show()
    val rowsRDD = df.rdd.map {
      case Row(features: Vector) => features
    }
    val results = rowsRDD.coalesce(1).mapPartitions { it: Iterator[Vector] =>
      val vectors: Array[Vector] = it.toArray
      val numColumns = vectors(0).size
      val CSRNumericTable = {
        OneDAL.vectorsToSparseNumericTable(vectors, numColumns)
      }
      Iterator(CSRNumericTable.getCNumericTable)
    }.collect()
    val csr = OneDAL.makeNumericTable(results(0))
    val resultMatrix = OneDAL.numericTableToMatrix(csr)
    val matrix = Matrices.fromVectors(data)

    assert((resultMatrix.toArray sameElements matrix.toArray) === true)
  }
}
