package com.intel.oap.mllib

import com.intel.oap.mllib.stat.{SummarizerDALImpl, SummarizerResult}
import com.intel.oneapi.dal.table.{Common, HomogenTable}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.FunctionsSuite
import org.junit.jupiter.api.Assertions.assertArrayEquals

class MomentsHomogenTableSuite extends FunctionsSuite with Logging{
    test("test compute low order moments HomogenTable") {
        val expectMean = Array(-0.069474,0.329666,0.152558,0.202047,-0.323516,-0.471058,0.162655,0.347832,-0.656644,0.213516)
        val expectVariance = Array(36.423088,32.901699,30.471611,29.994194,33.665047,30.980848,32.892132,33.231438,34.720554,35.646618)
        val expectMin = Array(-9.948613,-9.737347,-9.619284,-9.743521,-9.990568,-9.997300,-9.997706,-9.897363,-9.794658,-9.861839)
        val expectMax = Array(9.900860,9.869278,9.997498,9.868528,9.404993,9.768626,9.910092,9.896053,9.647098,9.971762)

        val sourceData = TestCommon.readCSV("src/test/scala/data/covcormoments_dense.csv")

        val dataTable = new HomogenTable(sourceData.length, sourceData(0).length, TestCommon.convertArray(sourceData), Common.ComputeDevice.HOST);
        val summarizerDAL = new SummarizerDALImpl(1, 1)
        val result = new SummarizerResult()
        summarizerDAL.cSummarizerTrainDAL(dataTable.getcObejct(), 1, Common.ComputeDevice.HOST.ordinal(), 0, "127.0.0.1_3000", result)
        val meanTable = OneDAL.homogenTable1xNToVector(OneDAL.makeHomogenTable(result.meanNumericTable), Common.ComputeDevice.HOST)
        val varianceTable = OneDAL.homogenTable1xNToVector(OneDAL.makeHomogenTable(result.varianceNumericTable), Common.ComputeDevice.HOST)
        val minimumTable = OneDAL.homogenTable1xNToVector(OneDAL.makeHomogenTable(result.minimumNumericTable), Common.ComputeDevice.HOST)
        val maximumTable = OneDAL.homogenTable1xNToVector(OneDAL.makeHomogenTable(result.maximumNumericTable), Common.ComputeDevice.HOST)

        assertArrayEquals(expectMean , meanTable.toArray, 0.000001)
        assertArrayEquals(expectVariance, varianceTable.toDense.values, 0.000001)
        assertArrayEquals(expectMin , minimumTable.toArray, 0.000001)
        assertArrayEquals(expectMax, maximumTable.toDense.values, 0.000001)
    }
}
