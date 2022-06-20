package com.intel.oap.mllib

import org.junit.jupiter.api.Assertions.assertArrayEquals

import com.intel.oap.mllib.feature.{PCADALImpl, PCAResult}
import com.intel.oneapi.dal.table.{Common, HomogenTable}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.FunctionsSuite
import org.apache.spark.ml.linalg.Vectors

class PCAHomogenTableSuite extends FunctionsSuite with Logging {

    test("test compute Kmeans HomogenTable") {
        val expectPC = Array(
            Vectors.dense(0.274023, 0.364611, 0.257968, 0.336951, 0.359581, 0.259052, 0.266801, 0.362287, 0.267906, 0.375863),
            Vectors.dense(0.251998, -0.409731, 0.471080, -0.238914, 0.047093, -0.114247, 0.392579, -0.264167, 0.454419, -0.209647),
            Vectors.dense(-0.179943, -0.047758, -0.085144, 0.489979, -0.493983, 0.502416, 0.042186, -0.085014, 0.363258, -0.283913),
            Vectors.dense(-0.451830, 0.223971, -0.282017, 0.104536, 0.050499, -0.294610, 0.560781, -0.404184, 0.125200, 0.269008),
            Vectors.dense(-0.521160, -0.058449, 0.563364, -0.053685, -0.283515, -0.024779, 0.248268, 0.403089, -0.312460, 0.044388))
        val expectExplainedVariance = Array(3.492693, 1.677875, 1.529460, 1.147213, 0.769543)
        val sourceData = TestCommon.readCSV("src/test/scala/data/pca_normalized.csv")

        val dataTable = new HomogenTable(sourceData.length, sourceData(0).length, TestCommon.convertArray(sourceData), Common.ComputeDevice.HOST);

        val pcaDAL = new PCADALImpl(5, 1, 1)
        val result = new PCAResult()
        pcaDAL.cPCATrainDAL(dataTable.getcObejct(), 1, Common.ComputeDevice.HOST.ordinal(), 0, "127.0.0.1_3000", result);
        println(result.explainedVarianceNumericTable)
        val pcNumericTable = OneDAL.makeHomogenTable(result.pcNumericTable)
        val explainedVarianceNumericTable = OneDAL.makeHomogenTable(
            result.explainedVarianceNumericTable)
        val principleComponents = pcaDAL.getPrincipleComponentsFromDAL(pcNumericTable, 5, Common.ComputeDevice.HOST)
        val explainedVariance = pcaDAL.getPrincipleComponentsFromDAL(explainedVarianceNumericTable, 1, Common.ComputeDevice.HOST)

        assertArrayEquals(expectExplainedVariance , explainedVariance.values, 0.000001)
        assertArrayEquals(TestCommon.convertArray(expectPC), principleComponents.values, 0.000001)
    }
}
