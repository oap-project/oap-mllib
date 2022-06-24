package com.intel.oap.mllib

import org.junit.jupiter.api.Assertions.assertArrayEquals
import com.intel.oap.mllib.feature.{PCADALImpl, PCAResult}
import com.intel.oneapi.dal.table.{Common, HomogenTable}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.FunctionsSuite
import org.apache.spark.ml.linalg.Vectors

class PCAHomogenTableSuite extends FunctionsSuite with Logging {

    test("test compute PCA HomogenTable with normalized data") {
        val expectPC = Array(
            Vectors.dense(0.274023,0.364611,0.257968,0.336951,0.359581,0.259052,0.266801,0.362287,0.267906,0.375863),
            Vectors.dense(-0.251998,0.409731,-0.471080,0.238914,-0.047093,0.114247,-0.392579,0.264167,-0.454419,0.209647),
            Vectors.dense(0.179943,0.047758,0.085144,-0.489979,0.493983,-0.502416,-0.042186,0.085014,-0.363258,0.283913),
            Vectors.dense(0.451830,-0.223971,0.282017,-0.104536,-0.050499,0.294610,-0.560781,0.404184,-0.125200,-0.269008),
            Vectors.dense(0.521160,0.058449,-0.563364,0.053685,0.283515,0.024779,-0.248268,-0.403089,0.312460,-0.044388),
            Vectors.dense(-0.416755,0.479102, 0.205644, -0.279602, 0.352750, 0.114219, -0.281338, -0.086593, 0.303044, -0.400433),
            Vectors.dense(-0.103690,-0.154297, 0.083978, -0.314329, 0.106878, 0.689745, -0.007447, -0.400096, -0.208115, 0.411060),
            Vectors.dense(0.177689,0.156514, -0.309582, -0.257324, 0.084021, 0.289763, 0.563550, 0.197458, -0.306011, -0.490540),
            Vectors.dense(0.147065,0.152258, 0.366147, 0.494387, 0.172869, -0.055088, 0.003342, -0.473715, -0.497569, -0.267949),
            Vectors.dense(-0.340048,-0.585059, -0.167712, 0.296446, 0.607235, 0.064941, 0.000978, 0.186870, -0.010096, -0.134500)
        )
        val expectExplainedVariance = Array(3.492693, 1.677875, 1.529460, 1.147213, 0.769543, 0.731955, 0.507616, 0.075259, 0.047056, 0.021329)
        val sourceData = TestCommon.readCSV("src/test/resources/data/pca_normalized.csv")

        val dataTable = new HomogenTable(sourceData.length, sourceData(0).length, TestCommon.convertArray(sourceData), TestCommon.getComputeDevice);

        val pcaDAL = new PCADALImpl(5, 1, 1)
        val result = new PCAResult()
        pcaDAL.cPCATrainDAL(dataTable.getcObejct(), 1, TestCommon.getComputeDevice.ordinal(), 0, "127.0.0.1_3000", result);
        val pcNumericTable = OneDAL.makeHomogenTable(result.pcNumericTable)
        val explainedVarianceNumericTable = OneDAL.makeHomogenTable(
            result.explainedVarianceNumericTable)
        val principleComponents = OneDAL.homogenTableToMatrix(pcNumericTable, TestCommon.getComputeDevice)
        val explainedVariance = OneDAL.homogenTable1xNToVector(explainedVarianceNumericTable, TestCommon.getComputeDevice)

        assertArrayEquals(expectExplainedVariance , explainedVariance.toArray, 0.000001)
        assertArrayEquals(TestCommon.convertArray(expectPC), principleComponents.toDense.values, 0.000001)
    }

    test("test compute PCA HomogenTable with not normalized data ") {
        val expectPC = Array(
            Vectors.dense(0.469204,-0.237334,0.123207,0.420058,0.076038,-0.562361,-0.284237,-0.285238,-0.036184,-0.215406),
            Vectors.dense(0.006613,0.490091,-0.019286,-0.023062,-0.305032,0.092426,0.256427,-0.589697,0.028859,-0.492895),
            Vectors.dense(-0.346932,-0.174710,-0.535556,0.104451,0.374790,0.001271,-0.136289,-0.096028,0.530453,-0.318965),
            Vectors.dense(0.037866,-0.248372,-0.288024,0.574567,-0.128985,0.146502,0.581490,-0.165897,-0.044184,0.343640),
            Vectors.dense(-0.285377,0.206258,-0.197166,-0.084090,0.483905,-0.424668,0.250495,-0.053075,-0.591567,-0.007014),
            Vectors.dense(0.278531,-0.119468,-0.691069,-0.185632,-0.320395,0.141864,-0.303984,0.040902,-0.414600,-0.085800),
            Vectors.dense(-0.519485,-0.333741,0.110594,0.103602,-0.533023,-0.293129,0.079634,0.269025,-0.129069,-0.361076),
            Vectors.dense(-0.142885,0.640603,-0.138803,0.542539,-0.143815,-0.082322,-0.331441,0.307685,0.033148,0.150048),
            Vectors.dense(-0.378262,-0.062825,-0.019166,-0.161871,-0.231809,-0.205942,-0.332073,-0.562371,0.051523,0.548695),
            Vectors.dense(0.256176,0.169571,-0.257595,-0.330404,-0.220727,-0.564058,0.339305,0.215309,0.415052,0.171641)
        )
        val expectExplainedVariance = Array(1.304369, 1.252785, 1.120113, 1.114487, 1.083063, 0.942371, 0.909094, 0.825540, 0.746623, 0.701554)
        val sourceData = TestCommon.readCSV("src/test/resources/data/pca_no_normalized.csv")

        val dataTable = new HomogenTable(sourceData.length, sourceData(0).length, TestCommon.convertArray(sourceData), TestCommon.getComputeDevice);

        val pcaDAL = new PCADALImpl(5, 1, 1)
        val result = new PCAResult()
        pcaDAL.cPCATrainDAL(dataTable.getcObejct(), 1, TestCommon.getComputeDevice.ordinal(), 0, "127.0.0.1_3000", result);
        val pcNumericTable = OneDAL.makeHomogenTable(result.pcNumericTable)
        val explainedVarianceNumericTable = OneDAL.makeHomogenTable(
            result.explainedVarianceNumericTable)
        val principleComponents = OneDAL.homogenTableToMatrix(pcNumericTable, TestCommon.getComputeDevice)
        val explainedVariance = OneDAL.homogenTable1xNToVector(explainedVarianceNumericTable, TestCommon.getComputeDevice)

        assertArrayEquals(expectExplainedVariance , explainedVariance.toArray, 0.000001)
        assertArrayEquals(TestCommon.convertArray(expectPC), principleComponents.toDense.values, 0.000001)
    }
}
