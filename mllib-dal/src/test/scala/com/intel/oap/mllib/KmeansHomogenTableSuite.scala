package com.intel.oap.mllib

import com.intel.oap.mllib.clustering.{KMeansDALImpl, KMeansResult}
import com.intel.oneapi.dal.table.{Common, HomogenTable}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.FunctionsSuite
import org.apache.spark.ml.linalg.Vectors
import org.junit.jupiter.api.Assertions.assertArrayEquals

class KmeansHomogenTableSuite extends FunctionsSuite with Logging {

    test("test compute Kmeans HomogenTable") {
        val expectCentroids = Array(
            Vectors.dense(46.964348, 53.593376, -52.973740, -39.790489, -70.001411, 71.253716, -47.509434, -98.567551, -61.581421, -78.373512, -76.407326, 52.974354, 54.360466, 93.591499, -95.720619, -13.518152, 7.915907, -0.310325, -87.498383, -3.236778),
            Vectors.dense(28.443605, 69.337891, -19.093294, -84.507889, -62.735977, -75.870331, 71.256622, -3.230225, 90.361931, 91.283150, 32.653885, 29.934725, 96.525673, -49.022247, 19.994081, -75.833252, 91.019196, -19.009062, -77.813286, -49.776817),
            Vectors.dense(-17.750511, -69.300621, -37.097565, -88.967613, -80.669693, -77.764206, 0.116336, 58.215664, 10.486547, 12.495481, -34.732445, 59.756210, 23.947205, 82.748268, -44.530640, 61.583588, -76.279083, -53.863396, -79.033432, 81.378654),
            Vectors.dense(-30.346266, 51.566120, -59.674381, 30.369638, -59.394058, 44.835983, 8.174916, 21.822199, -3.449510, -28.901081, 64.939026, -29.098310, 27.282028, 55.515087, -30.403708, 25.711794, -23.571680, -44.271053, -10.143802, -85.426796),
            Vectors.dense(31.456154, 49.778744, 31.556364, 65.320984, 65.146805, 19.129190, 22.527151, -40.797512, 20.346115, 44.843136, 28.901960, 31.593611, -62.861885, 74.933914, -26.116434, -49.510811, -25.313290, -4.823534, -28.099617, -26.868441),
            Vectors.dense(-79.348816, -12.127685, 44.048122, 62.684948, 13.492962, -54.889351, -43.606674, -9.992040, -28.723059, -9.103136, -32.303638, 3.731686, -4.291812, -12.375703, -92.110191, 28.482073, 66.563210, -88.285103, -47.353642, 52.795681),
            Vectors.dense(95.933769, -24.241076, 100.190689, -72.031937, -99.445274, 52.278160, 28.386942, 42.388264, -31.064482, 31.566105, 36.302757, -4.336817, 65.888016, -80.738472, -30.357895, -86.050514, 91.286636, 81.214050, 85.946823, 58.023907),
            Vectors.dense(9.264688, -21.057188, -57.136391, 97.944443, -95.647324, -57.414806, 82.151398, 72.063293, -38.974258, -21.996706, 30.306473, -94.495514, 32.717960, -7.012361, -66.483154, 87.542542, -64.854294, 59.850533, -59.211372, -83.742035),
            Vectors.dense(-0.653824, 73.624718, -60.380787, 8.740559, 0.580889, -95.541794, 25.895468, 56.856102, -59.315926, 12.156931, -24.910173, 99.267456, 83.599007, -23.720819, -43.849953, -57.535759, 17.560658, 82.451462, 47.234901, -96.285553),
            Vectors.dense(27.077707, -20.899839, 44.962280, 24.264837, 23.276260, -0.962210, -16.296940, -33.444206, 51.587513, 43.147736, 21.356207, -2.550465, 0.378649, 4.149162, 56.484566, -3.075402, -30.369482, 27.513243, -17.073362, 57.726898))

        val sourceData = TestCommon.readCSV("src/test/scala/data/kmeans_dense_train_data.csv")
        val centroidData = TestCommon.readCSV("src/test/scala/data/kmeans_dense_train_centroids.csv")

        val dataTable = new HomogenTable(sourceData.length, sourceData(0).length, TestCommon.convertArray(sourceData), TestCommon.getComputeDevice);
        val centroidsTable = new HomogenTable(centroidData.length, centroidData(0).length, TestCommon.convertArray(centroidData), TestCommon.getComputeDevice);

        val kmeansDAL = new KMeansDALImpl(0, 0, 0,
            null, null, 0, 0);
        val result = new KMeansResult();
        val centroids = kmeansDAL.cKMeansOneapiComputeWithInitCenters(dataTable.getcObejct(), centroidsTable.getcObejct(),10, 0.001,
            5, 1, TestCommon.getComputeDevice.ordinal(), 0, "127.0.0.1_3000" , result);
        val resultVectors = OneDAL.homogenTableToVectors(OneDAL.makeHomogenTable(centroids), TestCommon.getComputeDevice);
        assertArrayEquals(TestCommon.convertArray(expectCentroids), TestCommon.convertArray(resultVectors), 0.000001)
    }
}
