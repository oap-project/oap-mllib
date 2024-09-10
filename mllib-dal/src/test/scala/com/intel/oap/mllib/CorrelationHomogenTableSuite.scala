/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.oap.mllib

import com.intel.oap.mllib.feature.{PCADALImpl, PCAResult}
import com.intel.oap.mllib.stat.{CorrelationDALImpl, CorrelationResult}
import com.intel.oneapi.dal.table.{Common, HomogenTable}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.FunctionsSuite
import org.apache.spark.ml.linalg.{DenseMatrix, Vectors}
import org.junit.jupiter.api.Assertions.assertArrayEquals

class CorrelationHomogenTableSuite extends FunctionsSuite with Logging {

    test("test compute Correlation HomogenTable") {
        val expectCorrelation = Array(
            Vectors.dense(1.000000,-0.048620,0.074018,0.091225,-0.046254,-0.064293,-0.095292,-0.087823,-0.038801,0.010893),
            Vectors.dense(-0.048620,1.000000,0.031603,-0.097796,-0.034013,0.067157,0.057825,-0.092420,-0.044481,-0.076968),
            Vectors.dense(0.074018,0.031603,1.000000,-0.029907,-0.052421,-0.057258,-0.022879,0.024892,-0.006728,0.023333),
            Vectors.dense(0.091225,-0.097796,-0.029907,1.000000,-0.011731,-0.111816,0.044164,-0.077831,0.039770,0.021695),
            Vectors.dense(-0.046254,-0.034013,-0.052421,-0.011731,1.000000,-0.102220,-0.038227,0.037525,0.014982,0.041524),
            Vectors.dense(-0.064293,0.067157,-0.057258,-0.111816,-0.102220,1.000000,0.095177,0.052274,0.100059,0.082077),
            Vectors.dense(-0.095292,0.057825,-0.022879,0.044164,-0.038227,0.095177,1.000000,-0.076685,-0.061039,0.053211),
            Vectors.dense(-0.087823,-0.092420,0.024892,-0.077831,0.037525,0.052274,-0.076685,1.000000,-0.020806,0.157492),
            Vectors.dense(-0.038801,-0.044481,-0.006728,0.039770,0.014982,0.100059,-0.061039,-0.020806,1.000000,-0.058518),
            Vectors.dense(0.010893,-0.076968,0.023333,0.021695,0.041524,0.082077,0.053211,0.157492,-0.058518,1.000000))
        val sourceData = TestCommon.readCSV("src/test/resources/data/covcormoments_dense.csv")

        val dataTable = new HomogenTable(sourceData.length, sourceData(0).length, TestCommon.convertArray(sourceData), TestCommon.getComputeDevice);

        val correlationDAL = new CorrelationDALImpl(1, 1)
        val gpuIndices = Array(0)
        val result = new CorrelationResult()
        correlationDAL.cCorrelationTrainDAL(0, dataTable.getcObejct(), sourceData.length, sourceData(0).length, 1, 1, Common.ComputeDevice.HOST.ordinal(), gpuIndices, "", result);
        val correlationMatrix = TestCommon.getMatrixFromTable(OneDAL.makeHomogenTable(
            result.getCorrelationNumericTable))

        assertArrayEquals(TestCommon.convertArray(expectCorrelation), correlationMatrix.toArray, 0.000001)
    }
}
