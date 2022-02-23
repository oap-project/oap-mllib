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
package org.apache.spark.ml

import com.intel.oap.mllib.{OneCCL, OneDAL}
import com.intel.oap.mllib.recommendation.{ALSPartitionInfo, ALSResult}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.recommendation.ALS.Rating

class oneDALSuite extends FunctionsSuite with Logging {
  import testImplicits._

  test("test buffer to CSRNumericTable") {
    val data = Seq(
      Rating(0, 17, 0.938283.toFloat),
      Rating(1, 1, 0.124207.toFloat),
      Rating(1, 8, 0.411504.toFloat),
//      Rating(),
//      Rating(),
      Rating(1, 8, 0.411504.toFloat),
      Rating(0, 17, 0.938283.toFloat),
      Rating(1, 13, 0.746992.toFloat),
    )

  }
}
