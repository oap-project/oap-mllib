#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
An example for summarizer.
"""
from __future__ import print_function
import sys

from pyspark.sql import SparkSession
import numpy as np
from pyspark.mllib.stat import Statistics

if __name__ == "__main__":
    spark = SparkSession.builder.appName("SummaryStatisticsExample").getOrCreate()
    data = spark.parallelize([
        np.array([5.308206,9.869278,1.018934,4.292158,6.081011,6.585723,2.411094,4.767308,-3.256320,-6.029562]),
        np.array([7.279464,0.390664,-9.619284,3.435376,-4.769490,-4.873188,-0.118791,-5.117316,-0.418655,-0.475422]),
        np.array([-6.615791,-6.191542,0.402459,-9.743521,-9.990568,9.105346,1.691312,-2.605659,9.534952,-7.829027]),
        np.array([-4.792007,-2.491098,-2.939393,8.086467,3.773812,-9.997300,0.222378,8.995244,-5.753282,6.091060]),
        np.array([7.700725,-6.414918,1.684476,-8.983361,4.284580,-9.017608,0.552379,-7.705741,2.589852,0.411561]),
        np.array([6.991900,-1.063721,9.321163,-0.429719,-2.167696,-1.736425,-0.919139,6.980681,-0.711914,3.414347]),
        np.array([5.794488,-1.062261,0.955322,0.389642,3.012921,-9.953994,-3.197309,3.992421,-6.935902,8.147622]),
        np.array([-2.486670,6.973242,-4.047004,-5.655629,5.081786,5.533859,7.821403,2.763313,-0.454056,6.554309]),
        np.array([.204855,7.839522,7.381886,1.618749,-6.566877,7.584285,-8.355983,-5.501410,-8.191205,-2.608499]),
        np.array([-9.948613,-8.941953,-8.106389,4.863542,5.852806,-1.659259,6.342504,-8.190106,-3.110330,-7.484658])
    ])

    # Compute column summary statistics.
    summary = Statistics.colStats(data)
    print(summary.mean())  # a dense vector containing the mean value for each column
    print(summary.variance())  # column-wise variance
    print(summary.numNonzeros())  # number of nonzeros in each column
    # $example off$

    spark.stop()