/*******************************************************************************
 * Copyright 2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 *******************************************************************************/
package com.intel.oap.mllib.stat;

public class SummarizerResult {
  public long meanNumericTable;
  public long varianceNumericTable;
  public long minimumNumericTable;
  public long maximumNumericTable;

  public long getMeanNumericTable() {
    return meanNumericTable;
  }

  public void setMeanNumericTable(long meanNumericTable) {
    this.meanNumericTable = meanNumericTable;
  }

  public long getVarianceNumericTable() {
    return varianceNumericTable;
  }

  public void setVarianceNumericTable(long varianceNumericTable) {
    this.varianceNumericTable = varianceNumericTable;
  }

  public long getMinimumNumericTable() {
    return minimumNumericTable;
  }

  public void setMinimumNumericTable(long minimumNumericTable) {
    this.minimumNumericTable = minimumNumericTable;
  }

  public long getMaximumNumericTable() {
    return maximumNumericTable;
  }

  public void setMaximumNumericTable(long maximumNumericTable) {
    this.maximumNumericTable = maximumNumericTable;
  }
}
