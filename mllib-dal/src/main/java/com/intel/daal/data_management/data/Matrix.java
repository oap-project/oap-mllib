/* file: Matrix.java */
/*******************************************************************************
 * Copyright 2014 Intel Corporation
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

/**
 * @ingroup numeric_tables @{
 */
package com.intel.daal.data_management.data;

import com.intel.daal.services.DaalContext;
import com.intel.oap.mllib.LibLoader;

import java.io.IOException;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__MATRIX"></a>
 * @brief A derivative class of the NumericTable class, that provides methods to access the data
 *        that is stored as a contiguous array of homogeneous feature vectors. Table rows contain
 *        feature vectors, and columns contain values of individual features.
 */
public class Matrix extends HomogenNumericTable {

  /** @private */
  static {
    try {
      LibLoader.loadLibraries();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Constructs homogeneous numeric table using implementation provided by user
   *
   * @param context Context to manage created matrix
   * @param impl Implementation of matrix
   */
  public Matrix(DaalContext context, HomogenNumericTableImpl impl) {
    super(context, impl);
  }

  /**
   * Constructs homogeneous numeric table from C++ homogeneous numeric table
   * @param context Context to manage created matrix
   * @param cTable Pointer to C++ numeric table
   */
  public Matrix(DaalContext context, long cTable) {
    super(context, cTable);
  }

  public Matrix(DaalContext context, Class<? extends Number> cls, long nColumns) {
    super(context, cls, nColumns);
  }

  public Matrix(DaalContext context, DataDictionary.FeaturesEqual featuresEqual,
      Class<? extends Number> cls, long nColumns) {
    super(context, featuresEqual, cls, nColumns);
  }

  public Matrix(DaalContext context, Class<? extends Number> cls, long nColumns, long nRows,
      AllocationFlag allocFlag) {
    super(context, cls, nColumns, nRows, allocFlag);
  }

  public Matrix(DaalContext context, DataDictionary.FeaturesEqual featuresEqual,
      Class<? extends Number> cls, long nColumns, long nRows, AllocationFlag allocFlag) {
    super(context, featuresEqual, cls, nColumns, nRows, allocFlag);
  }

  public void set(long row, long column, double value) {
    ((HomogenNumericTableImpl) tableImpl).set(row, column, value);
  }

  public void set(long row, long column, float value) {
    ((HomogenNumericTableImpl) tableImpl).set(row, column, value);
  }

  public void set(long row, long column, long value) {
    ((HomogenNumericTableImpl) tableImpl).set(row, column, value);
  }

  public void set(long row, long column, int value) {
    ((HomogenNumericTableImpl) tableImpl).set(row, column, value);
  }

  public double getDouble(long row, long column) {
    return ((HomogenNumericTableImpl) tableImpl).getDouble(row, column);
  }

  public float getFloat(long row, long column) {
    return ((HomogenNumericTableImpl) tableImpl).getFloat(row, column);
  }

  public long getLong(long row, long column) {
    return ((HomogenNumericTableImpl) tableImpl).getLong(row, column);
  }

  public int getInt(long row, long column) {
    return ((HomogenNumericTableImpl) tableImpl).getInt(row, column);
  }
}
/** @} */
