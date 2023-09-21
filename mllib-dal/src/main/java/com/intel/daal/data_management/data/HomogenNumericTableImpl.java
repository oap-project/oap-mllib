/* file: HomogenNumericTableImpl.java */
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
 * <a name="DAAL-CLASS-DATA__HOMOGENNUMERICTABLEIMPL__HOMOGENNUMERICTABLEIMPL"></a>
 * @brief A derivative class of the NumericTableImpl class, that provides common interfaces for
 *        different implementations of a homogen numeric table
 */
public abstract class HomogenNumericTableImpl extends NumericTableImpl {
  protected Class<? extends Number> type;

  /** @private */
  static {
    try {
      LibLoader.loadLibraries();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * Constructs the homogen numeric table
   * @param context Context to manage the homogen numeric table
   */
  public HomogenNumericTableImpl(DaalContext context) {
    super(context);
  }

  public abstract void assign(long constValue);

  public abstract void assign(int constValue);

  public abstract void assign(double constValue);

  public abstract void assign(float constValue);

  public abstract double[] getDoubleArray();

  public abstract float[] getFloatArray();

  public abstract long[] getLongArray();

  public abstract Object getDataObject();

  public abstract Class<? extends Number> getNumericType();

  public abstract void set(long row, long column, double value);

  public abstract void set(long row, long column, float value);

  public abstract void set(long row, long column, long value);

  public abstract void set(long row, long column, int value);

  public abstract double getDouble(long row, long column);

  public abstract float getFloat(long row, long column);

  public abstract long getLong(long row, long column);

  public abstract int getInt(long row, long column);
}
/** @} */
