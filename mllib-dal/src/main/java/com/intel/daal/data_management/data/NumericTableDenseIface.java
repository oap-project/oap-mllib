/* file: NumericTableDenseIface.java */
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

import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

interface NumericTableDenseIface {

  /**
   * Reads block of rows from the table and returns it to java.nio.DoubleBuffer. This method needs
   * to be defined by user in the subclass of this class.
   *
   * @param vectorIndex Index of the first row to include into the block
   * @param vectorNum Number of rows in the block
   * @param buf Buffer to store results
   *
   * @return Block of table rows packed into DoubleBuffer
   */
  DoubleBuffer getBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf)
      throws IllegalAccessException;

  /**
   * Reads block of rows from the table and returns it to java.nio.FloatBuffer. This method needs to
   * be defined by user in the subclass of this class.
   *
   * @param vectorIndex Index of the first row to include into the block
   * @param vectorNum Number of rows in the block
   * @param buf Buffer to store results
   *
   * @return Block of table rows packed into FloatBuffer
   */
  FloatBuffer getBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf)
      throws IllegalAccessException;

  /**
   * Reads block of rows from the table and returns it to java.nio.IntBuffer. This method needs to
   * be defined by user in the subclass of this class.
   *
   * @param vectorIndex Index of the first row to include into the block
   * @param vectorNum Number of rows in the block
   * @param buf Buffer to store results
   *
   * @return Block of table rows packed into IntBuffer
   */
  IntBuffer getBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf)
      throws IllegalAccessException;

  /**
   * Transfers the data from the input DoubleBuffer into a block of table rows. This function needs
   * to be defined by user in the subclass of this class.
   *
   * @param vectorIndex Index of the first row to include into the block
   * @param vectorNum Number of rows in the block
   * @param buf Input DoubleBuffer with the capacity vectorNum * nColumns, where nColumns is the
   *        number of columns in the table
   */
  void releaseBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf)
      throws IllegalAccessException;

  /**
   * Transfers the data from the input FloatBuffer into a block of table rows. This function needs
   * to be defined by user in the subclass of this class.
   *
   * @param vectorIndex Index of the first row to include into the block
   * @param vectorNum Number of rows in the block
   * @param buf Input FloatBuffer with the capacity vectorNum * nColumns, where nColumns is the
   *        number of columns in the table
   */
  void releaseBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf)
      throws IllegalAccessException;

  /**
   * Transfers the data from the input IntBuffer into a block of table rows. This function needs to
   * be defined by user in the subclass of this class.
   *
   * @param vectorIndex Index of the first row to include into the block
   * @param vectorNum Number of rows in the block
   * @param buf Input IntBuffer with the capacity vectorNum * nColumns, where nColumns is the number
   *        of columns in the table
   */
  void releaseBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf)
      throws IllegalAccessException;

  /**
   * Gets block of values for a given feature and returns it to java.nio.DoubleBuffer. This function
   * needs to be defined by user in the subclass of this class.
   *
   * @param featureIndex Index of the feature
   * @param vectorIndex Index of the first row to include into the block
   * @param vectorNum Number of values in the block
   * @param buf Buffer to store results
   *
   * @return Block of values of the feature packed into the DoubleBuffer
   */
  DoubleBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex,
      long vectorNum, DoubleBuffer buf) throws IllegalAccessException;

  /**
   * Gets block of values for a given feature and returns it to java.nio.FloatBuffer. This function
   * needs to be defined by user in the subclass of this class.
   *
   * @param featureIndex Index of the feature
   * @param vectorIndex Index of the first row to include into the block
   * @param vectorNum Number of values in the block
   * @param buf Buffer to store results
   *
   * @return Block of values of the feature packed into the FloatBuffer
   */
  FloatBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex,
      long vectorNum, FloatBuffer buf) throws IllegalAccessException;

  /**
   * Gets block of values for a given feature and returns it to java.nio.IntBuffer. This function
   * needs to be defined by user in the subclass of this class.
   *
   * @param featureIndex Index of the feature
   * @param vectorIndex Index of the first row to include into the block
   * @param vectorNum Number of values in the block
   * @param buf Buffer to store results
   *
   * @return Block of values of the feature packed into the IntBuffer
   */
  IntBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex,
      long vectorNum, IntBuffer buf) throws IllegalAccessException;

  /**
   * Transfers the values of a given feature from the input DoubleBuffer into a block of values of
   * the feature in the table. This function needs to be defined by user in the subclass of this
   * class.
   *
   * @param featureIndex Index of the feature
   * @param vectorIndex Index of the first row to include into the block
   * @param vectorNum Number of values in the block
   * @param buf Input DoubleBuffer of size vectorNum
   */
  void releaseBlockOfColumnValues(long featureIndex, long vectorIndex,
      long vectorNum, DoubleBuffer buf) throws IllegalAccessException;

  /**
   * Transfers the values of a given feature from the input FloatBuffer into a block of values of
   * the feature in the table. This function needs to be defined by user in the subclass of this
   * class.
   *
   * @param featureIndex Index of the feature
   * @param vectorIndex Index of the first row to include into the block
   * @param vectorNum Number of values in the block
   * @param buf Input FloatBuffer of size vectorNum
   */
  void releaseBlockOfColumnValues(long featureIndex, long vectorIndex,
      long vectorNum, FloatBuffer buf) throws IllegalAccessException;

  /**
   * Transfers the values of a given feature from the input IntBuffer into a block of values of the
   * feature in the table. This function needs to be defined by user in the subclass of this class.
   *
   * @param featureIndex Index of the feature
   * @param vectorIndex Index of the first row to include into the block
   * @param vectorNum Number of values in the block
   * @param buf Input IntBuffer of size vectorNum
   */
  void releaseBlockOfColumnValues(long featureIndex, long vectorIndex,
      long vectorNum, IntBuffer buf) throws IllegalAccessException;
}
/** @} */
