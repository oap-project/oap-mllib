/*
 * Copyright 2020 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Based on oneDAL Java example code

package org.apache.spark.ml.util;

import com.intel.daal.data_management.data.CSRNumericTable;
import com.intel.daal.data_management.data.HomogenNumericTable;
import com.intel.daal.data_management.data.NumericTable;
import com.intel.daal.services.DaalContext;
import com.intel.daal.services.ErrorHandling;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.text.DecimalFormat;
import java.util.ArrayList;

public class Service {

  public static void printMatrix(double[] matrix, int nCols, int nRows, String header) {
    System.out.println(header);
    DecimalFormat numberFormat = new DecimalFormat("##0.00");
    for (int i = 0; i < nRows; i++) {
      for (int j = 0; j < nCols; j++) {
        System.out.print(numberFormat.format(matrix[i * nCols + j]) + "\t\t");
      }
      System.out.println();
    }
  }

  public static void printTriangularMatrix(double[] triangularMatrix, int nDimensions,
                                           String header) {
    int index = 0;
    for (int i = 0; i < nDimensions; i++) {
      for (int j = 0; j <= i; j++) {
        System.out.print(triangularMatrix[index++] + "   ");
      }
      System.out.println();
    }
  }

  public static void printPackedNumericTable(HomogenNumericTable nt, long nDimensions,
                                             String header) {
    double[] results = nt.getDoubleArray();
    printTriangularMatrix(results, (int) nDimensions, header);
  }

  public static boolean isUpper(NumericTable.StorageLayout layout) {
    return layout.ordinal() == NumericTable.StorageLayout.upperPackedSymmetricMatrix.ordinal()
            || layout.ordinal() == NumericTable.StorageLayout.upperPackedTriangularMatrix.ordinal();
  }

  public static boolean isLower(NumericTable.StorageLayout layout) {
    return layout.ordinal() == NumericTable.StorageLayout.lowerPackedSymmetricMatrix.ordinal()
            || layout.ordinal() == NumericTable.StorageLayout.lowerPackedTriangularMatrix.ordinal();
  }

  public static void printNumericTable(String header, NumericTable nt,
                                       long nPrintedRows, long nPrintedCols) {
    long nNtCols = nt.getNumberOfColumns();
    long nNtRows = nt.getNumberOfRows();
    long nRows = nNtRows;
    long nCols = nNtCols;

    NumericTable.StorageLayout layout = nt.getDataLayout();

    if (nPrintedRows > 0) {
      nRows = Math.min(nNtRows, nPrintedRows);
    }

    FloatBuffer result = FloatBuffer.allocate((int) (nNtCols * nRows));
    try {
      result = nt.getBlockOfRows(0, nRows, result);
    } catch (IllegalAccessException e) {
      ErrorHandling.printThrowable(e);
      return;
    }
    if (nPrintedCols > 0) {
      nCols = Math.min(nNtCols, nPrintedCols);
    }

    StringBuilder builder = new StringBuilder();
    builder.append(header);
    builder.append("\n");

    if (isLower(layout)) {
      for (long i = 0; i < nRows; i++) {
        for (long j = 0; j <= i; j++) {
          String tmp = String.format("%-6.3f   ", result.get((int) (i * nNtCols + j)));
          builder.append(tmp);
        }
        builder.append("\n");
      }
    } else if (isUpper(layout)) {

      for (long i = 0; i < nRows; i++) {
        for (int k = 0; k < i; k++) {
          builder.append("         ");
        }
        for (long j = i; j < nCols; j++) {
          String tmp = String.format("%-6.3f   ", result.get((int) (i * nNtCols + j)));
          builder.append(tmp);
        }
        builder.append("\n");
      }

    } else if (isLower(layout) != true && isUpper(layout) != true) {
      for (long i = 0; i < nRows; i++) {
        for (long j = 0; j < nCols; j++) {
          String tmp = String.format("%-6.3f   ", result.get((int) (i * nNtCols + j)));
          builder.append(tmp);
        }
        builder.append("\n");
      }
    }
    System.out.println(builder.toString());
  }

  public static void printNumericTable(String header, CSRNumericTable nt,
                                       long nPrintedRows, long nPrintedCols) {
    long[] rowOffsets = nt.getRowOffsetsArray();
    long[] colIndices = nt.getColIndicesArray();
    float[] values = nt.getFloatArray();

    long nNtCols = nt.getNumberOfColumns();
    long nNtRows = nt.getNumberOfRows();
    long nRows = nNtRows;
    long nCols = nNtCols;

    if (nPrintedRows > 0) {
      nRows = Math.min(nNtRows, nPrintedRows);
    }

    if (nPrintedCols > 0) {
      nCols = Math.min(nNtCols, nPrintedCols);
    }

    StringBuilder builder = new StringBuilder();
    builder.append(header);
    builder.append("\n");

    float[] oneDenseRow = new float[(int) nCols];
    for (int i = 0; i < nRows; i++) {
      for (int j = 0; j < nCols; j++) {
        oneDenseRow[j] = 0;
      }
      int nElementsInRow = (int) (rowOffsets[i + 1] - rowOffsets[i]);
      for (int k = 0; k < nElementsInRow; k++) {
        oneDenseRow[(int) (colIndices[(int) (rowOffsets[i] - 1 + k)] - 1)]
                = values[(int) (rowOffsets[i] - 1 + k)];
      }
      for (int j = 0; j < nCols; j++) {
        String tmp = String.format("%-6.3f   ", oneDenseRow[j]);
        builder.append(tmp);
      }
      builder.append("\n");
    }
    System.out.println(builder.toString());
  }

  public static void printNumericTable(String header, NumericTable nt, long nRows) {
    printNumericTable(header, nt, nRows, nt.getNumberOfColumns());
  }

  public static void printNumericTable(String header, NumericTable nt) {
    printNumericTable(header, nt, nt.getNumberOfRows());
  }

  public static void printNumericTable(String header, CSRNumericTable nt, long nRows) {
    printNumericTable(header, nt, nRows, nt.getNumberOfColumns());
  }

  public static void printNumericTable(String header, CSRNumericTable nt) {
    printNumericTable(header, nt, nt.getNumberOfRows());
  }

  public static void printNumericTables(NumericTable dataTable1, NumericTable dataTable2,
                                        String title1, String title2,
                                        String message, long nPrintedRows) {
    long nRows1 = dataTable1.getNumberOfRows();
    long nRows2 = dataTable2.getNumberOfRows();
    long nCols1 = dataTable1.getNumberOfColumns();
    long nCols2 = dataTable2.getNumberOfColumns();

    long nRows = Math.min(nRows1, nRows2);
    if (nPrintedRows > 0) {
      nRows = Math.min(Math.min(nRows1, nRows2), nPrintedRows);
    }

    FloatBuffer result1 = FloatBuffer.allocate((int) (nCols1 * nRows));
    FloatBuffer result2 = FloatBuffer.allocate((int) (nCols2 * nRows));
    try {
      result1 = dataTable1.getBlockOfRows(0, nRows, result1);
      result2 = dataTable2.getBlockOfRows(0, nRows, result2);
    } catch (IllegalAccessException e) {
      ErrorHandling.printThrowable(e);
      return;
    }
    StringBuilder builder = new StringBuilder();
    builder.append(message);
    builder.append("\n");
    builder.append(title1);

    StringBuilder builderHelp = new StringBuilder();
    for (long j = 0; j < nCols1; j++) {
      String tmp = String.format("%-6.3f   ", result1.get((int) (0 * nCols1 + j)));
      builderHelp.append(tmp);
    }
    int interval = builderHelp.length() - title1.length();

    for (int i = 0; i < interval; i++) {
      builder.append(" ");
    }
    builder.append("     ");
    builder.append(title2);
    builder.append("\n");

    for (long i = 0; i < nRows; i++) {
      for (long j = 0; j < nCols1; j++) {
        String tmp = String.format("%-6.3f   ", result1.get((int) (i * nCols1 + j)));
        builder.append(tmp);
      }
      builder.append("     ");
      for (long j = 0; j < nCols2; j++) {
        String tmp = String.format("%-6.3f   ", result2.get((int) (i * nCols2 + j)));
        builder.append(tmp);
      }
      builder.append("\n");
    }
    System.out.println(builder.toString());
  }

  public static void printALSRatings(NumericTable usersOffsetTable, NumericTable itemsOffsetTable,
                                     NumericTable ratings) {
    long nUsers = ratings.getNumberOfRows();
    long nItems = ratings.getNumberOfColumns();

    float[] ratingsData = ((HomogenNumericTable) ratings).getFloatArray();
    IntBuffer usersOffsetBuf = IntBuffer.allocate(1);
    IntBuffer itemsOffsetBuf = IntBuffer.allocate(1);
    try {
      usersOffsetBuf = usersOffsetTable.getBlockOfRows(0, 1, usersOffsetBuf);
      itemsOffsetBuf = itemsOffsetTable.getBlockOfRows(0, 1, itemsOffsetBuf);
    } catch (IllegalAccessException e) {
      ErrorHandling.printThrowable(e);
      return;
    }
    int[] usersOffsetData = new int[1];
    int[] itemsOffsetData = new int[1];
    usersOffsetBuf.get(usersOffsetData);
    itemsOffsetBuf.get(itemsOffsetData);
    long usersOffset = usersOffsetData[0];
    long itemsOffset = itemsOffsetData[0];

    System.out.println(" User ID, Item ID, rating");
    for (long i = 0; i < nUsers; i++) {
      for (long j = 0; j < nItems; j++) {
        long userId = i + usersOffset;
        long itemId = j + itemsOffset;
        System.out.println(userId + ", " + itemId + ", " + ratingsData[(int) (i * nItems + j)]);
      }
    }
  }
}
