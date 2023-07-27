/* file: CSRNumericTable.java */
/*******************************************************************************
 * Copyright 2014 Intel Corporation
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
 *******************************************************************************/

/**
 * @ingroup numeric_tables
 * @{
 */
package com.intel.daal.data_management.data;

import com.intel.daal.services.DaalContext;
import com.intel.daal.utils.LibUtils;
import com.intel.oap.mllib.LibLoader;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.LongBuffer;

/**
 * <a name="DAAL-CLASS-DATA_MANAGEMENT__DATA__CSRNUMERICTABLE"></a>
 * @brief Numeric table that provides methods to access data that is stored
 *        in the Compressed Sparse Row(CSR) data layout
 */
public class CSRNumericTable extends NumericTable {

    /** @private */
    static {
        try {
            LibLoader.loadLibraries();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * @brief %Indexing scheme used for accessing the data in CSR layout
     */
    public static final class Indexing {
        private int _value;

        /**
         * Constructs the indexing scheme object using the provided value
         * @param value     Value corresponding to the indexing scheme object
         */
        public Indexing(int value) {
            _value = value;
        }

        /**
         * Returns the value corresponding to the indexing scheme object
         * @return Value corresponding to the indexing scheme object
         */
        public int getValue() {
            return _value;
        }

        private static final int     zeroBasedValue = 0;
        private static final int     oneBasedValue  = 1;
        /**
         *  0-based indexing
         */
        public static final Indexing zeroBased      = new Indexing(zeroBasedValue);
        /**
         *  1-based indexing
         */
        public static final Indexing oneBased       = new Indexing(oneBasedValue);
    }

    public CSRNumericTable(DaalContext context, CSRNumericTableImpl impl) {
        super(context);
        tableImpl = impl;
    }

    /**
     * Constructs homogeneous numeric table from C++ homogeneous numeric
     *        table
     * @param context   Context to manage the ALS algorithm
     * @param cTable    Pointer to C++ numeric table
     */
    public CSRNumericTable(DaalContext context, long cTable) {
        super(context);
        tableImpl = new CSRNumericTableImpl(context, cTable);
    }

    public long getDataSize() {
        return ((CSRNumericTableImpl)tableImpl).getDataSize();
    }

    /**
     * Gets data as an array of longs
     * @return Table data as an array of longs
     */
    public long[] getRowOffsetsArray() {
        return ((CSRNumericTableImpl)tableImpl).getRowOffsetsArray();
    }

    /**
     * Gets data as an array of longs
     * @return Table data as an array of longs
     */
    public long[] getColIndicesArray() {
        return ((CSRNumericTableImpl)tableImpl).getColIndicesArray();
    }

    /**
     * Gets data as an array of doubles
     * @return Table data as an array of double
     */
    public double[] getDoubleArray() {
        return ((CSRNumericTableImpl)tableImpl).getDoubleArray();
    }

    /**
     * Gets data as an array of floats
     * @return Table data as an array of floats
     */
    public float[] getFloatArray() {
        return ((CSRNumericTableImpl)tableImpl).getFloatArray();
    }

    /**
     * Gets data as an array of longs
     * @return Table data as an array of longs
     */
    public long[] getLongArray() {
        return ((CSRNumericTableImpl)tableImpl).getLongArray();
    }

    /**
     * Reads block of rows from the table and returns it to java.nio.DoubleBuffer
     *
     * @param numType       Type of data requested
     * @param vectorIndex   Index of the first row to include into the block
     * @param vectorNum     Number of rows in the block
     * @param buf           Buffer to store non-zero values
     * @param colIndicesBuf Buffer to store indices of the columns containing values from buf
     * @param rowOffsetsBuf Buffer to store row offsets of the values from buf
     *
     * @return Number of rows obtained from the table
     */
    protected long getSparseBlock(DataFeatureUtils.InternalNumType numType, long vectorIndex, long vectorNum,
                                  ByteBuffer buf, LongBuffer colIndicesBuf, LongBuffer rowOffsetsBuf) {
        return ((CSRNumericTableImpl)tableImpl).getSparseBlock(numType, vectorIndex, vectorNum, buf, colIndicesBuf, rowOffsetsBuf);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, DoubleBuffer buf) {
        ((CSRNumericTableImpl)tableImpl).releaseBlockOfRows(vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, FloatBuffer buf) {
        ((CSRNumericTableImpl)tableImpl).releaseBlockOfRows(vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::releaseBlockOfRows(long,long,IntBuffer) */
    @Override
    public void releaseBlockOfRows(long vectorIndex, long vectorNum, IntBuffer buf) {
        ((CSRNumericTableImpl)tableImpl).releaseBlockOfRows(vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public DoubleBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        return ((CSRNumericTableImpl)tableImpl).getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public FloatBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        return ((CSRNumericTableImpl)tableImpl).getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::getBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public IntBuffer getBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        return ((CSRNumericTableImpl)tableImpl).getBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,DoubleBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, DoubleBuffer buf) {
        ((CSRNumericTableImpl)tableImpl).releaseBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,FloatBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, FloatBuffer buf) {
        ((CSRNumericTableImpl)tableImpl).releaseBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

    /** @copydoc NumericTable::releaseBlockOfColumnValues(long,long,long,IntBuffer) */
    @Override
    public void releaseBlockOfColumnValues(long featureIndex, long vectorIndex, long vectorNum, IntBuffer buf) {
        ((CSRNumericTableImpl)tableImpl).releaseBlockOfColumnValues(featureIndex, vectorIndex, vectorNum, buf);
    }

    protected long getSparseBlockSize(long vectorIndex, long vectorNum) {
        return ((CSRNumericTableImpl)tableImpl).getSparseBlockSize(vectorIndex, vectorNum);
    }

    protected long getDoubleSparseBlock(long vectorIndex, long vectorNum, ByteBuffer buf, ByteBuffer colIndicesBuf,
                                        ByteBuffer rowOffsetsBuf) {
        return ((CSRNumericTableImpl)tableImpl).getDoubleSparseBlock(vectorIndex, vectorNum, buf, colIndicesBuf, rowOffsetsBuf);

    }

    protected long getFloatSparseBlock(long vectorIndex, long vectorNum, ByteBuffer buf, ByteBuffer colIndicesBuf,
                                       ByteBuffer rowOffsetsBuf) {
        return ((CSRNumericTableImpl)tableImpl).getFloatSparseBlock(vectorIndex, vectorNum, buf, colIndicesBuf, rowOffsetsBuf);
    }

    protected long getIntSparseBlock(long vectorIndex, long vectorNum, ByteBuffer buf, ByteBuffer colIndicesBuf,
                                     ByteBuffer rowOffsetsBuf) {
        return ((CSRNumericTableImpl)tableImpl).getIntSparseBlock(vectorIndex, vectorNum, buf, colIndicesBuf, rowOffsetsBuf);
    }
}
/** @} */
