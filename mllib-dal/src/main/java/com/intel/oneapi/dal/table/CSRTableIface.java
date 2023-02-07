package com.intel.oneapi.dal.table;

public interface CSRTableIface extends TableIface {
    long[] getColumnIndices();
    long[] getRowIndices();

    int[] getIntData();
    long[] getLongData();
    float[] getFloatData();
    double[] getDoubleData();
}
