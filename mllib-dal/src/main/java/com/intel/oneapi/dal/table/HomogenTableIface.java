package com.intel.oneapi.dal.table;

public interface HomogenTableIface extends TableIface{
    int[] getIntData();
    long[] getLongData();
    float[] getFloatData();
    double[] getDoubleData();
}
