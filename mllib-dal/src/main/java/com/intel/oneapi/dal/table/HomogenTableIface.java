package com.intel.oneapi.dal.table;

interface HomogenTableIface extends TableIface{
    public int[] getIntData();
    public long[] getLongData();
    public float[] getFloatData();
    public double[] getDoubleData();

}
