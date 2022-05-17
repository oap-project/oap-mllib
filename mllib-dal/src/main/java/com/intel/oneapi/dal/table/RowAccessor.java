package com.intel.oneapi.dal.table;

public class RowAccessor {
    private long cObject;
    private Common.ComputeDevice cDevice;

    public RowAccessor(long cObject, Common.ComputeDevice device) {
        this.cObject = cObject;
        this.cDevice =device;
    }

    public double[] pullDouble(){
        return this.cPullDouble(this.cObject, 0, -1, this.cDevice.ordinal());
    }

    public double[] pullDouble(long rowStartIndex, long rowEndIndex){
        return this.cPullDouble(this.cObject, rowStartIndex, rowEndIndex, this.cDevice.ordinal());
    }

    public float[] pullFloat(){
        return this.cPullFloat(this.cObject, 0, -1, this.cDevice.ordinal());
    }

    public float[] pullFloat(long rowStartIndex, long rowEndIndex){
        return this.cPullFloat(this.cObject, rowStartIndex, rowEndIndex, this.cDevice.ordinal());
    }

    public int[] pullInt(){
        return this.cPullInt(this.cObject, 0, -1, this.cDevice.ordinal());
    }

    public int[] pullInt(long rowStartIndex, long rowEndIndex){
        return this.cPullInt(this.cObject, rowStartIndex, rowEndIndex, this.cDevice.ordinal());
    }

    private native double[] cPullDouble(long cObject, long cRowStartIndex,
                                        long cRowEndIndex, int computeDeviceIndex);
    private native float[] cPullFloat(long cObject, long cRowStartIndex,
                                      long cRowEndIndex, int computeDeviceIndex);
    private native int[] cPullInt(long cObject, long cRowStartIndex,
                                  long cRowEndIndex, int computeDeviceIndex);
}
