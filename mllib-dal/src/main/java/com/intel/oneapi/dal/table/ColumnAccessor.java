package com.intel.oneapi.dal.table;

public class ColumnAccessor {
    private long cObject;
    private Common.ComputeDevice cDevice;

    public ColumnAccessor(long cObject) {
        this.cObject = cObject;
        this.cDevice = Common.ComputeDevice.HOST;
    }

    public ColumnAccessor(long cObject, Common.ComputeDevice device) {
        this.cObject = cObject;
        this.cDevice = device;
    }

    public double[] pullDouble(long columnIndex){
        return this.cPullDouble(this.cObject, columnIndex, 0, -1, this.cDevice.ordinal());
    }

    public double[] pullDouble(long columnIndex, long rowStartIndex, long rowEndIndex){
        return this.cPullDouble(this.cObject,
                columnIndex,
                rowStartIndex,
                rowEndIndex,
                this.cDevice.ordinal());
    }

    public float[] pullFloat(long columnIndex) {
        return this.cPullFloat(this.cObject,
                columnIndex,
                0,
                -1,
                this.cDevice.ordinal());
    }

    public float[] pullFloat(long columnIndex, long rowStartIndex, long rowEndIndex) {
        return this.cPullFloat(this.cObject,
                columnIndex,
                rowStartIndex,
                rowEndIndex,
                this.cDevice.ordinal());
    }

    public int[] pullInt(long columnIndex){
        return this.cPullInt(this.cObject, columnIndex, 0, -1, this.cDevice.ordinal());
    }

    public int[] pullInt(long columnIndex, long rowStartIndex, long rowEndIndex){
        return this.cPullInt(this.cObject,
                columnIndex,
                rowStartIndex,
                rowEndIndex,
                this.cDevice.ordinal());
    }

    private native double[] cPullDouble(long cObject,
                                        long cColumnIndex,
                                        long cRowStartIndex,
                                        long cRowEndIndex,
                                        int computeDeviceIndex);
    private native int[] cPullInt(long cObject,
                                  long cColumnIndex,
                                  long cRowStartIndex,
                                  long cRowEndIndex,
                                  int computeDeviceIndex);
    private native float[] cPullFloat(long cObject,
                                      long cColumnIndex,
                                      long cRowStartIndex,
                                      long cRowEndIndex,
                                      int computeDeviceIndex);
}
