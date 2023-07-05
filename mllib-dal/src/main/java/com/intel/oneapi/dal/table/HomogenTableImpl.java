package com.intel.oneapi.dal.table;

import com.intel.oap.mllib.LibLoader;

import java.io.IOException;

public class HomogenTableImpl implements HomogenTableIface {
    static {
        try {
            LibLoader.loadLibraries();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    private long cObject;
    private TableMetadata metadata;
    private Common.ComputeDevice device;

    protected HomogenTableImpl(Common.ComputeDevice computeDevice) {
        super();
        this.device = computeDevice;
        this.cObject = this.cEmptyTableInit();
    }

    public HomogenTableImpl(long cTable) {
        this.cObject = cTable;
    }

    public HomogenTableImpl(long rowCount,
                            long colCount,
                            int[] data,
                            Common.DataLayout dataLayout,
                            Common.ComputeDevice computeDevice) {
        this.device = computeDevice;
        this.cObject = iInit(rowCount, colCount, data, dataLayout.ordinal(), this.device.ordinal());
    }

    public HomogenTableImpl(long rowCount,
                            long colCount,
                            float[] data,
                            Common.DataLayout dataLayout,
                            Common.ComputeDevice computeDevice) {
        this.device = computeDevice;
        this.cObject = fInit(rowCount, colCount, data, dataLayout.ordinal(), this.device.ordinal());

    }

    public HomogenTableImpl(long rowCount,
                            long colCount,
                            long[] data,
                            Common.DataLayout dataLayout,
                            Common.ComputeDevice computeDevice) {
        this.device = computeDevice;
        this.cObject = lInit(rowCount, colCount, data, dataLayout.ordinal(), this.device.ordinal());

    }

    public HomogenTableImpl(long rowCount,
                            long colCount,
                            double[] data,
                            Common.DataLayout dataLayout,
                            Common.ComputeDevice computeDevice) {
        this.device = computeDevice;
        this.cObject = dInit(rowCount, colCount, data, dataLayout.ordinal(), this.device.ordinal());

    }

    public HomogenTableImpl(long rowCount,
                            long colCount,
                            long dataPtr,
                            Common.DataType dataType,
                            Common.DataLayout dataLayout,
                            Common.ComputeDevice computeDevice) {
        this.device = computeDevice;
        switch (dataType) {
            case FLOAT32:
                this.cObject = fPtrInit(rowCount, colCount, dataPtr, dataLayout.ordinal(),
                        this.device.ordinal());
                break;
            case FLOAT64:
                this.cObject = dPtrInit(rowCount, colCount, dataPtr, dataLayout.ordinal(),
                        this.device.ordinal());
                break;
            default:
                System.err.println("oneapi algorithm only support float/double");
                System.exit(-1);
        }
    }

    @Override
    public long getColumnCount() {
        return cGetColumnCount(this.cObject);
    }

    @Override
    public long getRowCount() {
        return cGetRowCount(this.cObject);
    }

    @Override
    public long getKind() {
        return this.cGetKind(this.cObject);
    }

    @Override
    public Common.DataLayout getDataLayout() {

        return Common.DataLayout.get(cGetDataLayout(this.cObject));
    }

    @Override
    public TableMetadata getMetaData() {
        long cMetadata = cGetMetaData(this.cObject);
        this.metadata = new TableMetadata(cMetadata);
        return this.metadata;
    }

    @Override
    public long getPullRowsIface() {
        return 0;
    }

    @Override
    public ColumnAccessor getPullColumnIface() {
        ColumnAccessor accessor = new ColumnAccessor(
                cGetPullColumnIface(this.cObject), this.device);
        return accessor;
    }

    @Override
    public long getPullCSRBlockIface() {
        return 0;
    }

    @Override
    public boolean hasData() {
        return this.getColumnCount() > 0 && this.getRowCount() > 0;
    }

    @Override
    public int[] getIntData() {
        return this.cGetIntData(this.cObject);
    }

    @Override
    public long[] getLongData() {
        return this.cGetLongData(this.cObject);
    }

    @Override
    public float[] getFloatData() {
        return this.cGetFloatData(this.cObject);
    }

    @Override
    public double[] getDoubleData() {
        return cGetDoubleData(this.cObject);
    }

    public long getcObject(){
        return this.cObject;
    }

    public void addHomogenTable(long homogenTableAddr ) {
        this.cObject = cAddHomogenTable(this.cObject, homogenTableAddr, this.device.ordinal());
    }
    private native long iInit(long rowCount,
                              long colCount,
                              int[] data,
                              int  dataLayoutIndex,
                              int computeDeviceIndex);

    private native long fInit(long rowCount,
                              long colCount,
                              float[] data,
                              int dataLayoutIndex,
                              int computeDeviceIndex);

    private native long dInit(long rowCount,
                              long colCount,
                              double[] data,
                              int dataLayoutIndex,
                              int computeDeviceIndex);

    private native long lInit(long rowCount,
                              long colCount,
                              long[] data,
                              int dataLayoutIndex,
                              int computeDeviceIndex);

    private native long dPtrInit(long rowCount,
                                 long colCount,
                                 long dataPtr,
                                 int dataLayoutIndex,
                                 int computeDeviceIndex);

    private native long fPtrInit(long rowCount,
                                 long colCount,
                                 long dataPtr,
                                 int dataLayoutIndex,
                                 int computeDeviceIndex);
    private native long cGetColumnCount(long cObject);
    private native long cGetRowCount(long cObject);
    private native long cGetKind(long cObject);
    private native int cGetDataLayout(long cObject);
    private native long cGetMetaData(long cObject);
    private native long cGetPullRowsIface(long cObject);
    private native long cGetPullColumnIface(long cObject);
    private native long cGetPullCSRBlockIface(long cObject);
    private native int[] cGetIntData(long cObject);
    private native long[] cGetLongData(long cObject);
    private native float[] cGetFloatData(long cObject);
    private native double[] cGetDoubleData(long cObject);
    private native long cAddHomogenTable(long cObject,
                                         long homogenTableAddr,
                                         int computeDeviceIndex);
    private native long cEmptyTableInit();
}
