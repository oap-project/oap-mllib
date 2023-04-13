package com.intel.oneapi.dal.table;

import com.intel.oap.mllib.LibLoader;

import java.io.IOException;

public class CSRTableImpl implements CSRTableIface {
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

    protected CSRTableImpl(Common.ComputeDevice computeDevice) {
        super();
        this.device = computeDevice;
        this.cObject = this.cEmptyCSRTableInit();
    }

    public CSRTableImpl(long cTable) {
        this.cObject = cTable;
    }

    public CSRTableImpl(long rowCount,
                        long colCount,
                        int[] data,
                        long[] columnIndices,
                        long[] rowIndices,
                        Common.CSRIndexing csrIndexing,
                        Common.ComputeDevice computeDevice) {
        this.device = computeDevice;
        this.cObject = iInit(rowCount, colCount,
                data, columnIndices, rowIndices,
                csrIndexing.ordinal(), this.device.ordinal());
    }

    public CSRTableImpl(long rowCount,
                        long colCount,
                        float[] data,
                        long[] columnIndices,
                        long[] rowIndices,
                        Common.CSRIndexing csrIndexing,
                        Common.ComputeDevice computeDevice) {
        this.device = computeDevice;
        this.cObject = fInit(rowCount, colCount,
                data, columnIndices, rowIndices,
                csrIndexing.ordinal(), this.device.ordinal());

    }

    public CSRTableImpl(long rowCount,
                        long colCount,
                        long[] data,
                        long[] columnIndices,
                        long[] rowIndices,
                        Common.CSRIndexing csrIndexing,
                        Common.ComputeDevice computeDevice) {
        this.device = computeDevice;
        this.cObject = lInit(rowCount, colCount,
                data, columnIndices, rowIndices,
                csrIndexing.ordinal(), this.device.ordinal());

    }

    public CSRTableImpl(long rowCount,
                        long colCount,
                        double[] data,
                        long[] columnIndices,
                        long[] rowIndices,
                        Common.CSRIndexing csrIndexing,
                        Common.ComputeDevice computeDevice) {
        this.device = computeDevice;
        this.cObject = dInit(rowCount, colCount,
                data, columnIndices, rowIndices,
                csrIndexing.ordinal(), this.device.ordinal());

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
        return null;
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
    public long[] getColumnIndices() {
        return cGetColumnIndices(this.cObject);
    }

    @Override
    public long[] getRowIndices() {
        return cGetRowIndices(this.cObject);
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

    private native long iInit(long rowCount,
                              long colCount,
                              int[] data,
                              long[] columnIndices,
                              long[] rowIndices,
                              int  csrIndexing,
                              int computeDeviceIndex);

    private native long fInit(long rowCount,
                              long colCount,
                              float[] data,
                              long[] columnIndices,
                              long[] rowIndices,
                              int csrIndexing,
                              int computeDeviceIndex);

    private native long dInit(long rowCount,
                              long colCount,
                              double[] data,
                              long[] columnIndices,
                              long[] rowIndices,
                              int csrIndexing,
                              int computeDeviceIndex);

    private native long lInit(long rowCount,
                              long colCount,
                              long[] data,
                              long[] columnIndices,
                              long[] rowIndices,
                              int csrIndexing,
                              int computeDeviceIndex);

    private native long cEmptyCSRTableInit();
    private native long cGetColumnCount(long cObject);
    private native long cGetRowCount(long cObject);
    private native long cGetKind(long cObject);
    private native int cGetDataLayout(long cObject);
    private native long cGetMetaData(long cObject);
    private native long cGetPullColumnIface(long cObject);
    private native int[] cGetIntData(long cObject);
    private native long[] cGetLongData(long cObject);
    private native float[] cGetFloatData(long cObject);
    private native double[] cGetDoubleData(long cObject);
    private native long[] cGetColumnIndices(long cObject);
    private native long[] cGetRowIndices(long cObject);


}
