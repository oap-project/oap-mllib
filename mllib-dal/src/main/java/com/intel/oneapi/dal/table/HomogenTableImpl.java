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
    private transient long cObject;
    private TableMetadata metadata;

    protected HomogenTableImpl() {
        super();
    }

    public HomogenTableImpl(long cTable) {
        this.cObject = cTable;
    }

    public HomogenTableImpl(long rowCount,
                            long colCount,
                            int[] data,
                            Common.DataLayout dataLayout,
                            Common.ComputeDevice computeDevice) {
        this.cObject = iInit(rowCount, colCount, data, dataLayout.ordinal(), computeDevice.ordinal());
        System.out.println(" HomogenTableImpl object : " + this.cObject);

    }

    public HomogenTableImpl(long rowCount,
                            long colCount,
                            float[] data,
                            Common.DataLayout dataLayout,
                            Common.ComputeDevice computeDevice) {
        this.cObject = fInit(rowCount, colCount, data, dataLayout.ordinal(), computeDevice.ordinal());

    }

    public HomogenTableImpl(long rowCount,
                            long colCount,
                            long[] data,
                            Common.DataLayout dataLayout,
                            Common.ComputeDevice computeDevice) {
        this.cObject = lInit(rowCount, colCount, data, dataLayout.ordinal(), computeDevice.ordinal());

    }

    public HomogenTableImpl(long rowCount,
                            long colCount,
                            double[] data,
                            Common.DataLayout dataLayout,
                            Common.ComputeDevice computeDevice) {
        this.cObject = dInit(rowCount, colCount, data, dataLayout.ordinal(), computeDevice.ordinal());

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

    @Override
    public long getColumnCount() {
        return cGetColumnCount(this.cObject);
    }

    private native long cGetColumnCount(long cObject);

    @Override
    public long getRowCount() {
        return cGetRowCount(this.cObject);
    }

    private native long cGetRowCount(long cObject);

    @Override
    public long getKind() {
        return this.cGetKind(this.cObject);
    }

    private native long cGetKind(long cObject);

    @Override
    public Common.DataLayout getDataLayout() {
        return Common.DataLayout.get(cGetDataLayout(this.cObject));
    }

    private native int cGetDataLayout(long cObject);

    @Override
    public TableMetadata getMetaData() {
        long cMetadata = cGetMetaData(this.cObject);
        this.metadata = new TableMetadata(cMetadata);
        return this.metadata;
    }

    private native long cGetMetaData(long cObject);

    @Override
    public long getPullRowsIface() {
        return 0;
    }

    private native long cGetPullRowsIface(long cObject);

    @Override
    public long getPullColumnIface() {
        return 0;
    }

    private native long cGetPullColumnIface(long cObject);

    @Override
    public long getPullCSRBlockIface() {
        return 0;
    }

    private native long cGetPullCSRBlockIface(long cObject);

    @Override
    public boolean hasData() {
        return this.getColumnCount() > 0 && this.getRowCount() > 0;
    }


    @Override
    public int[] getIntData() {
        return this.cGetIntData(this.cObject);
    }

    private native int[] cGetIntData(long cObject);

    @Override
    public long[] getLongData() {
        return this.cGetLongData(this.cObject);
    }

    private native long[] cGetLongData(long cObject);

    @Override
    public float[] getFloatData() {
        return this.cGetFloatData(this.cObject);
    }

    private native float[] cGetFloatData(long cObject);


    @Override
    public double[] getDoubleData() {
        return cGetDoubleData(this.cObject);
    }

    private native double[] cGetDoubleData(long cObject);
}
