package com.intel.oneapi.dal.table;

import com.intel.oap.mllib.LibLoader;

import java.io.IOException;

public class HomogenTableImpl extends SerializableImpl implements HomogenTableTemplate {
    static {
        try {
            LibLoader.loadLibraries();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    protected Object jData;
    protected long rowCount;
    protected long colCount;
    protected Common.DataLayout dataLayout;
    protected TableMetadata metadata;

    protected HomogenTableImpl() {
        super();
        this.rowCount = 0;
        this.colCount = 0;
        this.jData = null;
        this.dataLayout = Common.DataLayout.unknown;
    }

    public HomogenTableImpl(long cTable) {
        this.cObject = cTable;
        this.rowCount = 0;
        this.colCount = 0;
        this.jData = null;
        this.dataLayout = Common.DataLayout.unknown;
    }

    public HomogenTableImpl(long rowCount,
                            long colCount,
                            int[] data,
                            Common.DataType dataType,
                            Common.DataLayout dataLayout) {
        initHomogenTable(rowCount, colCount, data, dataType, dataLayout);

    }

    public HomogenTableImpl(long rowCount,
                            long colCount,
                            float[] data,
                            Common.DataType dataType,
                            Common.DataLayout dataLayout) {
        initHomogenTable(rowCount, colCount, data, dataType, dataLayout);

    }

    public HomogenTableImpl(long rowCount,
                            long colCount,
                            long[] data,
                            Common.DataType dataType,
                            Common.DataLayout dataLayout) {
        initHomogenTable(rowCount, colCount, data, dataType, dataLayout);

    }

    public HomogenTableImpl(long rowCount,
                            long colCount,
                            double[] data,
                            Common.DataType dataType,
                            Common.DataLayout dataLayout) {
        initHomogenTable(rowCount, colCount, data, dataType, dataLayout);

    }

    private void initHomogenTable( long rowCount,
                                          long colCount,
                                          Object data,
                                          Common.DataType dataType,
                                          Common.DataLayout dataLayout) {
        System.out.println("initHomogenTable");

        if (dataType.toString() == Common.DataType.int32.toString()) {
            this.cObject = iInit(rowCount, colCount, (int[]) data , dataLayout.ordinal());
        }else if (dataType.toString() == Common.DataType.float32.toString()) {
            this.cObject = fInit(rowCount, colCount, (float[]) data, dataLayout.ordinal());
        }else if (dataType.toString() == Common.DataType.int64.toString()) {
            this.cObject = lInit(rowCount, colCount, (long[]) data, dataLayout.ordinal());
        }else if (dataType.toString() == Common.DataType.float64.toString()) {
            this.cObject = dInit(rowCount, colCount, (double[]) data, dataLayout.ordinal());
        }else {
            throw new IllegalArgumentException("type unsupported");
        }
    }

    private native long iInit(long rowCount,
                                long colCount,
                                int[] data,
                                int  dataLayoutIndex);

    private native long fInit(long rowCount,
                                long colCount,
                                float[] data,
                                int dataLayoutIndex);

    private native long dInit(long rowCount,
                                long colCount,
                                double[] data,
                                int dataLayoutIndex);

    private native long lInit(long rowCount,
                                long colCount,
                                long[] data,
                                int dataLayoutIndex);

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

    @Override
    public boolean hasData() {
        return this.getColumnCount() > 0 && this.getRowCount() > 0;
    }

    private native long cGetPullCSRBlockIface(long cObject);

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
    public long getAccessIfacehost() {
        return 0;
    }

    private native long cGetAccessIfacehost(long cObject);

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
