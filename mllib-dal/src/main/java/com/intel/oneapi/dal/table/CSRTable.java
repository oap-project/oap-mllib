package com.intel.oneapi.dal.table;

public class CSRTable extends Table {
    CSRTableImpl impl;
    public CSRTable(Common.ComputeDevice device){
        super();
        impl = new CSRTableImpl(device);
    }

    public CSRTable(long cTable){
        super();
        impl = new CSRTableImpl(cTable);
    }

    public CSRTable(long rowCount,
                    long colCount,
                    int[] data,
                    long[] columnIndices,
                    long[] rowIndices,
                    Common.ComputeDevice device){
        super();
        // default
        Common.CSRIndexing csrIndexing = Common.CSRIndexing.ONE_BASED;
        impl = new CSRTableImpl(rowCount, colCount, data,
                columnIndices, rowIndices, csrIndexing, device);
    }

    public CSRTable(long rowCount,
                    long colCount,
                    int[] data,
                    long[] columnIndices,
                    long[] rowIndices,
                    Common.CSRIndexing csrIndexing,
                    Common.ComputeDevice device){
        super();
        impl = new CSRTableImpl(rowCount, colCount, data,
                columnIndices, rowIndices, csrIndexing, device);

    }

    public CSRTable(long rowCount,
                    long colCount,
                    long[] data,
                    long[] columnIndices,
                    long[] rowIndices,
                    Common.ComputeDevice device){
        super();
        // default
        Common.CSRIndexing csrIndexing = Common.CSRIndexing.ONE_BASED;
        impl = new CSRTableImpl(rowCount, colCount, data,
                columnIndices, rowIndices, csrIndexing, device);
    }

    public CSRTable(long rowCount,
                    long colCount,
                    long[] data,
                    long[] columnIndices,
                    long[] rowIndices,
                    Common.CSRIndexing csrIndexing,
                    Common.ComputeDevice device){
        super();
        impl = new CSRTableImpl(rowCount, colCount, data,
                columnIndices, rowIndices, csrIndexing, device);
    }

    public CSRTable(long rowCount,
                    long colCount,
                    float[] data,
                    long[] columnIndices,
                    long[] rowIndices,
                    Common.ComputeDevice device){
        super();
        // default
        Common.CSRIndexing csrIndexing = Common.CSRIndexing.ONE_BASED;
        impl = new CSRTableImpl(rowCount, colCount, data,
                columnIndices, rowIndices, csrIndexing, device);
    }

    public CSRTable(long rowCount,
                    long colCount,
                    float[] data,
                    long[] columnIndices,
                    long[] rowIndices,
                    Common.CSRIndexing csrIndexing,
                    Common.ComputeDevice device){
        super();
        impl = new CSRTableImpl(rowCount, colCount, data,
                columnIndices, rowIndices, csrIndexing, device);
    }

    public CSRTable(long rowCount,
                    long colCount,
                    double[] data,
                    long[] columnIndices,
                    long[] rowIndices,
                    Common.ComputeDevice device){
        super();
        // default
        Common.CSRIndexing csrIndexing = Common.CSRIndexing.ONE_BASED;
        impl = new CSRTableImpl(rowCount, colCount, data,
                columnIndices, rowIndices, csrIndexing, device);
    }

    public CSRTable(long rowCount,
                    long colCount,
                    double[] data,
                    long[] columnIndices,
                    long[] rowIndices,
                    Common.CSRIndexing csrIndexing,
                    Common.ComputeDevice device){
        super();
        impl = new CSRTableImpl(rowCount, colCount, data,
                columnIndices, rowIndices, csrIndexing, device);
    }
    @Override
    protected Long getColumnCount() {
        return impl.getColumnCount();
    }

    @Override
    protected Long getRowCount() {
        return impl.getRowCount();
    }

    @Override
    protected Common.DataLayout getDataLayout() {
        return impl.getDataLayout();
    }

    @Override
    protected boolean hasData() {
        return impl.hasData();
    }

    @Override
    protected Long getKind() {
        return impl.getKind();
    }

    @Override
    protected TableMetadata getMetaData() {
        return impl.getMetaData();
    }

    @Override
    public long getcObejct() {
        return impl.getcObject();
    }

    public  int[] getIntData() {
        return  impl.getIntData();
    }

    public  float[] getFloatData() {
        return  impl.getFloatData();
    }

    public  long[] getLongData() {
        return  impl.getLongData();
    }

    public  double[] getDoubleData() {
        return  impl.getDoubleData();
    }
}
