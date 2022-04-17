package com.intel.oneapi.dal.table;

public class HomogenTable extends Table {
    HomogenTableImpl impl;

    public HomogenTable(long cTable){
        super();
        impl = new HomogenTableImpl(cTable);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        int[] data,
                        Common.DataLayout dataLayout){
        super();
        impl = new HomogenTableImpl(rowCount, colCount, data, dataLayout);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        float[] data,
                        Common.DataLayout dataLayout){
        super();
        impl = new HomogenTableImpl(rowCount, colCount, data, dataLayout);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        long[] data,
                        Common.DataLayout dataLayout){
        super();
        impl = new HomogenTableImpl(rowCount, colCount, data, dataLayout);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        double[] data,
                        Common.DataLayout dataLayout){
        super();
        impl = new HomogenTableImpl(rowCount, colCount, data, dataLayout);
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

    protected  int[] getIntData() {
        return  impl.getIntData();
    }

    protected  float[] getFloatData() {
        return  impl.getFloatData();
    }

    protected  long[] getLongData() {
        return  impl.getLongData();
    }

    protected  double[] getDoubleData() {
        return  impl.getDoubleData();
    }
}
