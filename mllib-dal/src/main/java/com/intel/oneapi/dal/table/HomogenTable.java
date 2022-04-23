package com.intel.oneapi.dal.table;

public class HomogenTable extends Table {
    HomogenTableImpl impl;

    public HomogenTable(long cTable){
        super();
        impl = new HomogenTableImpl(cTable);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        int[] data){
        super();
        // default
        Common.DataLayout dataLayout = Common.DataLayout.ROWMAJOR;
        impl = new HomogenTableImpl(rowCount, colCount, data,
                dataLayout);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        int[] data,
                        int Layoutindex){
        super();
        Common.DataLayout dataLayout = Common.DataLayout.get(Layoutindex);
        impl = new HomogenTableImpl(rowCount, colCount, data,
                dataLayout);

    }

    public HomogenTable(long rowCount,
                        long colCount,
                        long[] data){
        super();
        // default
        Common.DataLayout dataLayout = Common.DataLayout.ROWMAJOR;
        impl = new HomogenTableImpl(rowCount, colCount, (long[])data,
                dataLayout);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        long[] data,
                        int Layoutindex){
        super();
        Common.DataLayout dataLayout = Common.DataLayout.get(Layoutindex);
        impl = new HomogenTableImpl(rowCount, colCount, (long[])data,
                dataLayout);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        float[] data){
        super();
        // default
        Common.DataLayout dataLayout = Common.DataLayout.ROWMAJOR;
        impl = new HomogenTableImpl(rowCount, colCount, (float[])data,
                dataLayout);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        float[] data,
                        int Layoutindex){
        super();
        Common.DataLayout dataLayout = Common.DataLayout.get(Layoutindex);
        impl = new HomogenTableImpl(rowCount, colCount, (float[])data,
                dataLayout);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        double[] data){
        super();
        // default
        Common.DataLayout dataLayout = Common.DataLayout.ROWMAJOR;
        impl = new HomogenTableImpl(rowCount, colCount, (double[])data,
                dataLayout);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        double[] data,
                        int Layoutindex){
        super();
        Common.DataLayout dataLayout = Common.DataLayout.get(Layoutindex);
        impl = new HomogenTableImpl(rowCount, colCount, (double[])data,
                dataLayout);
    }

    @Override
    public Long getColumnCount() {
        return impl.getColumnCount();
    }

    @Override
    public Long getRowCount() {
        return impl.getRowCount();
    }

    @Override
    public Common.DataLayout getDataLayout() {
        return impl.getDataLayout();
    }

    @Override
    public boolean hasData() {
        return impl.hasData();
    }

    @Override
    public Long getKind() {
        return impl.getKind();
    }

    @Override
    public TableMetadata getMetaData() {
        return impl.getMetaData();
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
