package com.intel.oneapi.dal.table;

public class HomogenTable extends Table {
    HomogenTableImpl impl;

    public HomogenTable(long cTable){
        super();
        impl = new HomogenTableImpl(cTable);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        Object data,
                        Class<? extends Number> cls){
        super();
        // default
        Common.DataLayout dataLayout = Common.DataLayout.ROWMAJOR;
        if (cls == Integer.class) {
            impl = new HomogenTableImpl(rowCount, colCount, (int[])data,
                    dataLayout);
        } else if (cls == Long.class) {
            impl = new HomogenTableImpl(rowCount, colCount, (long[])data,
                    dataLayout);
        } else if (cls == Float.class) {
            impl = new HomogenTableImpl(rowCount, colCount, (float[])data,
                    dataLayout);
        } else if (cls == Double.class) {
            impl = new HomogenTableImpl(rowCount, colCount, (double[])data,
                    dataLayout);
        }
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        Object data,
                        Class<? extends Number> cls,
                        int Layoutindex){
        super();
        Common.DataLayout dataLayout = Common.DataLayout.get(Layoutindex);
        if (cls == Integer.class) {
            impl = new HomogenTableImpl(rowCount, colCount, (int[])data,
                    dataLayout);
        } else if (cls == Long.class) {
            impl = new HomogenTableImpl(rowCount, colCount, (long[])data,
                    dataLayout);
        } else if (cls == Float.class) {
            impl = new HomogenTableImpl(rowCount, colCount, (float[])data,
                    dataLayout);
        } else if (cls == Double.class) {
            impl = new HomogenTableImpl(rowCount, colCount, (double[])data,
                    dataLayout);
        }
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
