package com.intel.oneapi.dal.table;

public class HomogenTable extends Table {
    HomogenTableImpl impl;

    public HomogenTable(Common.ComputeDevice device){
        super();
        impl = new HomogenTableImpl(device);
    }

    public HomogenTable(long cTable){
        super();
        impl = new HomogenTableImpl(cTable);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        int[] data,
                        Common.ComputeDevice device){
        super();
        // default
        Common.DataLayout dataLayout = Common.DataLayout.ROW_MAJOR;
        impl = new HomogenTableImpl(rowCount, colCount, data,
                dataLayout, device);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        int[] data,
                        Common.DataLayout dataLayout,
                        Common.ComputeDevice device){
        super();
        impl = new HomogenTableImpl(rowCount, colCount, data,
                dataLayout, device);

    }

    public HomogenTable(long rowCount,
                        long colCount,
                        long[] data,
                        Common.ComputeDevice device){
        super();
        // default
        Common.DataLayout dataLayout = Common.DataLayout.ROW_MAJOR;
        impl = new HomogenTableImpl(rowCount, colCount, data,
                dataLayout, device);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        long[] data,
                        Common.DataLayout dataLayout,
                        Common.ComputeDevice device){
        super();
        impl = new HomogenTableImpl(rowCount, colCount, data,
                dataLayout, device);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        float[] data,
                        Common.ComputeDevice device){
        super();
        // default
        Common.DataLayout dataLayout = Common.DataLayout.ROW_MAJOR;
        impl = new HomogenTableImpl(rowCount, colCount, data,
                dataLayout, device);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        float[] data,
                        Common.DataLayout dataLayout,
                        Common.ComputeDevice device){
        super();
        impl = new HomogenTableImpl(rowCount, colCount, data,
                dataLayout, device);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        double[] data,
                        Common.ComputeDevice device){
        super();
        // default
        Common.DataLayout dataLayout = Common.DataLayout.ROW_MAJOR;
        impl = new HomogenTableImpl(rowCount, colCount, data,
                dataLayout, device);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        double[] data,
                        Common.DataLayout dataLayout,
                        Common.ComputeDevice device){
        super();
        impl = new HomogenTableImpl(rowCount, colCount, data,
                dataLayout, device);
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

    public void addHomogenTable(long homogenTableAddr ) {
        impl.addHomogenTable(homogenTableAddr);
    }
    @Override
    public long getcObejct() {
        return impl.getcObject();
    }
}
