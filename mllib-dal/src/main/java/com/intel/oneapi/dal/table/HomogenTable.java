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
                        int deviceIndex){
        super();
        // default
        Common.DataLayout dataLayout = Common.DataLayout.ROWMAJOR;
        Common.ComputeDevice device = Common.ComputeDevice.get(deviceIndex);
        impl = new HomogenTableImpl(rowCount, colCount, data,
                dataLayout, device);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        int[] data,
                        int layoutindex,
                        int deviceIndex){
        super();
        Common.DataLayout dataLayout = Common.DataLayout.get(layoutindex);
        Common.ComputeDevice device = Common.ComputeDevice.get(deviceIndex);
        impl = new HomogenTableImpl(rowCount, colCount, data,
                dataLayout, device);

    }

    public HomogenTable(long rowCount,
                        long colCount,
                        long[] data,
                        int deviceIndex){
        super();
        // default
        Common.DataLayout dataLayout = Common.DataLayout.ROWMAJOR;
        Common.ComputeDevice device = Common.ComputeDevice.get(deviceIndex);
        impl = new HomogenTableImpl(rowCount, colCount, data,
                dataLayout, device);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        long[] data,
                        int layoutIndex,
                        int deviceIndex){
        super();
        Common.DataLayout dataLayout = Common.DataLayout.get(layoutIndex);
        Common.ComputeDevice device = Common.ComputeDevice.get(deviceIndex);
        impl = new HomogenTableImpl(rowCount, colCount, data,
                dataLayout, device);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        float[] data,
                        int deviceIndex){
        super();
        // default
        Common.DataLayout dataLayout = Common.DataLayout.ROWMAJOR;
        Common.ComputeDevice device = Common.ComputeDevice.get(deviceIndex);
        impl = new HomogenTableImpl(rowCount, colCount, data,
                dataLayout, device);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        float[] data,
                        int layoutIndex,
                        int deviceIndex){
        super();
        Common.DataLayout dataLayout = Common.DataLayout.get(layoutIndex);
        Common.ComputeDevice device = Common.ComputeDevice.get(deviceIndex);
        impl = new HomogenTableImpl(rowCount, colCount, data,
                dataLayout, device);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        double[] data,
                        int deviceIndex){
        super();
        // default
        Common.DataLayout dataLayout = Common.DataLayout.ROWMAJOR;
        Common.ComputeDevice device = Common.ComputeDevice.get(deviceIndex);
        impl = new HomogenTableImpl(rowCount, colCount, data,
                dataLayout, device);
    }

    public HomogenTable(long rowCount,
                        long colCount,
                        double[] data,
                        int layoutIndex,
                        int deviceIndex){
        super();
        Common.DataLayout dataLayout = Common.DataLayout.get(layoutIndex);
        Common.ComputeDevice device = Common.ComputeDevice.get(deviceIndex);
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
}
