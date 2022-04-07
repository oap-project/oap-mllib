package com.intel.oneapi.dal.table;

public class EmptyMetadataImpl extends SerializableImpl implements TableMetadataImpl {
    public EmptyMetadataImpl(long cMetadata) {
        this.cObject = cMetadata;
    }

    @Override
    public long getFeatureCount() {
        return cGetFeatureCount(this.cObject);
    }

    protected native long cGetFeatureCount(long cObject);
    @Override
    public Common.FeatureType getFeatureType(int index) throws Exception {
        throw new Exception("cannot get data type from empty metadata");
    }

    protected native String cGetFeatureType(long cObject, int index);

    @Override
    public Common.DataType getDataType(int index) throws Exception {
        throw new Exception("cannot get feature type from  empty metadata");
    }


}
