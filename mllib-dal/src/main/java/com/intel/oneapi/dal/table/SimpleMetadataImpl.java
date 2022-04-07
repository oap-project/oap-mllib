package com.intel.oneapi.dal.table;

import com.intel.oap.mllib.LibLoader;

import java.io.IOException;
import java.util.ArrayList;

public class SimpleMetadataImpl extends SerializableImpl implements TableMetadataImpl{
    ArrayList dtypes;
    ArrayList ftypes;
    static {
        try {
            LibLoader.loadLibraries();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public SimpleMetadataImpl(long cMetadata) {
        this.cObject = cMetadata;
    }

    public SimpleMetadataImpl(ArrayList dtypes, ArrayList ftypes) throws Exception {
        this.dtypes = dtypes;
        this.ftypes = ftypes;
        if (this.dtypes.size() != this.ftypes.size()) {
            throw new Exception("element_count_in_data_type_and_feature_type_arrays_does_not_match");
        }
    }
    @Override
    public long getFeatureCount() {
        return cGetFeatureCount(this.cObject);
    }

    protected native long cGetFeatureCount(long cObject);

    @Override
    public Common.FeatureType getFeatureType(int index) {
        return Common.FeatureType.get(cGetFeatureType(this.cObject, index));
    }

    protected native int cGetFeatureType(long cObject, int index);

    @Override
    public Common.DataType getDataType(int index) throws Exception {
        return Common.DataType.get(cGetDataType(this.cObject, index));
    }

    protected native int cGetDataType(long cObject, int index);

}
