package com.intel.oneapi.dal.table;

import com.intel.oap.mllib.LibLoader;

import java.io.IOException;
import java.util.ArrayList;

public class SimpleMetadataImpl implements TableMetadataImpl{
    private long cObject;
    private ArrayList dtypes;
    private ArrayList ftypes;
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
            throw new Exception("element count and feature type arrays does not match");
        }
    }
    @Override
    public long getFeatureCount() {
        return cGetFeatureCount(this.cObject);
    }

    @Override
    public Common.FeatureType getFeatureType(int index) {
        return Common.FeatureType.get(cGetFeatureType(this.cObject, index));
    }

    @Override
    public Common.DataType getDataType(int index) throws Exception {
        return Common.DataType.get(cGetDataType(this.cObject, index));
    }

    private native int cGetDataType(long cObject, int index);
    private native long cGetFeatureCount(long cObject);
    private native int cGetFeatureType(long cObject, int index);
}
