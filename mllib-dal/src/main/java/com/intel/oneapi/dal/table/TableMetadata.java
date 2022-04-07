package com.intel.oneapi.dal.table;

import com.intel.oap.mllib.LibLoader;

import java.io.IOException;
import java.util.ArrayList;

public class TableMetadata {
    private TableMetadataImpl impl;
    public TableMetadata(long cMetadata) {
        this.impl = new SimpleMetadataImpl(cMetadata);
    }

    public TableMetadata(ArrayList dtype , ArrayList ftype) throws Exception{
        this.impl = new SimpleMetadataImpl(dtype, ftype);
    }

    public long getFeatureCount() {
         return this.impl.getFeatureCount();
    }

    public Common.FeatureType  getFeatureType(int index) throws Exception {
          return this.impl.getFeatureType(index);
    }

    public Common.DataType getDataType(int index) throws Exception {
        System.out.println("TableMetadata");
        System.out.println(this.impl);
        return this.impl.getDataType(index);
    }
}

