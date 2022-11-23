package com.intel.oneapi.dal.table;

public interface TableMetadataIface {
     long getFeatureCount();
     Common.FeatureType getFeatureType(int index) throws Exception;
     Common.DataType getDataType(int index) throws Exception;
}


