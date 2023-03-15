package com.intel.oneapi.dal.table;

public interface TableMetadataImpl {
     long getFeatureCount();
     Common.FeatureType getFeatureType(int index) throws Exception;
     Common.DataType getDataType(int index) throws Exception;
}
