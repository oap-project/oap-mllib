package com.intel.oneapi.dal.table;

interface TableMetadataImpl {

     public long getFeatureCount();
     public Common.FeatureType getFeatureType(int index) throws Exception;
     public Common.DataType getDataType(int index) throws Exception;
}


