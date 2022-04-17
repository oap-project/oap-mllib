package com.intel.oneapi.dal.table;

public class Common {
    enum FeatureType {
        NOMINAL, ORDINAL, INTERVAL, RATIO;
        private static final FeatureType[] values = values();
        public static FeatureType get(int ordinal) {
            return values[ordinal];
        }
    }
    enum DataLayout {
        UNKNOW, ROWMAJOR, COLUMNMAJOR;
        private static final DataLayout[] values = values();
        public static DataLayout get(int ordinal) {
            return values[ordinal];
        }
    }
    enum DataType {
        INT8, INT16, INT32, INT64, UINT8,
        UINT16, UINT32, UINT64, FLOAT32, FLOAT64, BFLOAT16;
        private static final DataType[] values = values();
        public static DataType get(int ordinal) {

            return values[ordinal];
        }
    }
}
