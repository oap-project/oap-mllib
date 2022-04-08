package com.intel.oneapi.dal.table;

public class Common {
    enum FeatureType {
        nominal, ordinal, interval, ratio ;
        private static final FeatureType[] values = values();
        public static FeatureType get(int ordinal) {
            return values[ordinal];
        }
    }
    enum DataLayout {
        unknown, row_major, column_major;
        private static final DataLayout[] values = values();
        public static DataLayout get(int ordinal) {
            System.out.println("enum FeatureType ");
            return values[ordinal];
        }
    }
    enum DataType {
        int8, int16, int32, int64, uint8,
        uint16, uint32, uint64, float32, float64, bfloat16;
        private static final DataType[] values = values();
        public static DataType get(int ordinal) {
            return values[ordinal];
        }
    }
}
