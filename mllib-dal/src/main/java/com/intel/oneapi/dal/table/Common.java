package com.intel.oneapi.dal.table;

public class Common {
    public enum FeatureType {
        NOMINAL, ORDINAL, INTERVAL, RATIO;
        private static final FeatureType[] values = values();
        public static FeatureType get(int ordinal) {
            return values[ordinal];
        }
    }
    public enum DataLayout {
        UNKNOWN, ROW_MAJOR, COLUMN_MAJOR;
        private static final DataLayout[] values = values();
        public static DataLayout get(int ordinal) {
            return values[ordinal];
        }

    }
    public enum DataType {
        INT8, INT16, INT32, INT64, UINT8,
        UINT16, UINT32, UINT64, FLOAT32, FLOAT64, BFLOAT16;
        private static final DataType[] values = values();
        public static DataType get(int ordinal) {

            return values[ordinal];
        }
    }
    public enum ComputeDevice {
        HOST, CPU, GPU;
        private static final ComputeDevice[] values = values();
        public static ComputeDevice get(int ordinal) {
            return values[ordinal];
        }
        public static ComputeDevice getDeviceByName(String deviceName){
            ComputeDevice device = null;
            switch (deviceName.toUpperCase()){
                case "HOST":
                    device = HOST;
                    break;
                case "CPU":
                    device = CPU;
                    break;
                case "GPU":
                    device = GPU;
                    break;
            }
            return device;
        }
    }
}
