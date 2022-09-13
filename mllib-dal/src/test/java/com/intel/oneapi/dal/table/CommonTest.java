package com.intel.oneapi.dal.table;

public class CommonTest {
    public static Common.ComputeDevice getComputeDevice(){
        String device = System.getProperty("computeDevice");
        if(device == null){
            device = "HOST";
        }
        System.out.println("getDevice : " + device);
        return Common.ComputeDevice.getDeviceByName(device);
    }
}
