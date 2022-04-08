package com.intel.oneapi.dal.table;

public interface HomogenTableTemplate extends HomogenTableIface{
    long getPullRowsIface();
    long getPullColumnIface();
    long getPullCSRBlockIface();
}
