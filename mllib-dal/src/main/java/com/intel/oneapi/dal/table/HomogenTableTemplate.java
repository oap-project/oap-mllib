package com.intel.oneapi.dal.table;

interface HomogenTableTemplate extends HomogenTableIface{
    public long getPullRowsIface();

    public long getPullColumnIface();

    public long getPullCSRBlockIface();
}
