package com.intel.oneapi.dal.table;

interface  GenericTableTemplate extends TableIface{
    abstract public long getPullRowsIface();

    abstract public long getPullColumnIface();

    abstract public long getPullCSRBlockIface();
}
