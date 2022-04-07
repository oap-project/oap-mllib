package com.intel.oneapi.dal.table;


interface TableIface extends AccessProviderIface{

    public long getColumnCount();

    public long getRowCount();

    public long getKind();

    public Common.DataLayout getDataLayout();

    public TableMetadata getMetaData();

    public long getPullRowsIface();

    public long getPullColumnIface();

    public long getPullCSRBlockIface();

    public boolean hasData();
}
