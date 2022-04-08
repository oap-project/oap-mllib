package com.intel.oneapi.dal.table;


public interface TableIface extends AccessProviderIface{
   long getColumnCount();
   long getRowCount();
   long getKind();
   Common.DataLayout getDataLayout();
   TableMetadata getMetaData();
   long getPullRowsIface();
   long getPullColumnIface();
   long getPullCSRBlockIface();
   boolean hasData();
}
