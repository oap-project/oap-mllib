package com.intel.oneapi.dal.table;

public interface TableIface {
   long getColumnCount();
   long getRowCount();
   long getKind();
   Common.DataLayout getDataLayout();
   TableMetadata getMetaData();
   long getPullRowsIface();
   ColumnAccessor getPullColumnIface();
   long getPullCSRBlockIface();
   boolean hasData();
}
