package com.intel.oneapi.dal.table;


public interface TableIface {
   long getColumnCount();
   long getRowCount();
   long getKind();
   Common.DataLayout getDataLayout();
   TableMetadata getMetaData();
   RowAccessor getPullRowsIface();
   ColumnAccessor getPullColumnIface();
   CSRAccessor getPullCSRBlockIface();
   boolean hasData();
}
