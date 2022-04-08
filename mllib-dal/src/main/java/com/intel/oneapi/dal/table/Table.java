package com.intel.oneapi.dal.table;

public abstract class Table {
    public Table() {
    }
    protected abstract Long getColumnCount();

    protected abstract Long getRowCount();

    protected abstract Common.DataLayout getDataLayout();

    protected abstract boolean hasData();

    protected abstract Long getKind();

    protected abstract TableMetadata getMetaData();
}
