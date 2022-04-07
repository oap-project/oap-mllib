package com.intel.oneapi.dal.table;

import java.io.Serializable;

abstract class Table implements Serializable {

    public Table() {}

    protected abstract Long getColumnCount();
    protected abstract Long getRowCount();
    protected abstract Common.DataLayout getDataLayout();
    protected abstract boolean hasData();
    protected abstract Long getKind();
    protected abstract TableMetadata getMetaData();
}