package com.intel.oneapi.dal.table;

public class  EmptyTableImpl extends SerializableImpl implements GenericTableTemplate{
    protected long rowCount;
    protected long colCount;
    protected Common.DataLayout dataLayout;

    protected EmptyTableImpl() {
        super();
        rowCount = 0;
        colCount = 0;
        dataLayout = Common.DataLayout.unknown;
    }

    protected native long cEmptyTableInit(long rowCount,
                                long colCount,
                                String dataLayout);

    @Override
    public long getColumnCount() {
        return 0;
    }

    protected native long cGetColumnCount();

    @Override
    public long getRowCount() {
        return 0;
    }

    protected native long cGetRowCount();

    @Override
    public long getKind() {
        return 0;
    }

    protected native long cGetKind();

    @Override
    public Common.DataLayout getDataLayout() {
        return null;
    }

    protected native long cGetDataLayout();

    @Override
    public TableMetadata getMetaData() {
        return null;
    }

    protected native long cGetMetaData();

    @Override
    public long getPullRowsIface() {
        return 0;
    }

    protected native long cGetPullRowsIface();

    @Override
    public long getPullColumnIface() {
        return 0;
    }

    protected native long cGetPullColumnIface();

    @Override
    public long getPullCSRBlockIface() {
        return 0;
    }

    @Override
    public boolean hasData() {
        return false;
    }

    protected native long cGetPullCSRBlockIface();

    @Override
    public long getAccessIfacehost() {
        return 0;
    }

    protected native long cGetAccessIfacehost();
}
