package com.intel.oneapi.dal.table;

import java.io.Serializable;

public class SerializableImpl implements Serializable {
    protected transient long cObject;

    public SerializableImpl() {
        this.cObject = 0L;
    }
    protected long serializableId() {
        return cObject;
    }

    protected long getSerializableId() {
        return serializableId();
    }
}
