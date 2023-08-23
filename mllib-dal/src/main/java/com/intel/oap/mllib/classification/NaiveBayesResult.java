package com.intel.oap.mllib.classification;

public class NaiveBayesResult {
    private long piNumericTable;
    private long thetaNumericTable;

    public long getPiNumericTable() {
        return piNumericTable;
    }

    public void setPiNumericTable(long piNumericTable) {
        this.piNumericTable = piNumericTable;
    }

    public long getThetaNumericTable() {
        return thetaNumericTable;
    }

    public void setThetaNumericTable(long thetaNumericTable) {
        this.thetaNumericTable = thetaNumericTable;
    }
}
