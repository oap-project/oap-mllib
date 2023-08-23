package com.intel.oap.mllib.classification;

public class RandomForestResult {
    private long predictionNumericTable;
    private long probabilitiesNumericTable;
    private long importancesNumericTable;

    public long getProbabilitiesNumericTable() {
        return probabilitiesNumericTable;
    }

    public void setProbabilitiesNumericTable(long probabilitiesNumericTable) {
        this.probabilitiesNumericTable = probabilitiesNumericTable;
    }

    public long getPredictionNumericTable() {
        return predictionNumericTable;
    }

    public void setPredictionNumericTable(long predictionNumericTable) {
        this.predictionNumericTable = predictionNumericTable;
    }

    public long getImportancesNumericTable() {
        return importancesNumericTable;
    }

    public void setImportancesNumericTable(long importancesNumericTable) {
        this.importancesNumericTable = importancesNumericTable;
    }
}
