package com.intel.oap.mllib.classification;

import java.util.ArrayList;
import java.util.Map;

public class RandomForestResult {
    public long predictionNumericTable;
    public long probabilitiesNumericTable;
    public Map<Integer, ArrayList<LearningNode>> treesMap;
}
