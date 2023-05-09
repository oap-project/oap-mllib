package com.intel.oap.mllib.classification;

import java.io.Serializable;

public class LearningNode implements Serializable {
    public int level;
    public double impurity;
    public int splitIndex;
    public double splitValue;
    public boolean isLeaf;
    public double[] probability;
    public int sampleCount ;

    public String toString() {
        String str = String.format("level is %s; " +
                        "impurity is %s; " +
                        "splitIndex is %s; " +
                        "splitValue is %s; " +
                        "isLeaf is %s; " +
                        "probability size is %s;" +
                        "sampleCount is %s;",
                new Integer(level),
                new Double(impurity),
                new Integer(splitIndex),
                new Double(splitValue),
                new Boolean(isLeaf),
                probability == null ? "null" : probability.length,
                new Integer(sampleCount));

        return str;
    }
}
