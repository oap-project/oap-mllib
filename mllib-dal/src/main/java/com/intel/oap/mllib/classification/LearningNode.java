package com.intel.oap.mllib.classification;
public class LearningNode {
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
                        "sampleCount is %s;"
                ,level, impurity, splitIndex, splitValue, isLeaf, probability.length, sampleCount);

        return str;
    }
}
