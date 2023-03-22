package com.intel.oap.mllib.classification;
public class LearningNode {
    public int level;
    public double impurity;
    public int splitIndex;
    public double splitValue;
    public boolean isLeaf;
    public double[] probability;
    public int sampleCount ;
}
