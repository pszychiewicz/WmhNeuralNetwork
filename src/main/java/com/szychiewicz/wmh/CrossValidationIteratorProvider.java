package com.szychiewicz.wmh;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;

public class CrossValidationIteratorProvider {
    public KFoldIterator iterator(final INDArray inputs, final INDArray expectedOutputs, int numOfFolds) {
        final DataSet allData = new DataSet(inputs, expectedOutputs);
        return new KFoldIterator(numOfFolds, allData);
    }
}
