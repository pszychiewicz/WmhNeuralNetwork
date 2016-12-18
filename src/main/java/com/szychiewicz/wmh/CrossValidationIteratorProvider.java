package com.szychiewicz.wmh;

import com.szychiewicz.wmh.function.MathFunction;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;

public class CrossValidationIteratorProvider {
    public KFoldIterator iterator(final INDArray x, final MathFunction function, int numOfFolds) {
        final INDArray y = function.getFunctionValues(x);
        final DataSet allData = new DataSet(x, y);
        return new KFoldIterator(numOfFolds, allData);
    }
}
