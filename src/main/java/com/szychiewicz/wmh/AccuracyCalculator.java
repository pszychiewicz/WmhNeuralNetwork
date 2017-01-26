package com.szychiewicz.wmh;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Accumulation;
import org.nd4j.linalg.api.ops.impl.accum.Mean;
import org.nd4j.linalg.api.ops.impl.transforms.Abs;
import org.nd4j.linalg.factory.Nd4j;

public class AccuracyCalculator {
    public double calc(INDArray expected, INDArray actual) {
        INDArray error = expected.sub(actual);
        INDArray accuracies = Nd4j.getExecutioner().execAndReturn(new Abs(error.dup())).div(expected);
        Accumulation accumulation = Nd4j.getExecutioner().execAndReturn(new Mean(accuracies));
        Number finalResult = accumulation.getFinalResult();
        return finalResult.doubleValue()*100;
    }
}
