package com.szychiewicz.wmh.function;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Pow;
import org.nd4j.linalg.factory.Nd4j;

public class SquarePowFunction implements MathFunction {

    @Override
    public INDArray getFunctionValues(INDArray x) {
        return Nd4j.getExecutioner().execAndReturn(new Pow(x.dup(), 2)); // x^2
    }

    @Override
    public String getName() {
        return "x^2";
    }
}
