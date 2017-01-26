package com.szychiewicz.wmh.function;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.Sin;
import org.nd4j.linalg.factory.Nd4j;

public class XYZFunction implements MathFunction {

    @Override
    public INDArray getFunctionValues(INDArray xy) {
        INDArray x = xy.getColumn(0).dup();
        INDArray y = xy.getColumn(1).dup();
        INDArray mult = x.mul(y);
        return Nd4j.getExecutioner().execAndReturn(new Sin(mult.dup())).div(mult.dup()) ;
    }

    @Override
    public String getName() {
        return "sin(x * y)/(x * y)";
    }
}
