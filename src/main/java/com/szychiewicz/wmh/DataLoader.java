package com.szychiewicz.wmh;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

public class DataLoader {
    public INDArray load(int numSamples) {
        final INDArray x = Nd4j.linspace(-10, 10, numSamples).reshape(numSamples, 1);
        final INDArray y = Nd4j.linspace(-10, 10, numSamples).reshape(numSamples, 1);
        List<INDArray> indArrays = Arrays.asList(x, y);

        int rows = 2;
        int columns = x.length();
        int[] shape = {rows, columns};
        return Nd4j.create(indArrays, shape).transpose();
    }
}
