package com.szychiewicz.wmh;

import com.szychiewicz.wmh.function.MathFunction;
import com.szychiewicz.wmh.function.XYZFunction;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.linalg.api.ndarray.INDArray;

@Slf4j
public class Main {

    //Random number generator seed, for reproducability
    private static final int SEED = 12345;
    private static final int N_SAMPLES = 1000;

    private Main() {
    }

    public static void main(final String[] args) {
        final DataLoader dataLoader = new DataLoader();

        final MathFunction fn = new XYZFunction();
        final INDArray xy = dataLoader.load(N_SAMPLES);
        INDArray expectedOutputs = fn.getFunctionValues(xy);

        NoiseAdder noiseAdder = new NoiseAdder(SEED);
        INDArray expectedOutputsWithNoise = noiseAdder.addNoise(expectedOutputs, 0.1);

        BatchTestRunner batchTestRunner = new BatchTestRunner();
        batchTestRunner.runBatch(xy, expectedOutputsWithNoise);
    }



}


