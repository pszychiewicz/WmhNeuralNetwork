package com.szychiewicz.wmh;

import com.szychiewicz.wmh.function.MathFunction;
import com.szychiewicz.wmh.function.XYZFunction;
import com.szychiewicz.wmh.network.HiddenLayerSetup;
import com.szychiewicz.wmh.network.NetworkConfigurator;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Arrays;
import java.util.List;

@Slf4j
public class Main {

    //Random number generator seed, for reproducability
    private static final int SEED = 12345;
    private static final int ITERATIONS = 1;
    private static final int N_EPOCHS = 100;
    private static final int N_SAMPLES = 1000;
    private static final double LEARNING_RATE = 0.01;
    private static final int NUM_INPUTS = 2;
    private static final int NUM_OUTPUTS = 1;
    private static final int NUM_OF_FOLDS = 10;

    private Main() {
    }

    public static void main(final String[] args) {
        final DataLoader dataLoader = new DataLoader();
        final NetworkConfigurator networkConfigurator = new NetworkConfigurator(SEED, NUM_INPUTS, NUM_OUTPUTS);
        TestRunner testRunner = new TestRunner(new CrossValidationIteratorProvider());

        final MathFunction fn = new XYZFunction();
        final INDArray xy = dataLoader.load(N_SAMPLES);

        final List<HiddenLayerSetup> hiddenLayerSetups = Arrays.asList(new HiddenLayerSetup("tanh", 10));
        final MultiLayerConfiguration conf = networkConfigurator.getConf(ITERATIONS, LEARNING_RATE, 0.9, "tanh", hiddenLayerSetups);
        testRunner.run(conf, fn, xy, N_EPOCHS, NUM_OF_FOLDS);
    }



}
