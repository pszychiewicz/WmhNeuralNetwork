package com.szychiewicz.wmh;

import com.szychiewicz.wmh.network.HiddenLayerSetup;
import com.szychiewicz.wmh.network.NetworkConfigurator;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.output.TeeOutputStream;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

@Slf4j
public class BatchTestRunner {

//    public static final double[] LEARNING_RATES = {0.001, 0.01, 0.1};
    public static final double[] LEARNING_RATES = {0.1};
//    public static final double[] MOMENTUMS = {0.3, 0.6, 1};
    public static final double[] MOMENTUMS = {1};
//    public static final int[] LAYER_NUMS = {1, 2, 4, 8, 10};
    public static final int[] LAYER_NUMS = {1};
//    public static final int[] NEURON_NUMS = {5, 10, 20, 40};
    public static final int[] NEURON_NUMS = {14, 20};
    private static final int SEED = 12345;
    private static final int N_EPOCHS = 100;
    private static final int NUM_INPUTS = 2;
    private static final int NUM_OUTPUTS = 1;
    private static final int NUM_OF_FOLDS = 10;
    private static final String ACTIVATION = "tanh";
    private final NetworkConfigurator networkConfigurator = new NetworkConfigurator(SEED, NUM_INPUTS, NUM_OUTPUTS);

    public void runBatch(INDArray inputs, INDArray expectedOutputs) {
        List<BatchResult> batchResults = new ArrayList<>();
        for (double learningRate : LEARNING_RATES) {
            for (double momentum : MOMENTUMS) {
                for (int layerNum : LAYER_NUMS) {
                    for (int neuronNum : NEURON_NUMS) {
                        MultiLayerConfiguration config = buildConfig(layerNum, neuronNum, learningRate, momentum);
                        TestRunner testRunner = new TestRunner(config);
                        TestResult result = testRunner.run(inputs, expectedOutputs, N_EPOCHS, NUM_OF_FOLDS);
                        batchResults.add(new BatchResult(learningRate, momentum, layerNum, neuronNum, result.getMse(),
                                result.getTime()));
                    }
                }
            }
        }
        batchResults.sort(BatchResult::compareTo);
        printBatchResults(batchResults);
    }

    private void printBatchResults(List<BatchResult> batchResults) {
        File f = new File("WMH-" + UUID.randomUUID());
        try (FileOutputStream fos = new FileOutputStream(f)) {
            //we will want to print in standard "System.out" and in "file"
            TeeOutputStream myOut = new TeeOutputStream(System.out, fos);
            PrintStream ps = new PrintStream(myOut);
            System.setOut(ps);
            batchResults.forEach(r -> System.out.println(r.mse + "," + r.learningRate + "," + r.momentum + ","
                    + r.layerNum + "," + r.neuronNum + "," + r.time));

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private MultiLayerConfiguration buildConfig(int layerNum, int neuronNum, double learningRate, double momentum) {
        List<HiddenLayerSetup> layers = IntStream.range(0, layerNum)
                .mapToObj(i -> new HiddenLayerSetup(ACTIVATION, neuronNum))
                .collect(Collectors.toList());
        return networkConfigurator.getConf(1, learningRate, momentum, ACTIVATION, layers);
    }

    static class BatchResult implements Comparable<BatchResult> {
        double learningRate;
        double momentum;
        int layerNum;
        int neuronNum;
        double mse;
        String time;

        BatchResult(double learningRate, double momentum, int layerNum, int neuronNum, double mse, String time) {
            this.learningRate = learningRate;
            this.momentum = momentum;
            this.layerNum = layerNum;
            this.neuronNum = neuronNum;
            this.mse = mse;
            this.time = time;
        }

        @Override
        public int compareTo(BatchResult batchResult) {
            if (this.mse < batchResult.mse) {
                return -1;
            } else if (this.mse == batchResult.mse) {
                return 0;
            } else {
                return 1;
            }
        }
    }
}
