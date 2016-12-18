package com.szychiewicz.wmh.network;

import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.List;

public class NetworkConfigurator {

    private long seed;
    private int numInputs;
    private int numOutputs;

    public NetworkConfigurator(long seed, int numInputs, int numOutputs) {
        this.seed = seed;
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
    }

    public MultiLayerConfiguration getConf(int iterations, double learningRate,
                                           double momentum, String inputActivation,
                                           List<HiddenLayerSetup> layerSetup) {

        if(layerSetup == null || layerSetup.isEmpty()) {
            throw new IllegalArgumentException("LayerSetup cannot be empty.");
        }

        NeuralNetConfiguration.ListBuilder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(learningRate)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(momentum)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(layerSetup.get(0).getHiddenNodes())
                        .activation(inputActivation)
                        .build());

        for (int i = 0; i < layerSetup.size(); i++) {
            HiddenLayerSetup setup = layerSetup.get(i);

            int numIn;
            if (i == 0) {
                numIn = setup.getHiddenNodes();
            } else {
                numIn = layerSetup.get(i - 1).getHiddenNodes();
            }

            builder.layer(i + 1, new DenseLayer.Builder().nIn(numIn).nOut(setup.getHiddenNodes())
                    .activation(setup.getActivation())
                    .build());
        }

        return builder.layer(layerSetup.size() + 1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation("identity")
                .nIn(layerSetup.get(layerSetup.size() - 1).getHiddenNodes()).nOut(numOutputs).build())
                .pretrain(false).backprop(true).build();
    }
}
