package com.szychiewicz.wmh;

public class Garbage {

     /* System.out.println("WITHOUT K FOLD");
        final DataSetIterator iterator = getTrainingData(xy, fn, BATCH_SIZE, RANDOM);
        DataSet next = iterator.next();
        test(net, next, next.getFeatures(), fn);

        System.out.println("ALL DATA");

        DataSet allData = getAllData(xy, fn);
        test(net, allData, allData.getFeatures(), fn);

        System.out.println("WITH BATCHES");
        testWithBatches(net, iterator, allData.getFeatures(), fn);*/

    /*private static void test(MultiLayerNetwork net, DataSet trainData, INDArray testData, MathFunction fn) {
        for (int i = 0; i < N_EPOCHS; i++) {
            net.fit(trainData);
        }

        INDArray outputs = net.output(testData, false);
        RegressionEvaluation eval = new RegressionEvaluation("Result");
        eval.eval(outputs, fn.getFunctionValues(testData));

        log.info(eval.stats());

        double mse = eval.meanSquaredError(0);
        log.info("MSE: " + mse);
    }

    private static void testWithBatches(MultiLayerNetwork net, DataSetIterator iterator, INDArray testData, MathFunction fn) {
        for (int i = 0; i < N_EPOCHS; i++) {
            iterator.reset();
            net.fit(iterator);
        }

        INDArray outputs = net.output(testData, false);
        RegressionEvaluation eval = new RegressionEvaluation("Result");
        eval.eval(outputs, fn.getFunctionValues(testData));

        log.info(eval.stats());

        double mse = eval.meanSquaredError(0);
        log.info("MSE: " + mse);
    }

    *//**
     * Returns the network configuration, 2 hidden DenseLayers of size 50.
     *//*
    private static MultiLayerConfiguration getDeepDenseLayerNetworkConfiguration() {
        final int numHiddenNodes = 5;
        return new NeuralNetConfiguration.Builder()
                .seed(SEED)
                .iterations(ITERATIONS)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .learningRate(LEARNING_RATE)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(NUM_INPUTS).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
                        .activation("tanh")
                        .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation("identity")
                        .nIn(numHiddenNodes).nOut(NUM_OUTPUTS).build())
                .pretrain(false).backprop(true).build();
    }

    private static DataSetIterator getTrainingData(final INDArray x, final MathFunction function, final int batchSize, final Random rng) {
        final INDArray y = function.getFunctionValues(x);
        final DataSet allData = new DataSet(x, y);

        final List<DataSet> list = allData.asList();
        Collections.shuffle(list, rng);

        return new ListDataSetIterator(list, batchSize);
    }

    private static DataSet getAllData(final INDArray x, final MathFunction function) {
        final INDArray y = function.getFunctionValues(x);
        return new DataSet(x, y);
    }*/
}
