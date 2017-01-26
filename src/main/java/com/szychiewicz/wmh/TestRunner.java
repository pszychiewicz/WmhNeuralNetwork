package com.szychiewicz.wmh;

import com.google.common.base.Stopwatch;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.KFoldIterator;

@Slf4j
public class TestRunner {

    private final CrossValidationIteratorProvider crossValidationIteratorProvider;

    private MultiLayerNetwork net;
    private final MultiLayerConfiguration conf;


    public TestRunner(MultiLayerConfiguration conf) {
        this.crossValidationIteratorProvider = new CrossValidationIteratorProvider();
        this.conf = conf;
    }

    private MultiLayerNetwork provideNewNetwork() {
        return new MultiLayerNetwork(conf);
    }
    public TestResult run(INDArray inputs, INDArray expectedOutputs, int numEpochs, int numOfFolds) {
        Stopwatch timer = new Stopwatch().start();

        double[] mseArr = new double[numOfFolds + 1];
        KFoldIterator crossValIter = crossValidationIteratorProvider.iterator(inputs, expectedOutputs, numOfFolds);
        log.info("Evaluate model....");
        while (crossValIter.hasNext()) {
            net = provideNewNetwork();
            net.setListeners(new ScoreIterationListener(1000));

            DataSet trainData = crossValIter.next();
            for (int i = 0; i < numEpochs; i++) {
                net.fit(trainData);
            }
            INDArray testData = crossValIter.testFold().getFeatures();
            INDArray expectedData = crossValIter.testFold().getLabels();

            INDArray outputs = net.output(testData, false);
            RegressionEvaluation eval = new RegressionEvaluation("Result");
//            eval.eval(outputs, fn.getFunctionValues(testData));
            eval.eval(outputs, expectedData);

//            log.info(eval.stats());
            double mse = eval.meanSquaredError(0);
//            double mse = eval.rootMeanSquaredError(0);

            mseArr[crossValIter.cursor()] = mse;
//            log.info("MSE for " + crossValIter.cursor() + " fold: " + mse);
        }
        double finalMse = Util.mean(mseArr);
        timer.stop();
        log.info("Final MSE: " + finalMse);
        log.info("Elapsed time: " + timer.toString());
        return new TestResult(finalMse, timer.toString());
    }
}
