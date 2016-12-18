package com.szychiewicz.wmh;

import com.szychiewicz.wmh.function.MathFunction;
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

    private CrossValidationIteratorProvider crossValidationIteratorProvider;

    public TestRunner(CrossValidationIteratorProvider crossValidationIteratorProvider) {
        this.crossValidationIteratorProvider = crossValidationIteratorProvider;
    }

    public double run(MultiLayerConfiguration conf, MathFunction fn, INDArray inputs, int numEpochs, int numOfFolds) {
        final MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1000));

        double[] mseArr = new double[numOfFolds + 1];
        KFoldIterator crossValIter = crossValidationIteratorProvider.iterator(inputs, fn, numOfFolds);
        while (crossValIter.hasNext()) {
            DataSet trainData = crossValIter.next();
            for (int i = 0; i < numEpochs; i++) {
                net.fit(trainData);
            }

            log.info("Evaluate model....");

            INDArray testData = crossValIter.testFold().getFeatures();
            INDArray outputs = net.output(testData, false);
            RegressionEvaluation eval = new RegressionEvaluation("Result");
            eval.eval(outputs, fn.getFunctionValues(testData));

            double mse = eval.meanSquaredError(0);
            log.info("MSE for " + crossValIter.cursor() + " fold: " + mse);

            mseArr[crossValIter.cursor()] = mse;
            log.info(eval.stats());
        }

        double finalMse = Util.mean(mseArr);

        log.info("Final MSE: " + finalMse);

        return finalMse;
    }
}
