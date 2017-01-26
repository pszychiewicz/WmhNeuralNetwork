package com.szychiewicz.wmh;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.random.impl.UniformDistribution;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;


public class NoiseAdder {

    private final Random rng;

    public NoiseAdder(long seed) {
        this.rng = Nd4j.getRandomFactory().getNewRandomInstance(seed);
    }

    public INDArray addNoise(INDArray input, double percent) {
        INDArray ret = Nd4j.createUninitialized(input.shape(), Nd4j.order());
        final INDArray x = Nd4j.getExecutioner().exec(new UniformDistribution(ret), rng);
        INDArray noise = x.mul(percent).sub(percent / 2).add(1);
        return input.mul(noise);
    }
}
