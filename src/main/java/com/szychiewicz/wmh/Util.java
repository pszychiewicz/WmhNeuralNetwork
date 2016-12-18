package com.szychiewicz.wmh;

public class Util {
    public static double mean(double[] m) {
        double sum = 0;
        for (double aM : m) {
            sum += aM;
        }
        return sum / m.length;
    }
}
