package com.szychiewicz.wmh;

public class TestResult {
    private final double mse;
    private final String time;

    public TestResult(double mse, String time) {
        this.mse = mse;
        this.time = time;
    }

    public double getMse() {
        return mse;
    }

    public String getTime() {
        return time;
    }
}
