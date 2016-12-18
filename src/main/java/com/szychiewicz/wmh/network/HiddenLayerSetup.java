package com.szychiewicz.wmh.network;

public class HiddenLayerSetup {
    private String activation;
    private int hiddenNodes;

    public HiddenLayerSetup(String activation, int hiddenNodes) {
        this.activation = activation;
        this.hiddenNodes = hiddenNodes;
    }

    public String getActivation() {
        return activation;
    }

    public int getHiddenNodes() {
        return hiddenNodes;
    }
}
