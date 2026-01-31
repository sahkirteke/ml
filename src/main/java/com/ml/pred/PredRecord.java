package com.ml.pred;

import com.fasterxml.jackson.annotation.JsonPropertyOrder;

@JsonPropertyOrder({
        "symbol",
        "tf",
        "closeTimeMs",
        "featuresVersion",
        "modelVersion",
        "pUp",
        "decision",
        "loggedAtMs"
})
public class PredRecord {

    private String symbol;
    private String tf;
    private long closeTimeMs;
    private String featuresVersion;
    private String modelVersion;
    private double pUp;
    private String decision;
    private long loggedAtMs;

    public String getSymbol() {
        return symbol;
    }

    public void setSymbol(String symbol) {
        this.symbol = symbol;
    }

    public String getTf() {
        return tf;
    }

    public void setTf(String tf) {
        this.tf = tf;
    }

    public long getCloseTimeMs() {
        return closeTimeMs;
    }

    public void setCloseTimeMs(long closeTimeMs) {
        this.closeTimeMs = closeTimeMs;
    }

    public String getFeaturesVersion() {
        return featuresVersion;
    }

    public void setFeaturesVersion(String featuresVersion) {
        this.featuresVersion = featuresVersion;
    }

    public String getModelVersion() {
        return modelVersion;
    }

    public void setModelVersion(String modelVersion) {
        this.modelVersion = modelVersion;
    }

    public double getPUp() {
        return pUp;
    }

    public void setPUp(double pUp) {
        this.pUp = pUp;
    }

    public String getDecision() {
        return decision;
    }

    public void setDecision(String decision) {
        this.decision = decision;
    }

    public long getLoggedAtMs() {
        return loggedAtMs;
    }

    public void setLoggedAtMs(long loggedAtMs) {
        this.loggedAtMs = loggedAtMs;
    }
}
