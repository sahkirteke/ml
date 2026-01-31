package com.ml.pred;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({
        "type",
        "symbol",
        "tf",
        "closeTimeMs",
        "closeTime",
        "featuresVersion",
        "modelVersion",
        "pUp",
        "confidence",
        "expectedPct",
        "decision",
        "decisionReason",
        "loggedAtMs",
        "loggedAt"
})
public class PredRecord {

    private String type;
    private String symbol;
    private String tf;
    private long closeTimeMs;
    private String closeTime;
    private String featuresVersion;
    private String modelVersion;
    private double pUp;
    private Double confidence;
    private Double expectedPct;
    private String decision;
    private String decisionReason;
    private long loggedAtMs;
    private String loggedAt;

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

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

    public String getCloseTime() {
        return closeTime;
    }

    public void setCloseTime(String closeTime) {
        this.closeTime = closeTime;
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

    public Double getConfidence() {
        return confidence;
    }

    public void setConfidence(Double confidence) {
        this.confidence = confidence;
    }

    public Double getExpectedPct() {
        return expectedPct;
    }

    public void setExpectedPct(Double expectedPct) {
        this.expectedPct = expectedPct;
    }

    public String getDecision() {
        return decision;
    }

    public void setDecision(String decision) {
        this.decision = decision;
    }

    public String getDecisionReason() {
        return decisionReason;
    }

    public void setDecisionReason(String decisionReason) {
        this.decisionReason = decisionReason;
    }

    public long getLoggedAtMs() {
        return loggedAtMs;
    }

    public void setLoggedAtMs(long loggedAtMs) {
        this.loggedAtMs = loggedAtMs;
    }

    public String getLoggedAt() {
        return loggedAt;
    }

    public void setLoggedAt(String loggedAt) {
        this.loggedAt = loggedAt;
    }

}
