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
        "direction",
        "entryPrice",
        "tpPrice",
        "slPrice",
        "featuresVersion",
        "modelVersion",
        "pUp",
        "confidence",
        "edgeAbs",
        "expectedPct",
        "expectedBp",
        "minConfidence",
        "minAbsExpectedPct",
        "minAbsEdge",
        "decisionReason",
        "failedGate",
        "loggedAtMs",
        "loggedAt"
})
public class PredRecord {

    private String type;
    private String symbol;
    private String tf;
    private long closeTimeMs;
    private String closeTime;
    private String direction;
    private Double entryPrice;
    private Double tpPrice;
    private Double slPrice;
    private String featuresVersion;
    private String modelVersion;
    private double pUp;
    private Double confidence;
    private Double edgeAbs;
    private Double expectedPct;
    private Double expectedBp;
    private Double minConfidence;
    private Double minAbsExpectedPct;
    private Double minAbsEdge;
    private String decisionReason;
    private String failedGate;
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

    public String getDirection() {
        return direction;
    }

    public void setDirection(String direction) {
        this.direction = direction;
    }

    public Double getEntryPrice() {
        return entryPrice;
    }

    public void setEntryPrice(Double entryPrice) {
        this.entryPrice = entryPrice;
    }

    public Double getTpPrice() {
        return tpPrice;
    }

    public void setTpPrice(Double tpPrice) {
        this.tpPrice = tpPrice;
    }

    public Double getSlPrice() {
        return slPrice;
    }

    public void setSlPrice(Double slPrice) {
        this.slPrice = slPrice;
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

    public Double getEdgeAbs() {
        return edgeAbs;
    }

    public void setEdgeAbs(Double edgeAbs) {
        this.edgeAbs = edgeAbs;
    }

    public Double getExpectedPct() {
        return expectedPct;
    }

    public void setExpectedPct(Double expectedPct) {
        this.expectedPct = expectedPct;
    }

    public Double getExpectedBp() {
        return expectedBp;
    }

    public void setExpectedBp(Double expectedBp) {
        this.expectedBp = expectedBp;
    }

    public Double getMinConfidence() {
        return minConfidence;
    }

    public void setMinConfidence(Double minConfidence) {
        this.minConfidence = minConfidence;
    }

    public Double getMinAbsExpectedPct() {
        return minAbsExpectedPct;
    }

    public void setMinAbsExpectedPct(Double minAbsExpectedPct) {
        this.minAbsExpectedPct = minAbsExpectedPct;
    }

    public Double getMinAbsEdge() {
        return minAbsEdge;
    }

    public void setMinAbsEdge(Double minAbsEdge) {
        this.minAbsEdge = minAbsEdge;
    }

    public String getDecisionReason() {
        return decisionReason;
    }

    public void setDecisionReason(String decisionReason) {
        this.decisionReason = decisionReason;
    }

    public String getFailedGate() {
        return failedGate;
    }

    public void setFailedGate(String failedGate) {
        this.failedGate = failedGate;
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
