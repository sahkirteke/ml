package com.ml.pred;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({
        "type",
        "symbol",
        "tf",
        "predCloseTimeMs",
        "predCloseTime",
        "predDecision",
        "predPUp",
        "actualCloseTimeMs",
        "actualCloseTime",
        "futureRet_1",
        "actualUp",
        "result",
        "evaluatedAtMs",
        "evaluatedAt"
})
public class EvalRecord {

    private String type;
    private String symbol;
    private String tf;
    private long predCloseTimeMs;
    private String predCloseTime;
    private String predDecision;
    private Double predPUp;
    private long actualCloseTimeMs;
    private String actualCloseTime;
    private Double futureRet_1;
    private Boolean actualUp;
    private String result;
    private long evaluatedAtMs;
    private String evaluatedAt;

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

    public long getPredCloseTimeMs() {
        return predCloseTimeMs;
    }

    public void setPredCloseTimeMs(long predCloseTimeMs) {
        this.predCloseTimeMs = predCloseTimeMs;
    }

    public String getPredCloseTime() {
        return predCloseTime;
    }

    public void setPredCloseTime(String predCloseTime) {
        this.predCloseTime = predCloseTime;
    }

    public String getPredDecision() {
        return predDecision;
    }

    public void setPredDecision(String predDecision) {
        this.predDecision = predDecision;
    }

    public Double getPredPUp() {
        return predPUp;
    }

    public void setPredPUp(Double predPUp) {
        this.predPUp = predPUp;
    }

    public long getActualCloseTimeMs() {
        return actualCloseTimeMs;
    }

    public void setActualCloseTimeMs(long actualCloseTimeMs) {
        this.actualCloseTimeMs = actualCloseTimeMs;
    }

    public String getActualCloseTime() {
        return actualCloseTime;
    }

    public void setActualCloseTime(String actualCloseTime) {
        this.actualCloseTime = actualCloseTime;
    }

    public Double getFutureRet_1() {
        return futureRet_1;
    }

    public void setFutureRet_1(Double futureRet_1) {
        this.futureRet_1 = futureRet_1;
    }

    public Boolean getActualUp() {
        return actualUp;
    }

    public void setActualUp(Boolean actualUp) {
        this.actualUp = actualUp;
    }

    public String getResult() {
        return result;
    }

    public void setResult(String result) {
        this.result = result;
    }

    public long getEvaluatedAtMs() {
        return evaluatedAtMs;
    }

    public void setEvaluatedAtMs(long evaluatedAtMs) {
        this.evaluatedAtMs = evaluatedAtMs;
    }

    public String getEvaluatedAt() {
        return evaluatedAt;
    }

    public void setEvaluatedAt(String evaluatedAt) {
        this.evaluatedAt = evaluatedAt;
    }
}
