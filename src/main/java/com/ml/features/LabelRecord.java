package com.ml.features;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({
        "symbol",
        "tf",
        "closeTimeMs",
        "labelType",
        "futureRet_1",
        "labelUp"
})
public class LabelRecord {

    private String symbol;
    private String tf;
    private long closeTimeMs;
    private String labelType;
    private Double futureRet_1;
    private Integer labelUp;

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

    public String getLabelType() {
        return labelType;
    }

    public void setLabelType(String labelType) {
        this.labelType = labelType;
    }

    public Double getFutureRet_1() {
        return futureRet_1;
    }

    public void setFutureRet_1(Double futureRet_1) {
        this.futureRet_1 = futureRet_1;
    }

    public Integer getLabelUp() {
        return labelUp;
    }

    public void setLabelUp(Integer labelUp) {
        this.labelUp = labelUp;
    }
}
