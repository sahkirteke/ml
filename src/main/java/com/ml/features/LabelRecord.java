package com.ml.features;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({
        "symbol",
        "closeTimeMs",
        "futureCloseTimeMs",
        "futureRet_1",
        "labelUp",
        "labelValid",
        "gapMs",
        "expectedGapMs"
})
public class LabelRecord {

    private String symbol;
    private long closeTimeMs;
    private long futureCloseTimeMs;
    private double futureRet_1;
    private boolean labelUp;
    private boolean labelValid;
    private long gapMs;
    private long expectedGapMs;

    public String getSymbol() {
        return symbol;
    }

    public void setSymbol(String symbol) {
        this.symbol = symbol;
    }

    public long getCloseTimeMs() {
        return closeTimeMs;
    }

    public void setCloseTimeMs(long closeTimeMs) {
        this.closeTimeMs = closeTimeMs;
    }

    public long getFutureCloseTimeMs() {
        return futureCloseTimeMs;
    }

    public void setFutureCloseTimeMs(long futureCloseTimeMs) {
        this.futureCloseTimeMs = futureCloseTimeMs;
    }

    public double getFutureRet_1() {
        return futureRet_1;
    }

    public void setFutureRet_1(double futureRet_1) {
        this.futureRet_1 = futureRet_1;
    }

    public boolean isLabelUp() {
        return labelUp;
    }

    public void setLabelUp(boolean labelUp) {
        this.labelUp = labelUp;
    }

    public boolean isLabelValid() {
        return labelValid;
    }

    public void setLabelValid(boolean labelValid) {
        this.labelValid = labelValid;
    }

    public long getGapMs() {
        return gapMs;
    }

    public void setGapMs(long gapMs) {
        this.gapMs = gapMs;
    }

    public long getExpectedGapMs() {
        return expectedGapMs;
    }

    public void setExpectedGapMs(long expectedGapMs) {
        this.expectedGapMs = expectedGapMs;
    }
}
