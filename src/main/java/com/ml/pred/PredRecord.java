package com.ml.pred;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({
        "type",
        "symbol",
        "tf",
        "closeTimeMs",
        "closeTime",
        "horizonBars",
        "tpPct",
        "slPct",
        "direction",
        "entryPrice",
        "tpPrice",
        "slPrice",
        "pHit",
        "pTrade",
        "confidence",
        "loggedAtMs",
        "loggedAt"
})
public class PredRecord {

    private String type;
    private String symbol;
    private String tf;
    private long closeTimeMs;
    private String closeTime;
    private Integer horizonBars;
    private Double tpPct;
    private Double slPct;
    private String direction;
    private Double entryPrice;
    private Double tpPrice;
    private Double slPrice;
    @JsonProperty("pHit")
    private Double pHit;
    @JsonProperty("pTrade")
    private Double pTrade;
    private Double confidence;
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

    public Integer getHorizonBars() {
        return horizonBars;
    }

    public void setHorizonBars(Integer horizonBars) {
        this.horizonBars = horizonBars;
    }

    public Double getTpPct() {
        return tpPct;
    }

    public void setTpPct(Double tpPct) {
        this.tpPct = tpPct;
    }

    public Double getSlPct() {
        return slPct;
    }

    public void setSlPct(Double slPct) {
        this.slPct = slPct;
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

    public Double getPHit() {
        return pHit;
    }

    public void setPHit(Double pHit) {
        this.pHit = pHit;
    }

    public Double getPTrade() {
        return pTrade;
    }

    public void setPTrade(Double pTrade) {
        this.pTrade = pTrade;
    }

    public Double getConfidence() {
        return confidence;
    }

    public void setConfidence(Double confidence) {
        this.confidence = confidence;
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
