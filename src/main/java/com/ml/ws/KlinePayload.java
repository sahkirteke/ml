package com.ml.ws;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;

@JsonIgnoreProperties(ignoreUnknown = true)
public class KlinePayload {

    @JsonProperty("t")
    private Long openTime;

    @JsonProperty("T")
    private Long closeTime;

    @JsonProperty("o")
    private String openPrice;

    @JsonProperty("h")
    private String highPrice;

    @JsonProperty("l")
    private String lowPrice;

    @JsonProperty("c")
    private String closePrice;

    @JsonProperty("v")
    private String volume;

    @JsonProperty("q")
    private String quoteVolume;

    @JsonProperty("n")
    private Long tradeCount;

    @JsonProperty("V")
    private String takerBuyBaseVol;

    @JsonProperty("Q")
    private String takerBuyQuoteVol;

    @JsonProperty("x")
    private Boolean isFinal;

    public Long getOpenTime() {
        return openTime;
    }

    public void setOpenTime(Long openTime) {
        this.openTime = openTime;
    }

    public Long getCloseTime() {
        return closeTime;
    }

    public void setCloseTime(Long closeTime) {
        this.closeTime = closeTime;
    }

    public String getOpenPrice() {
        return openPrice;
    }

    public void setOpenPrice(String openPrice) {
        this.openPrice = openPrice;
    }

    public String getHighPrice() {
        return highPrice;
    }

    public void setHighPrice(String highPrice) {
        this.highPrice = highPrice;
    }

    public String getLowPrice() {
        return lowPrice;
    }

    public void setLowPrice(String lowPrice) {
        this.lowPrice = lowPrice;
    }

    public String getClosePrice() {
        return closePrice;
    }

    public void setClosePrice(String closePrice) {
        this.closePrice = closePrice;
    }

    public String getVolume() {
        return volume;
    }

    public void setVolume(String volume) {
        this.volume = volume;
    }

    public String getQuoteVolume() {
        return quoteVolume;
    }

    public void setQuoteVolume(String quoteVolume) {
        this.quoteVolume = quoteVolume;
    }

    public Long getTradeCount() {
        return tradeCount;
    }

    public void setTradeCount(Long tradeCount) {
        this.tradeCount = tradeCount;
    }

    public String getTakerBuyBaseVol() {
        return takerBuyBaseVol;
    }

    public void setTakerBuyBaseVol(String takerBuyBaseVol) {
        this.takerBuyBaseVol = takerBuyBaseVol;
    }

    public String getTakerBuyQuoteVol() {
        return takerBuyQuoteVol;
    }

    public void setTakerBuyQuoteVol(String takerBuyQuoteVol) {
        this.takerBuyQuoteVol = takerBuyQuoteVol;
    }

    public Boolean getIsFinal() {
        return isFinal;
    }

    public void setIsFinal(Boolean isFinal) {
        this.isFinal = isFinal;
    }
}
