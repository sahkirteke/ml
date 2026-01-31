package com.ml.raw;

import com.fasterxml.jackson.annotation.JsonPropertyOrder;

@JsonPropertyOrder({
        "symbol",
        "tf",
        "eventTimeMs",
        "openTimeMs",
        "closeTimeMs",
        "openPrice",
        "highPrice",
        "lowPrice",
        "closePrice",
        "volume",
        "quoteVolume",
        "tradeCount",
        "takerBuyBaseVol",
        "takerBuyQuoteVol",
        "isFinal",
        "receivedAtMs",
        "sellBaseVol",
        "buySellRatio",
        "deltaBaseVol"
})
public class RawRecord {

    private String symbol;
    private String tf;
    private long eventTimeMs;
    private long openTimeMs;
    private long closeTimeMs;
    private String openPrice;
    private String highPrice;
    private String lowPrice;
    private String closePrice;
    private String volume;
    private String quoteVolume;
    private long tradeCount;
    private String takerBuyBaseVol;
    private String takerBuyQuoteVol;
    @com.fasterxml.jackson.annotation.JsonProperty("isFinal")
    private boolean isFinal;
    private long receivedAtMs;
    private String sellBaseVol;
    private String buySellRatio;
    private String deltaBaseVol;

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

    public long getEventTimeMs() {
        return eventTimeMs;
    }

    public void setEventTimeMs(long eventTimeMs) {
        this.eventTimeMs = eventTimeMs;
    }

    public long getOpenTimeMs() {
        return openTimeMs;
    }

    public void setOpenTimeMs(long openTimeMs) {
        this.openTimeMs = openTimeMs;
    }

    public long getCloseTimeMs() {
        return closeTimeMs;
    }

    public void setCloseTimeMs(long closeTimeMs) {
        this.closeTimeMs = closeTimeMs;
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

    public long getTradeCount() {
        return tradeCount;
    }

    public void setTradeCount(long tradeCount) {
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

    public boolean isFinal() {
        return isFinal;
    }

    public void setFinal(boolean aFinal) {
        isFinal = aFinal;
    }

    public long getReceivedAtMs() {
        return receivedAtMs;
    }

    public void setReceivedAtMs(long receivedAtMs) {
        this.receivedAtMs = receivedAtMs;
    }

    public String getSellBaseVol() {
        return sellBaseVol;
    }

    public void setSellBaseVol(String sellBaseVol) {
        this.sellBaseVol = sellBaseVol;
    }

    public String getBuySellRatio() {
        return buySellRatio;
    }

    public void setBuySellRatio(String buySellRatio) {
        this.buySellRatio = buySellRatio;
    }

    public String getDeltaBaseVol() {
        return deltaBaseVol;
    }

    public void setDeltaBaseVol(String deltaBaseVol) {
        this.deltaBaseVol = deltaBaseVol;
    }
}
