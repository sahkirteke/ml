package com.ml.features;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({
        "symbol",
        "tf",
        "closeTimeMs",
        "closePrice",
        "featuresVersion",
        "windowReady",
        "ret_1",
        "logRet_1",
        "ret_3",
        "ret_12",
        "realizedVol_6",
        "realizedVol_24",
        "rangePct",
        "bodyPct",
        "upperWickPct",
        "lowerWickPct",
        "closePos",
        "volRatio_12",
        "tradeRatio_12",
        "buySellRatio",
        "deltaVolNorm",
        "rsi14",
        "atr14",
        "ema20DistPct",
        "ema200DistPct"
})
public class FeatureRecord {

    private String symbol;
    private String tf;
    private long closeTimeMs;
    private double closePrice;
    private String featuresVersion;
    private boolean windowReady;
    private Double ret_1;
    private Double logRet_1;
    private Double ret_3;
    private Double ret_12;
    private Double realizedVol_6;
    private Double realizedVol_24;
    private Double rangePct;
    private Double bodyPct;
    private Double upperWickPct;
    private Double lowerWickPct;
    private Double closePos;
    private Double volRatio_12;
    private Double tradeRatio_12;
    private Double buySellRatio;
    private Double deltaVolNorm;
    private Double rsi14;
    private Double atr14;
    private Double ema20DistPct;
    private Double ema200DistPct;

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

    public double getClosePrice() {
        return closePrice;
    }

    public void setClosePrice(double closePrice) {
        this.closePrice = closePrice;
    }

    public String getFeaturesVersion() {
        return featuresVersion;
    }

    public void setFeaturesVersion(String featuresVersion) {
        this.featuresVersion = featuresVersion;
    }

    public boolean isWindowReady() {
        return windowReady;
    }

    public void setWindowReady(boolean windowReady) {
        this.windowReady = windowReady;
    }

    public Double getRet_1() {
        return ret_1;
    }

    public void setRet_1(Double ret_1) {
        this.ret_1 = ret_1;
    }

    public Double getLogRet_1() {
        return logRet_1;
    }

    public void setLogRet_1(Double logRet_1) {
        this.logRet_1 = logRet_1;
    }

    public Double getRet_3() {
        return ret_3;
    }

    public void setRet_3(Double ret_3) {
        this.ret_3 = ret_3;
    }

    public Double getRet_12() {
        return ret_12;
    }

    public void setRet_12(Double ret_12) {
        this.ret_12 = ret_12;
    }

    public Double getRealizedVol_6() {
        return realizedVol_6;
    }

    public void setRealizedVol_6(Double realizedVol_6) {
        this.realizedVol_6 = realizedVol_6;
    }

    public Double getRealizedVol_24() {
        return realizedVol_24;
    }

    public void setRealizedVol_24(Double realizedVol_24) {
        this.realizedVol_24 = realizedVol_24;
    }

    public Double getRangePct() {
        return rangePct;
    }

    public void setRangePct(Double rangePct) {
        this.rangePct = rangePct;
    }

    public Double getBodyPct() {
        return bodyPct;
    }

    public void setBodyPct(Double bodyPct) {
        this.bodyPct = bodyPct;
    }

    public Double getUpperWickPct() {
        return upperWickPct;
    }

    public void setUpperWickPct(Double upperWickPct) {
        this.upperWickPct = upperWickPct;
    }

    public Double getLowerWickPct() {
        return lowerWickPct;
    }

    public void setLowerWickPct(Double lowerWickPct) {
        this.lowerWickPct = lowerWickPct;
    }

    public Double getClosePos() {
        return closePos;
    }

    public void setClosePos(Double closePos) {
        this.closePos = closePos;
    }

    public Double getVolRatio_12() {
        return volRatio_12;
    }

    public void setVolRatio_12(Double volRatio_12) {
        this.volRatio_12 = volRatio_12;
    }

    public Double getTradeRatio_12() {
        return tradeRatio_12;
    }

    public void setTradeRatio_12(Double tradeRatio_12) {
        this.tradeRatio_12 = tradeRatio_12;
    }

    public Double getBuySellRatio() {
        return buySellRatio;
    }

    public void setBuySellRatio(Double buySellRatio) {
        this.buySellRatio = buySellRatio;
    }

    public Double getDeltaVolNorm() {
        return deltaVolNorm;
    }

    public void setDeltaVolNorm(Double deltaVolNorm) {
        this.deltaVolNorm = deltaVolNorm;
    }

    public Double getRsi14() {
        return rsi14;
    }

    public void setRsi14(Double rsi14) {
        this.rsi14 = rsi14;
    }

    public Double getAtr14() {
        return atr14;
    }

    public void setAtr14(Double atr14) {
        this.atr14 = atr14;
    }

    public Double getEma20DistPct() {
        return ema20DistPct;
    }

    public void setEma20DistPct(Double ema20DistPct) {
        this.ema20DistPct = ema20DistPct;
    }

    public Double getEma200DistPct() {
        return ema200DistPct;
    }

    public void setEma200DistPct(Double ema200DistPct) {
        this.ema200DistPct = ema200DistPct;
    }
}
