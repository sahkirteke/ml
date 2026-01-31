package com.ml.features;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import java.util.List;

@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({
        "symbol",
        "openTimeMs",
        "closeTimeMs",
        "eventTimeMs",
        "featuresVersion",
        "windowReady",
        "x"
})
public class FeatureRecord {

    private String symbol;
    private long openTimeMs;
    private long closeTimeMs;
    private Long eventTimeMs;
    private String featuresVersion;
    private boolean windowReady;
    private List<Double> x;

    public String getSymbol() {
        return symbol;
    }

    public void setSymbol(String symbol) {
        this.symbol = symbol;
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

    public Long getEventTimeMs() {
        return eventTimeMs;
    }

    public void setEventTimeMs(Long eventTimeMs) {
        this.eventTimeMs = eventTimeMs;
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

    public List<Double> getX() {
        return x;
    }

    public void setX(List<Double> x) {
        this.x = x;
    }
}
