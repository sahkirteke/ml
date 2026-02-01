package com.ml.pred;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({
        "type",
        "symbol",
        "tf",
        "predCloseTimeMs",
        "eventCloseTimeMs",
        "direction",
        "entryPrice",
        "tpPrice",
        "slPrice",
        "result",
        "event",
        "loggedAtMs",
        "loggedAt"
})
public class EvalRecord {

    private String type;
    private String symbol;
    private String tf;
    private long predCloseTimeMs;
    private long eventCloseTimeMs;
    private String direction;
    private Double entryPrice;
    private Double tpPrice;
    private Double slPrice;
    private String result;
    private String event;
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

    public long getPredCloseTimeMs() {
        return predCloseTimeMs;
    }

    public void setPredCloseTimeMs(long predCloseTimeMs) {
        this.predCloseTimeMs = predCloseTimeMs;
    }

    public long getEventCloseTimeMs() {
        return eventCloseTimeMs;
    }

    public void setEventCloseTimeMs(long eventCloseTimeMs) {
        this.eventCloseTimeMs = eventCloseTimeMs;
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

    public String getResult() {
        return result;
    }

    public void setResult(String result) {
        this.result = result;
    }

    public String getEvent() {
        return event;
    }

    public void setEvent(String event) {
        this.event = event;
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
