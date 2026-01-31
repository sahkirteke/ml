package com.ml.ws;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public class WsEnvelope {

    private String stream;
    private KlineEvent data;

    public String getStream() {
        return stream;
    }

    public void setStream(String stream) {
        this.stream = stream;
    }

    public KlineEvent getData() {
        return data;
    }

    public void setData(KlineEvent data) {
        this.data = data;
    }
}
