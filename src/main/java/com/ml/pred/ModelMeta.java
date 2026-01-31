package com.ml.pred;

import java.util.List;

public class ModelMeta {

    private String modelVersion;
    private String featuresVersion;
    private List<String> featureOrder;
    private String imputeStrategy;
    private Double meanRetUp;
    private Double meanRetDown;
    private Long nUp;
    private Long nDown;

    public String getModelVersion() {
        return modelVersion;
    }

    public void setModelVersion(String modelVersion) {
        this.modelVersion = modelVersion;
    }

    public String getFeaturesVersion() {
        return featuresVersion;
    }

    public void setFeaturesVersion(String featuresVersion) {
        this.featuresVersion = featuresVersion;
    }

    public List<String> getFeatureOrder() {
        return featureOrder;
    }

    public void setFeatureOrder(List<String> featureOrder) {
        this.featureOrder = featureOrder;
    }

    public String getImputeStrategy() {
        return imputeStrategy;
    }

    public void setImputeStrategy(String imputeStrategy) {
        this.imputeStrategy = imputeStrategy;
    }

    public Double getMeanRetUp() {
        return meanRetUp;
    }

    public void setMeanRetUp(Double meanRetUp) {
        this.meanRetUp = meanRetUp;
    }

    public Double getMeanRetDown() {
        return meanRetDown;
    }

    public void setMeanRetDown(Double meanRetDown) {
        this.meanRetDown = meanRetDown;
    }

    public Long getNUp() {
        return nUp;
    }

    public void setNUp(Long nUp) {
        this.nUp = nUp;
    }

    public Long getNDown() {
        return nDown;
    }

    public void setNDown(Long nDown) {
        this.nDown = nDown;
    }
}
