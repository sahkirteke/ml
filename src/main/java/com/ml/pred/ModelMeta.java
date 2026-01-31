package com.ml.pred;

import java.util.List;

public class ModelMeta {

    private String modelVersion;
    private String featuresVersion;
    private List<String> featureOrder;
    private String imputeStrategy;

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
}
