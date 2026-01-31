package com.ml.pred;

import java.util.List;

public class ModelMeta {

    private String symbol;
    private String modelVersion;
    private String featuresVersion;
    private List<String> featureOrder;
    private String imputeStrategy;
    private Double meanRetUp;
    private Double meanRetDown;
    private Long nUp;
    private Long nDown;
    private Double upRate;
    private List<Integer> classes;
    private Integer upClass;
    private Integer upClassIndex;
    private List<String> onnxOutputs;
    private String probOutputName;
    private DecisionPolicy decisionPolicy;

    public String getSymbol() {
        return symbol;
    }

    public void setSymbol(String symbol) {
        this.symbol = symbol;
    }

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

    public Double getUpRate() {
        return upRate;
    }

    public void setUpRate(Double upRate) {
        this.upRate = upRate;
    }

    public List<Integer> getClasses() {
        return classes;
    }

    public void setClasses(List<Integer> classes) {
        this.classes = classes;
    }

    public Integer getUpClass() {
        return upClass;
    }

    public void setUpClass(Integer upClass) {
        this.upClass = upClass;
    }

    public Integer getUpClassIndex() {
        return upClassIndex;
    }

    public void setUpClassIndex(Integer upClassIndex) {
        this.upClassIndex = upClassIndex;
    }

    public List<String> getOnnxOutputs() {
        return onnxOutputs;
    }

    public void setOnnxOutputs(List<String> onnxOutputs) {
        this.onnxOutputs = onnxOutputs;
    }

    public String getProbOutputName() {
        return probOutputName;
    }

    public void setProbOutputName(String probOutputName) {
        this.probOutputName = probOutputName;
    }

    public DecisionPolicy getDecisionPolicy() {
        return decisionPolicy;
    }

    public void setDecisionPolicy(DecisionPolicy decisionPolicy) {
        this.decisionPolicy = decisionPolicy;
    }

    public static class DecisionPolicy {
        private Double minConfidence;
        private Double minAbsExpectedPct;
        private String mode;

        public Double getMinConfidence() {
            return minConfidence;
        }

        public void setMinConfidence(Double minConfidence) {
            this.minConfidence = minConfidence;
        }

        public Double getMinAbsExpectedPct() {
            return minAbsExpectedPct;
        }

        public void setMinAbsExpectedPct(Double minAbsExpectedPct) {
            this.minAbsExpectedPct = minAbsExpectedPct;
        }

        public String getMode() {
            return mode;
        }

        public void setMode(String mode) {
            this.mode = mode;
        }
    }
}
