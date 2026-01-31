package com.ml.config;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;
import jakarta.validation.constraints.NotNull;
import java.math.BigDecimal;
import java.nio.file.Path;
import java.time.ZoneId;
import java.util.List;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.validation.annotation.Validated;

@Validated
@ConfigurationProperties(prefix = "ml")
public class RawIngestionProperties {

    @NotEmpty
    private List<String> symbols = List.of("MANAUSDT", "XRPUSDT");

    @NotBlank
    private String tf = "5m";

    @NotNull
    private Path dataDir = Path.of("data");

    @NotNull
    private ZoneId partitionZone = ZoneId.of("Europe/Istanbul");

    @NotNull
    private Integer warmupBars = 500;

    @NotBlank
    private String featuresVersion = "ftr_5m_v1";

    @NotNull
    private Path baselineDir = Path.of("data_baseline");

    @NotNull
    private Path trainDir = Path.of("data", "train");

    @NotNull
    private Path modelsBaseDir = Path.of(System.getProperty("user.dir"), "models");

    @NotNull
    private Long expectedGapMs = 300_000L;

    private boolean parityEnabled = false;

    private String parityDate;

    private String paritySymbol;

    @NotNull
    private BigDecimal eps = new BigDecimal("1e-12");

    private DecisionProperties decision = new DecisionProperties();

    private boolean smokeTestEnabled = false;

    public List<String> getSymbols() {
        return symbols;
    }

    public void setSymbols(List<String> symbols) {
        this.symbols = symbols;
    }

    public String getTf() {
        return tf;
    }

    public void setTf(String tf) {
        this.tf = tf;
    }

    public Path getDataDir() {
        return dataDir;
    }

    public void setDataDir(Path dataDir) {
        this.dataDir = dataDir;
    }

    public ZoneId getPartitionZone() {
        return partitionZone;
    }

    public void setPartitionZone(ZoneId partitionZone) {
        this.partitionZone = partitionZone;
    }

    public Integer getWarmupBars() {
        return warmupBars;
    }

    public void setWarmupBars(Integer warmupBars) {
        this.warmupBars = warmupBars;
    }

    public String getFeaturesVersion() {
        return featuresVersion;
    }

    public void setFeaturesVersion(String featuresVersion) {
        this.featuresVersion = featuresVersion;
    }

    public Path getBaselineDir() {
        return baselineDir;
    }

    public void setBaselineDir(Path baselineDir) {
        this.baselineDir = baselineDir;
    }

    public Path getTrainDir() {
        return trainDir;
    }

    public void setTrainDir(Path trainDir) {
        this.trainDir = trainDir;
    }

    public Path getModelsBaseDir() {
        return modelsBaseDir;
    }

    public void setModelsBaseDir(Path modelsBaseDir) {
        this.modelsBaseDir = modelsBaseDir;
    }

    public Long getExpectedGapMs() {
        return expectedGapMs;
    }

    public void setExpectedGapMs(Long expectedGapMs) {
        this.expectedGapMs = expectedGapMs;
    }

    public boolean isParityEnabled() {
        return parityEnabled;
    }

    public void setParityEnabled(boolean parityEnabled) {
        this.parityEnabled = parityEnabled;
    }

    public String getParityDate() {
        return parityDate;
    }

    public void setParityDate(String parityDate) {
        this.parityDate = parityDate;
    }

    public String getParitySymbol() {
        return paritySymbol;
    }

    public void setParitySymbol(String paritySymbol) {
        this.paritySymbol = paritySymbol;
    }

    public BigDecimal getEps() {
        return eps;
    }

    public void setEps(BigDecimal eps) {
        this.eps = eps;
    }

    public DecisionProperties getDecision() {
        return decision;
    }

    public void setDecision(DecisionProperties decision) {
        this.decision = decision;
    }

    public boolean isSmokeTestEnabled() {
        return smokeTestEnabled;
    }

    public void setSmokeTestEnabled(boolean smokeTestEnabled) {
        this.smokeTestEnabled = smokeTestEnabled;
    }

    public static class DecisionProperties {
        private Double minConfidence = 0.55d;
        private Double minAbsExpectedPct = 0.05d;

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
    }
}
