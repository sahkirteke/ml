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
@ConfigurationProperties(prefix = "raw")
public class RawIngestionProperties {

    @NotEmpty
    private List<String> symbols = List.of("MANAUSDT", "XRPUSDT");

    @NotBlank
    private String tf = "5m";

    @NotNull
    private Path dataDir = Path.of("data");

    @NotNull
    private ZoneId partitionZone = ZoneId.of("UTC");

    @NotNull
    private BigDecimal eps = new BigDecimal("1e-12");

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

    public BigDecimal getEps() {
        return eps;
    }

    public void setEps(BigDecimal eps) {
        this.eps = eps;
    }
}
