package com.ml.dataset;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.ml.config.RawIngestionProperties;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;

@Component
public class FeatureParityRunner implements ApplicationRunner {

    private static final Logger log = LoggerFactory.getLogger(FeatureParityRunner.class);
    private static final double NUM_TOL = 1e-9;

    private final RawIngestionProperties properties;
    private final ObjectMapper objectMapper;

    public FeatureParityRunner(RawIngestionProperties properties, ObjectMapper objectMapper) {
        this.properties = properties;
        this.objectMapper = objectMapper;
    }

    @Override
    public void run(ApplicationArguments args) throws Exception {
        if (!properties.isParityEnabled()) {
            return;
        }
        Path baseline = properties.getBaselineDir()
                .resolve("features")
                .resolve("MANAUSDT")
                .resolve("MANAUSDT-5m-20260130.jsonl.gz");
        Path generated = properties.getDataDir()
                .resolve("features")
                .resolve("MANAUSDT")
                .resolve("MANAUSDT-5m-20260130.jsonl.gz");
        if (!Files.exists(baseline) || !Files.exists(generated)) {
            throw new IllegalStateException("Parity files missing: baseline=" + baseline + " generated=" + generated);
        }
        compareFiles(baseline, generated);
        log.info("FEATURE_PARITY_OK baseline={} generated={}", baseline, generated);
    }

    private void compareFiles(Path baseline, Path generated) throws Exception {
        List<JsonNode> baselineRecords = JsonlGzipUtils.readJsonl(baseline, objectMapper);
        List<JsonNode> generatedRecords = JsonlGzipUtils.readJsonl(generated, objectMapper);
        Map<Long, JsonNode> baselineMap = indexByCloseTime(baselineRecords);
        Map<Long, JsonNode> generatedMap = indexByCloseTime(generatedRecords);
        Set<Long> baselineKeys = new HashSet<>(baselineMap.keySet());
        Set<Long> generatedKeys = new HashSet<>(generatedMap.keySet());
        if (!baselineKeys.equals(generatedKeys)) {
            Set<Long> missing = new HashSet<>(baselineKeys);
            missing.removeAll(generatedKeys);
            Set<Long> extra = new HashSet<>(generatedKeys);
            extra.removeAll(baselineKeys);
            log.error("FEATURE_PARITY_KEYS missing={} extra={}", missing, extra);
            throw new IllegalStateException("Feature parity key mismatch");
        }
        int diffCount = 0;
        for (Long key : baselineKeys) {
            JsonNode baseNode = baselineMap.get(key);
            JsonNode genNode = generatedMap.get(key);
            List<String> diffs = new ArrayList<>();
            if (!compareNodes("", baseNode, genNode, diffs)) {
                diffCount++;
                log.error("FEATURE_PARITY_DIFF closeTimeMs={} diffs={}", key, diffs);
                if (diffCount >= 5) {
                    throw new IllegalStateException("Feature parity failed (first 5 diffs logged)");
                }
            }
        }
        if (diffCount > 0) {
            throw new IllegalStateException("Feature parity failed");
        }
    }

    private Map<Long, JsonNode> indexByCloseTime(List<JsonNode> nodes) {
        Map<Long, JsonNode> map = new HashMap<>();
        for (JsonNode node : nodes) {
            JsonNode close = node.get("closeTimeMs");
            if (close != null && close.isNumber()) {
                map.put(close.asLong(), node);
            }
        }
        return map;
    }

    private boolean compareNodes(String path, JsonNode base, JsonNode gen, List<String> diffs) {
        if (base == null && gen == null) {
            return true;
        }
        if (base == null || gen == null) {
            diffs.add(path + ": null mismatch");
            return false;
        }
        if (base.isNumber() && gen.isNumber()) {
            double a = base.asDouble();
            double b = gen.asDouble();
            if (Double.isNaN(a) != Double.isNaN(b) || Math.abs(a - b) > NUM_TOL) {
                diffs.add(path + ": " + a + " != " + b);
                return false;
            }
            return true;
        }
        if (base.isTextual() || base.isBoolean() || base.isNull()) {
            if (!base.equals(gen)) {
                diffs.add(path + ": " + base + " != " + gen);
                return false;
            }
            return true;
        }
        if (base.isArray() && gen.isArray()) {
            if (base.size() != gen.size()) {
                diffs.add(path + ": array size " + base.size() + " != " + gen.size());
                return false;
            }
            for (int i = 0; i < base.size(); i++) {
                if (!compareNodes(path + "[" + i + "]", base.get(i), gen.get(i), diffs)) {
                    return false;
                }
            }
            return true;
        }
        if (base.isObject() && gen.isObject()) {
            Set<String> fields = new HashSet<>();
            base.fieldNames().forEachRemaining(fields::add);
            gen.fieldNames().forEachRemaining(fields::add);
            boolean equal = true;
            for (String field : fields) {
                if (!compareNodes(path + "." + field, base.get(field), gen.get(field), diffs)) {
                    equal = false;
                }
            }
            return equal;
        }
        if (!base.equals(gen)) {
            diffs.add(path + ": " + base + " != " + gen);
            return false;
        }
        return true;
    }
}
