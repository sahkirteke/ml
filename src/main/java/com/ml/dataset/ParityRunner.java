package com.ml.dataset;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.ml.config.RawIngestionProperties;
import com.ml.features.FeatureCalculator;
import com.ml.features.FeatureRecord;
import com.ml.features.LabelRecord;
import com.ml.features.RollingFeatureState;
import com.ml.raw.RawRecord;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.stereotype.Component;

@Component
public class ParityRunner implements ApplicationRunner {

    private static final Logger log = LoggerFactory.getLogger(ParityRunner.class);
    private static final DateTimeFormatter BASIC_DATE = DateTimeFormatter.BASIC_ISO_DATE;
    private static final double NUM_TOL = 1e-9;

    private final RawIngestionProperties properties;
    private final ObjectMapper objectMapper;
    private final FeatureCalculator featureCalculator;

    public ParityRunner(
            RawIngestionProperties properties,
            ObjectMapper objectMapper,
            FeatureCalculator featureCalculator
    ) {
        this.properties = properties;
        this.objectMapper = objectMapper;
        this.featureCalculator = featureCalculator;
    }

    @Override
    public void run(ApplicationArguments args) throws Exception {
        if (!properties.isParityEnabled()) {
            return;
        }
        String dateValue = properties.getParityDate();
        if (dateValue == null || dateValue.isBlank()) {
            throw new IllegalStateException("Parity date is required (ml.parity-date=YYYYMMDD)");
        }
        LocalDate date = LocalDate.parse(dateValue, BASIC_DATE);
        List<String> symbols = resolveSymbols();
        for (String symbol : symbols) {
            runParity(symbol, date);
        }
    }

    private List<String> resolveSymbols() {
        String symbol = properties.getParitySymbol();
        if (symbol == null || symbol.isBlank()) {
            return properties.getSymbols();
        }
        return List.of(symbol.toUpperCase());
    }

    private void runParity(String symbol, LocalDate date) throws Exception {
        String tf = properties.getTf();
        String dateStr = BASIC_DATE.format(date);
        Path baselineRaw = properties.getBaselineDir()
                .resolve("raw")
                .resolve(symbol)
                .resolve(symbol + "-" + tf + "-" + dateStr + ".jsonl.gz");
        Path baselineFeatures = properties.getBaselineDir()
                .resolve("features")
                .resolve(symbol)
                .resolve(symbol + "-" + tf + "-" + dateStr + ".jsonl.gz");
        Path baselineLabels = properties.getBaselineDir()
                .resolve("labels")
                .resolve(symbol)
                .resolve(symbol + "-" + tf + "-" + dateStr + ".jsonl.gz");

        if (!Files.exists(baselineRaw)) {
            throw new IllegalStateException("Baseline raw file missing: " + baselineRaw);
        }
        if (!Files.exists(baselineFeatures) || !Files.exists(baselineLabels)) {
            throw new IllegalStateException("Baseline features/labels file missing for " + symbol + " " + dateStr);
        }

        List<RawRecord> warmup = loadWarmup(symbol, date, properties.getWarmupBars());
        RollingFeatureState state = new RollingFeatureState(symbol, Math.max(properties.getWarmupBars(), 1000));
        for (RawRecord record : warmup) {
            state.add(record);
        }

        List<RawRecord> dayRaw = readRawRecords(baselineRaw);
        List<JsonNode> generatedFeatures = new ArrayList<>();
        List<JsonNode> generatedLabels = new ArrayList<>();
        long expectedGap = properties.getExpectedGapMs();
        for (RawRecord record : dayRaw) {
            LabelRecord label = buildLabel(state, record, expectedGap);
            if (label != null) {
                generatedLabels.add(objectMapper.valueToTree(label));
            }
            state.add(record);
            FeatureRecord feature = featureCalculator.calculate(state);
            if (feature != null) {
                generatedFeatures.add(objectMapper.valueToTree(feature));
            }
        }

        Path tempDir = Files.createTempDirectory("parity-" + symbol + "-" + dateStr);
        Path tempFeatures = tempDir.resolve("features").resolve(symbol)
                .resolve(symbol + "-" + tf + "-" + dateStr + ".jsonl.gz");
        Path tempLabels = tempDir.resolve("labels").resolve(symbol)
                .resolve(symbol + "-" + tf + "-" + dateStr + ".jsonl.gz");
        JsonlGzipUtils.appendJsonl(tempFeatures, generatedFeatures, objectMapper);
        JsonlGzipUtils.appendJsonl(tempLabels, generatedLabels, objectMapper);

        compareFiles("FEATURES", baselineFeatures, tempFeatures);
        compareFiles("LABELS", baselineLabels, tempLabels);
        log.info("PARITY_OK symbol={} date={} tempDir={}", symbol, dateStr, tempDir);
    }

    private List<RawRecord> loadWarmup(String symbol, LocalDate date, int warmupBars) {
        Path symbolDir = properties.getBaselineDir().resolve("raw").resolve(symbol);
        if (!Files.exists(symbolDir)) {
            return List.of();
        }
        String tf = properties.getTf();
        List<Path> files;
        try (Stream<Path> stream = Files.list(symbolDir)) {
            files = stream
                    .filter(path -> {
                        String name = path.getFileName().toString();
                        return name.startsWith(symbol + "-" + tf + "-") && name.endsWith(".jsonl.gz");
                    })
                    .sorted(Comparator.comparing(Path::getFileName).reversed())
                    .collect(Collectors.toList());
        } catch (Exception ex) {
            log.warn("PARITY_WARMUP_LIST_FAIL symbol={} dir={}", symbol, symbolDir, ex);
            return List.of();
        }
        List<RawRecord> records = new ArrayList<>();
        for (Path file : files) {
            String name = file.getFileName().toString();
            String fileDate = name.substring((symbol + "-" + tf + "-").length(), name.length() - ".jsonl.gz".length());
            if (fileDate.compareTo(BASIC_DATE.format(date)) > 0) {
                continue;
            }
            records.addAll(readRawRecords(file));
            if (records.size() >= warmupBars) {
                break;
            }
        }
        records.sort(Comparator.comparingLong(RawRecord::getCloseTimeMs));
        if (records.size() > warmupBars) {
            return records.subList(records.size() - warmupBars, records.size());
        }
        return records;
    }

    private List<RawRecord> readRawRecords(Path file) {
        List<RawRecord> records = new ArrayList<>();
        try (GZIPInputStream gzip = new GZIPInputStream(Files.newInputStream(file));
             BufferedReader reader = new BufferedReader(new InputStreamReader(gzip, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                records.add(objectMapper.readValue(line, RawRecord.class));
            }
        } catch (Exception ex) {
            throw new IllegalStateException("Failed reading raw file " + file, ex);
        }
        return records;
    }

    private void compareFiles(String label, Path baseline, Path generated) throws Exception {
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
            log.error("PARITY_KEY_MISMATCH kind={} missing={} extra={}", label, missing, extra);
            throw new IllegalStateException("Parity key mismatch for " + label);
        }
        int diffCount = 0;
        for (Long key : baselineKeys) {
            JsonNode baseNode = baselineMap.get(key);
            JsonNode genNode = generatedMap.get(key);
            List<String> diffs = new ArrayList<>();
            if (!compareNodes("", baseNode, genNode, diffs)) {
                diffCount++;
                log.error("PARITY_DIFF kind={} closeTimeMs={} diffs={}", label, key, diffs);
                if (diffCount >= 5) {
                    throw new IllegalStateException("Parity failed for " + label + " (first 5 diffs logged)");
                }
            }
        }
        if (diffCount > 0) {
            throw new IllegalStateException("Parity failed for " + label);
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

    private LabelRecord buildLabel(RollingFeatureState state, RawRecord current, long expectedGap) {
        RollingFeatureState.Bar previous = state.getLatest();
        if (previous == null) {
            return null;
        }
        double prevClose = previous.getClose();
        if (prevClose == 0.0d) {
            return null;
        }
        double currentClose = parseDouble(current.getClosePrice());
        if (currentClose == 0.0d) {
            return null;
        }
        double futureRet = currentClose / prevClose - 1.0d;
        long gapMs = current.getCloseTimeMs() - previous.getCloseTimeMs();
        LabelRecord label = new LabelRecord();
        label.setSymbol(current.getSymbol());
        label.setCloseTimeMs(previous.getCloseTimeMs());
        label.setFutureCloseTimeMs(current.getCloseTimeMs());
        label.setFutureRet_1(futureRet);
        label.setLabelUp(futureRet > 0.0d);
        label.setExpectedGapMs(expectedGap);
        label.setGapMs(gapMs);
        label.setLabelValid(gapMs == expectedGap);
        return label;
    }

    private double parseDouble(String value) {
        if (value == null || value.isBlank()) {
            return 0.0d;
        }
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException ex) {
            return 0.0d;
        }
    }
}
