package com.ml.dataset;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.ml.config.RawIngestionProperties;
import com.ml.raw.DailyPartitionResolver;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.zip.GZIPInputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
public class DatasetCommitJob {

    private static final Logger log = LoggerFactory.getLogger(DatasetCommitJob.class);

    private static final Set<String> RAW_FIELDS = Set.of(
            "symbol",
            "tf",
            "eventTimeMs",
            "openTimeMs",
            "closeTimeMs",
            "openPrice",
            "highPrice",
            "lowPrice",
            "closePrice",
            "volume",
            "quoteVolume",
            "tradeCount",
            "takerBuyBaseVol",
            "takerBuyQuoteVol",
            "isFinal",
            "receivedAtMs",
            "sellBaseVol",
            "buySellRatio",
            "deltaBaseVol"
    );

    private static final Set<String> FEATURE_FIELDS = Set.of(
            "symbol",
            "tf",
            "closeTimeMs",
            "closePrice",
            "featuresVersion",
            "windowReady",
            "ret_1",
            "logRet_1",
            "ret_3",
            "ret_12",
            "realizedVol_6",
            "realizedVol_24",
            "rangePct",
            "bodyPct",
            "upperWickPct",
            "lowerWickPct",
            "closePos",
            "volRatio_12",
            "tradeRatio_12",
            "buySellRatio",
            "deltaVolNorm",
            "rsi14",
            "atr14",
            "ema20DistPct",
            "ema200DistPct"
    );

    private static final Set<String> LABEL_FIELDS = Set.of(
            "symbol",
            "tf",
            "closeTimeMs",
            "labelType",
            "futureRet_1",
            "labelUp"
    );

    private final RawIngestionProperties properties;
    private final DailyPartitionResolver partitionResolver;
    private final ObjectMapper objectMapper;

    public DatasetCommitJob(
            RawIngestionProperties properties,
            DailyPartitionResolver partitionResolver,
            ObjectMapper objectMapper
    ) {
        this.properties = properties;
        this.partitionResolver = partitionResolver;
        this.objectMapper = objectMapper;
    }

    @Scheduled(cron = "0 0 1 * * *", zone = "Europe/Istanbul")
    public void commitYesterday() {
        LocalDate today = LocalDate.now(properties.getPartitionZone());
        LocalDate targetDate = today.minusDays(1);
        String date = partitionResolver.resolveDate(targetDate.atStartOfDay(properties.getPartitionZone())
                .toInstant()
                .toEpochMilli());
        String tf = properties.getTf();
        boolean allOk = true;
        Map<String, ObjectNode> symbolStates = new LinkedHashMap<>();
        for (String symbol : properties.getSymbols()) {
            ObjectNode state = objectMapper.createObjectNode();
            symbolStates.put(symbol, state);
            Path rawFile = resolveDailyFile(properties.getDataDir().resolve("raw"), symbol, tf, date);
            Path featuresFile = resolveDailyFile(properties.getDataDir().resolve("features"), symbol, tf, date);
            Path labelsFile = resolveDailyFile(properties.getDataDir().resolve("labels"), symbol, tf, date);
            FileStats rawStats = analyzeFile(rawFile, RAW_FIELDS, true);
            FileStats featureStats = analyzeFile(featuresFile, FEATURE_FIELDS, false);
            FileStats labelStats = analyzeFile(labelsFile, LABEL_FIELDS, false);
            state.put("rawCount", rawStats.count);
            state.put("featureCount", featureStats.count);
            state.put("labelCount", labelStats.count);
            state.put("lastCloseTimeMs", rawStats.lastCloseTimeMs);
            state.put("gapsDetected", rawStats.gapsDetected);
            boolean ok = true;
            if (!rawStats.exists || rawStats.count == 0) {
                ok = false;
            }
            if (!featureStats.exists || featureStats.count == 0) {
                ok = false;
            }
            if (!labelStats.exists || labelStats.count > featureStats.count) {
                ok = false;
            }
            if (!rawStats.schemaOk || !featureStats.schemaOk || !labelStats.schemaOk) {
                ok = false;
            }
            if (!ok) {
                allOk = false;
            }
        }
        ObjectNode entry = objectMapper.createObjectNode();
        entry.put("date", date);
        entry.set("symbols", objectMapper.valueToTree(symbolStates));
        entry.put("commitAtMs", System.currentTimeMillis());
        try {
            writeDatasetState(entry);
            if (allOk) {
                log.info("COMMIT_OK date={}", date);
            } else {
                log.info("COMMIT_FAIL date={}", date);
            }
        } catch (Exception ex) {
            log.info("COMMIT_FAIL date={}", date, ex);
        }
    }

    private Path resolveDailyFile(Path baseDir, String symbol, String tf, String date) {
        Path symbolDir = baseDir.resolve(symbol);
        return symbolDir.resolve(symbol + "-" + tf + "-" + date + ".jsonl.gz");
    }

    private FileStats analyzeFile(Path file, Set<String> expectedFields, boolean detectGaps) {
        FileStats stats = new FileStats(file);
        if (!stats.exists) {
            return stats;
        }
        long expectedGap = properties.getExpectedGapMs() == null ? 300_000L : properties.getExpectedGapMs();
        long prevClose = -1L;
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new GZIPInputStream(Files.newInputStream(file)),
                StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                JsonNode node = objectMapper.readTree(line);
                if (stats.count == 0) {
                    stats.schemaOk = hasExactFields(node, expectedFields);
                }
                stats.count++;
                JsonNode closeNode = node.get("closeTimeMs");
                if (closeNode != null && closeNode.isNumber()) {
                    long closeTime = closeNode.asLong();
                    if (closeTime > stats.lastCloseTimeMs) {
                        stats.lastCloseTimeMs = closeTime;
                    }
                    if (detectGaps && prevClose > 0 && closeTime - prevClose != expectedGap) {
                        stats.gapsDetected++;
                    }
                    prevClose = closeTime;
                }
            }
        } catch (Exception ex) {
            stats.schemaOk = false;
        }
        return stats;
    }

    private boolean hasExactFields(JsonNode node, Set<String> expectedFields) {
        if (node == null || !node.isObject()) {
            return false;
        }
        Set<String> fieldNames = new java.util.HashSet<>();
        node.fieldNames().forEachRemaining(fieldNames::add);
        return fieldNames.equals(expectedFields);
    }

    private void writeDatasetState(ObjectNode entry) throws IOException {
        Path metaDir = properties.getDataDir().resolve("_meta");
        Files.createDirectories(metaDir);
        Path stateFile = metaDir.resolve("dataset_state.json");
        ArrayNode entries;
        if (Files.exists(stateFile)) {
            JsonNode existing = objectMapper.readTree(stateFile.toFile());
            if (existing != null && existing.isArray()) {
                entries = (ArrayNode) existing;
            } else {
                entries = objectMapper.createArrayNode();
                if (existing != null) {
                    entries.add(existing);
                }
            }
        } else {
            entries = objectMapper.createArrayNode();
        }
        entries.add(entry);
        objectMapper.writerWithDefaultPrettyPrinter().writeValue(stateFile.toFile(), entries);
    }

    private static final class FileStats {
        private final boolean exists;
        private long count;
        private long lastCloseTimeMs = -1L;
        private long gapsDetected;
        private boolean schemaOk = true;

        private FileStats(Path file) {
            this.exists = Files.exists(file);
        }
    }
}
