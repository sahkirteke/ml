package com.ml.dataset;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import com.ml.config.RawIngestionProperties;
import com.ml.raw.DailyPartitionResolver;
import java.nio.file.Path;
import java.time.LocalDate;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
public class TrainJoinJob {

    private static final Logger log = LoggerFactory.getLogger(TrainJoinJob.class);

    private final RawIngestionProperties properties;
    private final DailyPartitionResolver partitionResolver;
    private final ObjectMapper objectMapper;

    public TrainJoinJob(
            RawIngestionProperties properties,
            DailyPartitionResolver partitionResolver,
            ObjectMapper objectMapper
    ) {
        this.properties = properties;
        this.partitionResolver = partitionResolver;
        this.objectMapper = objectMapper;
    }

    @Scheduled(cron = "0 0 1 * * *")
    public void joinYesterday() {
        LocalDate today = LocalDate.now(properties.getPartitionZone());
        LocalDate targetDate = today.minusDays(1);
        String date = partitionResolver.resolveDate(targetDate.atStartOfDay(properties.getPartitionZone())
                .toInstant()
                .toEpochMilli());
        String tf = properties.getTf();
        for (String symbol : properties.getSymbols()) {
            Path featuresFile = resolveDailyFile(properties.getDataDir().resolve("features"), symbol, tf, date);
            Path labelsFile = resolveDailyFile(properties.getDataDir().resolve("labels"), symbol, tf, date);
            if (!featuresFile.toFile().exists() || !labelsFile.toFile().exists()) {
                log.warn("TRAIN_JOIN_SKIP symbol={} date={} featuresExists={} labelsExists={}",
                        symbol,
                        date,
                        featuresFile.toFile().exists(),
                        labelsFile.toFile().exists());
                continue;
            }
            try {
                List<JsonNode> features = JsonlGzipUtils.readJsonl(featuresFile, objectMapper);
                List<JsonNode> labels = JsonlGzipUtils.readJsonl(labelsFile, objectMapper);
                Map<Long, JsonNode> labelByClose = indexByCloseTime(labels);
                List<JsonNode> joined = features.stream()
                        .map(feature -> mergeWithLabel(feature, labelByClose))
                        .filter(node -> node != null)
                        .toList();
                Path trainFile = resolveDailyFile(properties.getTrainDir(), symbol, tf, date);
                JsonlGzipUtils.appendJsonl(trainFile, joined, objectMapper);
                log.info("TRAIN_JOIN_DONE symbol={} date={} joinedCount={}", symbol, date, joined.size());
            } catch (Exception ex) {
                log.error("TRAIN_JOIN_FAIL symbol={} date={}", symbol, date, ex);
            }
        }
    }

    private Path resolveDailyFile(Path baseDir, String symbol, String tf, String date) {
        Path symbolDir = baseDir.resolve(symbol);
        return symbolDir.resolve(symbol + "-" + tf + "-" + date + ".jsonl.gz");
    }

    private Map<Long, JsonNode> indexByCloseTime(List<JsonNode> nodes) {
        Map<Long, JsonNode> map = new HashMap<>();
        for (JsonNode node : nodes) {
            JsonNode closeNode = node.get("closeTimeMs");
            if (closeNode == null || !closeNode.isNumber()) {
                continue;
            }
            map.put(closeNode.asLong(), node);
        }
        return map;
    }

    private JsonNode mergeWithLabel(JsonNode feature, Map<Long, JsonNode> labels) {
        if (feature == null || !feature.isObject()) {
            return null;
        }
        JsonNode closeNode = feature.get("closeTimeMs");
        if (closeNode == null || !closeNode.isNumber()) {
            return null;
        }
        JsonNode label = labels.get(closeNode.asLong());
        if (label == null || !label.isObject()) {
            return null;
        }
        ObjectNode merged = objectMapper.createObjectNode();
        merged.setAll((ObjectNode) feature);
        merged.setAll((ObjectNode) label);
        return merged;
    }
}
