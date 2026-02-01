package com.ml.pred;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ml.config.RawIngestionProperties;
import com.ml.raw.DailyPartitionResolver;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class PredWriter {

    private static final Logger log = LoggerFactory.getLogger(PredWriter.class);

    private final ObjectMapper objectMapper;
    private final RawIngestionProperties properties;
    private final DailyPartitionResolver partitionResolver;

    public PredWriter(
            ObjectMapper objectMapper,
            RawIngestionProperties properties,
            DailyPartitionResolver partitionResolver
    ) {
        this.objectMapper = objectMapper;
        this.properties = properties;
        this.partitionResolver = partitionResolver;
    }

    public Path append(Object record) throws IOException {
        String symbol = resolveSymbol(record);
        long partitionTimeMs = resolvePartitionTimeMs(record);
        String date = partitionResolver.resolveDate(partitionTimeMs);
        Path symbolDir = properties.getDataDir().resolve("pred").resolve(symbol);
        Files.createDirectories(symbolDir);
        String tf = resolveTf(record);
        Path file = symbolDir.resolve(symbol + "-" + tf + "-" + date + ".jsonl");
        String json = objectMapper.writeValueAsString(record);
        try (BufferedWriter writer = Files.newBufferedWriter(file, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND)) {
            writer.write(json);
            writer.newLine();
        }
        log.info("PRED_WRITE symbol={} partition={}", symbol, date);
        return file;
    }

    private String resolveSymbol(Object record) {
        if (record instanceof PredRecord pred) {
            return pred.getSymbol().toUpperCase();
        }
        if (record instanceof EvalRecord eval) {
            return eval.getSymbol().toUpperCase();
        }
        return "UNKNOWN";
    }

    private String resolveTf(Object record) {
        if (record instanceof PredRecord pred) {
            return pred.getTf();
        }
        if (record instanceof EvalRecord eval) {
            return eval.getTf();
        }
        return properties.getTf();
    }

    private long resolvePartitionTimeMs(Object record) {
        if (record instanceof PredRecord pred) {
            return pred.getCloseTimeMs();
        }
        if (record instanceof EvalRecord eval) {
            return eval.getEventCloseTimeMs();
        }
        return System.currentTimeMillis();
    }
}
