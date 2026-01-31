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

    public Path append(PredRecord record) throws IOException {
        String symbol = record.getSymbol().toUpperCase();
        String date = partitionResolver.resolveDate(record.getCloseTimeMs());
        Path symbolDir = properties.getDataDir().resolve("pred").resolve(symbol);
        Files.createDirectories(symbolDir);
        Path file = symbolDir.resolve(symbol + "-" + record.getTf() + "-" + date + ".jsonl");
        String json = objectMapper.writeValueAsString(record);
        try (BufferedWriter writer = Files.newBufferedWriter(file, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE,
                StandardOpenOption.APPEND)) {
            writer.write(json);
            writer.newLine();
        }
        log.info("PRED_WRITE symbol={} closeTimeMs={} partition={} decision={}",
                record.getSymbol(),
                record.getCloseTimeMs(),
                date,
                record.getDecision());
        return file;
    }
}
