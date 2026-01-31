package com.ml.raw;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ml.config.RawIngestionProperties;
import com.ml.features.FeatureRecord;
import com.ml.features.LabelRecord;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.GZIPOutputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class GzipJsonlAppender {

    private static final Logger log = LoggerFactory.getLogger(GzipJsonlAppender.class);

    private final ObjectMapper objectMapper;
    private final RawIngestionProperties properties;
    private final DailyPartitionResolver partitionResolver;

    public GzipJsonlAppender(
            ObjectMapper objectMapper,
            RawIngestionProperties properties,
            DailyPartitionResolver partitionResolver
    ) {
        this.objectMapper = objectMapper;
        this.properties = properties;
        this.partitionResolver = partitionResolver;
    }

    public Path append(RawRecord record) throws IOException {
        Path file = appendPayload("raw", record.getSymbol(), record.getCloseTimeMs(), record);
        log.info("RAW_WRITE symbol={} closeTimeMs={} file={}", record.getSymbol(), record.getCloseTimeMs(), file);
        return file;
    }

    public Path appendFeature(FeatureRecord record) throws IOException {
        return appendPayload("features", record.getSymbol(), record.getCloseTimeMs(), record);
    }

    public Path appendLabel(LabelRecord record) throws IOException {
        return appendPayload("labels", record.getSymbol(), record.getCloseTimeMs(), record);
    }

    private Path appendPayload(String category, String symbol, long closeTimeMs, Object payload) throws IOException {
        String normalized = symbol.toUpperCase();
        String date = partitionResolver.resolveDate(closeTimeMs);
        Path symbolDir = properties.getDataDir().resolve(category).resolve(normalized);
        Files.createDirectories(symbolDir);
        String tf = properties.getTf();
        Path file = symbolDir.resolve(normalized + "-" + tf + "-" + date + ".jsonl.gz");
        String json = objectMapper.writeValueAsString(payload);
        try (GZIPOutputStream gzip = new GZIPOutputStream(Files.newOutputStream(file,
                java.nio.file.StandardOpenOption.CREATE,
                java.nio.file.StandardOpenOption.APPEND));
             BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(gzip, StandardCharsets.UTF_8))) {
            writer.write(json);
            writer.newLine();
        }
        return file;
    }
}
