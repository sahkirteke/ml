package com.ml.raw;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.ml.config.RawIngestionProperties;
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
        String symbol = record.getSymbol().toUpperCase();
        String date = partitionResolver.resolveDate(record.getCloseTimeMs());
        Path symbolDir = properties.getDataDir().resolve("raw").resolve(symbol);
        Files.createDirectories(symbolDir);
        String tf = properties.getTf();
        Path file = symbolDir.resolve(symbol + "-" + tf + "-" + date + ".jsonl.gz");
        String payload = toJson(record);
        try (GZIPOutputStream gzip = new GZIPOutputStream(Files.newOutputStream(file,
                java.nio.file.StandardOpenOption.CREATE,
                java.nio.file.StandardOpenOption.APPEND));
             BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(gzip, StandardCharsets.UTF_8))) {
            writer.write(payload);
            writer.newLine();
        }
        log.info("RAW_WRITE symbol={} closeTimeMs={} file={}", symbol, record.getCloseTimeMs(), file);
        return file;
    }

    private String toJson(RawRecord record) throws JsonProcessingException {
        return objectMapper.writeValueAsString(record);
    }
}
