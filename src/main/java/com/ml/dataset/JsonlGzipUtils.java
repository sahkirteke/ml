package com.ml.dataset;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

public final class JsonlGzipUtils {

    private JsonlGzipUtils() {
    }

    public static List<JsonNode> readJsonl(Path file, ObjectMapper objectMapper) throws IOException {
        List<JsonNode> records = new ArrayList<>();
        if (!Files.exists(file)) {
            return records;
        }
        try (GZIPInputStream gzip = new GZIPInputStream(Files.newInputStream(file));
             BufferedReader reader = new BufferedReader(new InputStreamReader(gzip, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                records.add(objectMapper.readTree(line));
            }
        }
        return records;
    }

    public static void appendJsonl(Path file, List<JsonNode> records, ObjectMapper objectMapper) throws IOException {
        if (records == null || records.isEmpty()) {
            return;
        }
        Files.createDirectories(file.getParent());
        try (GZIPOutputStream gzip = new GZIPOutputStream(Files.newOutputStream(file,
                java.nio.file.StandardOpenOption.CREATE,
                java.nio.file.StandardOpenOption.APPEND));
             BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(gzip, StandardCharsets.UTF_8))) {
            for (JsonNode node : records) {
                writer.write(objectMapper.writeValueAsString(node));
                writer.newLine();
            }
        }
    }
}
