package com.ml.raw;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ml.config.RawIngestionProperties;
import com.ml.features.RollingFeatureState;
import com.ml.features.RollingFeatureStateRegistry;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.zip.GZIPInputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class RawWarmupLoader {

    private static final Logger log = LoggerFactory.getLogger(RawWarmupLoader.class);

    private final ObjectMapper objectMapper;
    private final RawIngestionProperties properties;
    private final RollingFeatureStateRegistry stateRegistry;
    private final SymbolState symbolState;

    public RawWarmupLoader(
            ObjectMapper objectMapper,
            RawIngestionProperties properties,
            RollingFeatureStateRegistry stateRegistry,
            SymbolState symbolState
    ) {
        this.objectMapper = objectMapper;
        this.properties = properties;
        this.stateRegistry = stateRegistry;
        this.symbolState = symbolState;
    }

    public void warmup() {
        for (String symbol : properties.getSymbols()) {
            warmupSymbol(symbol);
        }
    }

    private void warmupSymbol(String symbol) {
        Path symbolDir = properties.getDataDir().resolve("raw").resolve(symbol);
        if (!Files.exists(symbolDir)) {
            log.info("WARMUP_DONE symbol={} loadedBars=0 firstClose=-1 lastClose=-1", symbol);
            return;
        }
        List<Path> files = listRecentFiles(symbolDir, symbol);
        List<RawRecord> records = new ArrayList<>();
        int warmupBars = properties.getWarmupBars();
        for (Path file : files) {
            records.addAll(readFile(file));
            if (records.size() >= warmupBars) {
                break;
            }
        }
        records = records.stream()
                .sorted(Comparator.comparingLong(RawRecord::getCloseTimeMs))
                .collect(Collectors.toList());
        if (records.size() > warmupBars) {
            records = records.subList(records.size() - warmupBars, records.size());
        }
        RollingFeatureState state = stateRegistry.getOrCreate(symbol);
        for (RawRecord record : records) {
            state.add(record);
            symbolState.updateIfNewer(symbol, record.getCloseTimeMs());
        }
        long firstClose = records.isEmpty() ? -1L : records.get(0).getCloseTimeMs();
        long lastClose = records.isEmpty() ? -1L : records.get(records.size() - 1).getCloseTimeMs();
        log.info("WARMUP_DONE symbol={} loadedBars={} firstClose={} lastClose={}",
                symbol,
                records.size(),
                firstClose,
                lastClose);
    }

    private List<Path> listRecentFiles(Path symbolDir, String symbol) {
        String tf = properties.getTf();
        try (Stream<Path> stream = Files.list(symbolDir)) {
            return stream
                    .filter(path -> {
                        String name = path.getFileName().toString();
                        return name.startsWith(symbol + "-" + tf + "-") && name.endsWith(".jsonl.gz");
                    })
                    .sorted(Comparator.comparing(Path::getFileName).reversed())
                    .collect(Collectors.toList());
        } catch (IOException ex) {
            log.warn("WARMUP_LIST_FAIL symbol={} dir={}", symbol, symbolDir, ex);
            return List.of();
        }
    }

    private List<RawRecord> readFile(Path file) {
        List<RawRecord> records = new ArrayList<>();
        try (GZIPInputStream gzip = new GZIPInputStream(Files.newInputStream(file));
             BufferedReader reader = new BufferedReader(new InputStreamReader(gzip, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                if (line.isBlank()) {
                    continue;
                }
                RawRecord record = objectMapper.readValue(line, RawRecord.class);
                records.add(record);
            }
        } catch (Exception ex) {
            log.warn("WARMUP_READ_FAIL file={}", file, ex);
        }
        return records;
    }
}
