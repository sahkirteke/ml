package com.ml.pred;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ml.config.RawIngestionProperties;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

@Component
public class OnnxModelLoader {

    private static final Logger log = LoggerFactory.getLogger(OnnxModelLoader.class);

    private final Path modelsBaseDir;
    private final ObjectMapper objectMapper;
    private final OrtEnvironment environment;
    private final Map<String, ModelBundle> models = new ConcurrentHashMap<>();

    public OnnxModelLoader(RawIngestionProperties properties, ObjectMapper objectMapper) {
        this.modelsBaseDir = properties.getModelsBaseDir().toAbsolutePath().normalize();
        this.objectMapper = objectMapper;
        this.environment = OrtEnvironment.getEnvironment();
        log.info("MODELS_BASE_DIR path={}", modelsBaseDir);
    }

    public Optional<ModelBundle> getModel(String symbol) {
        if (symbol == null || symbol.isBlank()) {
            return Optional.empty();
        }
        String key = symbol.toUpperCase();
        ModelBundle bundle = models.computeIfAbsent(key, k -> new ModelBundle());
        return loadIfNeeded(key, bundle);
    }

    @Scheduled(fixedDelay = 10_000L)
    public void checkAndReload() {
        for (Map.Entry<String, ModelBundle> entry : models.entrySet()) {
            loadIfNeeded(entry.getKey(), entry.getValue());
        }
    }

    private Optional<ModelBundle> loadIfNeeded(String symbol, ModelBundle bundle) {
        Path modelDir = modelsBaseDir.resolve(symbol).resolve("current");
        Path modelPath = modelDir.resolve("model.onnx");
        Path metaPath = modelDir.resolve("model_meta.json");
        boolean modelExists = Files.exists(modelPath);
        boolean metaExists = Files.exists(metaPath);
        if (!modelExists || !metaExists) {
            closeQuietly(bundle.session);
            bundle.session = null;
            bundle.meta = null;
            bundle.lastModelModified = -1L;
            bundle.lastMetaModified = -1L;
            if (!bundle.missingLogged) {
                log.info("PRED_SKIP_NO_MODEL symbol={} modelDir={}", symbol, modelDir);
                bundle.missingLogged = true;
            }
            return Optional.empty();
        }
        bundle.missingLogged = false;
        try {
            long modelModified = Files.getLastModifiedTime(modelPath).toMillis();
            long metaModified = Files.getLastModifiedTime(metaPath).toMillis();
            if (modelModified == bundle.lastModelModified
                    && metaModified == bundle.lastMetaModified
                    && bundle.session != null
                    && bundle.meta != null) {
                return Optional.of(bundle);
            }
            ModelMeta meta = objectMapper.readValue(metaPath.toFile(), ModelMeta.class);
            closeQuietly(bundle.session);
            OrtSession session = environment.createSession(modelPath.toString(), new OrtSession.SessionOptions());
            bundle.session = session;
            bundle.meta = meta;
            bundle.lastModelModified = modelModified;
            bundle.lastMetaModified = metaModified;
            log.info("MODEL_LOADED symbol={} modelVersion={} nFeatures={}",
                    symbol,
                    meta.getModelVersion(),
                    meta.getFeatureOrder() == null ? 0 : meta.getFeatureOrder().size());
            return Optional.of(bundle);
        } catch (Exception ex) {
            log.info("PRED_SKIP_NO_MODEL symbol={} modelDir={} error=load_failed", symbol, modelDir, ex);
            closeQuietly(bundle.session);
            bundle.session = null;
            bundle.meta = null;
            bundle.lastModelModified = -1L;
            bundle.lastMetaModified = -1L;
            return Optional.empty();
        }
    }

    private void closeQuietly(AutoCloseable closeable) {
        if (closeable == null) {
            return;
        }
        try {
            closeable.close();
        } catch (Exception ignored) {
            // ignore
        }
    }

    public static class ModelBundle {
        private OrtSession session;
        private ModelMeta meta;
        private long lastModelModified = -1L;
        private long lastMetaModified = -1L;
        private boolean missingLogged = false;

        public OrtSession getSession() {
            return session;
        }

        public ModelMeta getMeta() {
            return meta;
        }
    }
}
