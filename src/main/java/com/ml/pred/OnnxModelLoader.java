package com.ml.pred;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ml.config.RawIngestionProperties;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

@Component
public class OnnxModelLoader {

    private static final Logger log = LoggerFactory.getLogger(OnnxModelLoader.class);

    private final Path modelDir;
    private final ObjectMapper objectMapper;
    private OrtEnvironment environment;
    private OrtSession session;
    private ModelMeta meta;
    private long lastModelModified = -1L;
    private long lastMetaModified = -1L;
    private boolean missingLogged = false;

    public OnnxModelLoader(RawIngestionProperties properties, ObjectMapper objectMapper) {
        this.modelDir = properties.getModelDir().toAbsolutePath().normalize();
        this.objectMapper = objectMapper;
        log.info("MODEL_DIR path={}", modelDir);
        checkAndReload();
    }

    public Optional<OrtSession> getSession() {
        return Optional.ofNullable(session);
    }

    public Optional<ModelMeta> getMeta() {
        return Optional.ofNullable(meta);
    }

    public Optional<OrtEnvironment> getEnvironment() {
        return Optional.ofNullable(environment);
    }

    public boolean isLoaded() {
        return session != null && meta != null;
    }

    @Scheduled(fixedDelay = 10_000L)
    public void checkAndReload() {
        Path modelPath = modelDir.resolve("model.onnx");
        Path metaPath = modelDir.resolve("model_meta.json");
        boolean modelExists = Files.exists(modelPath);
        boolean metaExists = Files.exists(metaPath);
        if (!modelExists || !metaExists) {
            session = null;
            meta = null;
            environment = null;
            lastModelModified = -1L;
            lastMetaModified = -1L;
            if (!missingLogged) {
                log.info("PRED_SKIP_NO_MODEL modelDir={} modelExists={} metaExists={}",
                        modelDir,
                        modelExists,
                        metaExists);
                missingLogged = true;
            }
            return;
        }
        missingLogged = false;
        try {
            long modelModified = Files.getLastModifiedTime(modelPath).toMillis();
            long metaModified = Files.getLastModifiedTime(metaPath).toMillis();
            if (modelModified == lastModelModified && metaModified == lastMetaModified && session != null) {
                return;
            }
            loadModel(modelPath, metaPath, modelModified, metaModified);
        } catch (Exception ex) {
            log.info("PRED_SKIP_NO_MODEL modelDir={} error=load_failed", modelDir, ex);
        }
    }

    private void loadModel(Path modelPath, Path metaPath, long modelModified, long metaModified) {
        try {
            meta = objectMapper.readValue(metaPath.toFile(), ModelMeta.class);
            environment = OrtEnvironment.getEnvironment();
            session = environment.createSession(modelPath.toString(), new OrtSession.SessionOptions());
            lastModelModified = modelModified;
            lastMetaModified = metaModified;
            log.info("MODEL_LOADED modelVersion={} featuresVersion={} nFeatures={}",
                    meta.getModelVersion(),
                    meta.getFeaturesVersion(),
                    meta.getFeatureOrder() == null ? 0 : meta.getFeatureOrder().size());
        } catch (Exception ex) {
            log.info("PRED_SKIP_NO_MODEL modelDir={} error=load_failed", modelDir, ex);
            meta = null;
            session = null;
            environment = null;
            lastModelModified = -1L;
            lastMetaModified = -1L;
        }
    }
}
