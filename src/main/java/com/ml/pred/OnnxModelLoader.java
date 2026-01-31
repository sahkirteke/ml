package com.ml.pred;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ml.config.RawIngestionProperties;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

@Component
public class OnnxModelLoader {

    private static final Logger log = LoggerFactory.getLogger(OnnxModelLoader.class);

    private final RawIngestionProperties properties;
    private final ObjectMapper objectMapper;
    private OrtEnvironment environment;
    private OrtSession session;
    private ModelMeta meta;

    public OnnxModelLoader(RawIngestionProperties properties, ObjectMapper objectMapper) {
        this.properties = properties;
        this.objectMapper = objectMapper;
        loadModel();
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

    private void loadModel() {
        Path modelDir = properties.getModelDir();
        Path modelPath = modelDir.resolve("model.onnx");
        Path metaPath = modelDir.resolve("model_meta.json");
        if (!Files.exists(modelPath) || !Files.exists(metaPath)) {
            log.info("PRED_SKIP_NO_MODEL modelDir={} modelExists={} metaExists={}",
                    modelDir,
                    Files.exists(modelPath),
                    Files.exists(metaPath));
            return;
        }
        try {
            meta = objectMapper.readValue(metaPath.toFile(), ModelMeta.class);
            environment = OrtEnvironment.getEnvironment();
            session = environment.createSession(modelPath.toString(), new OrtSession.SessionOptions());
            log.info("MODEL_LOADED modelVersion={} featuresVersion={} featureCount={}",
                    meta.getModelVersion(),
                    meta.getFeaturesVersion(),
                    meta.getFeatureOrder() == null ? 0 : meta.getFeatureOrder().size());
        } catch (Exception ex) {
            log.info("PRED_SKIP_NO_MODEL modelDir={} error=load_failed", modelDir, ex);
            meta = null;
            session = null;
            environment = null;
        }
    }
}
