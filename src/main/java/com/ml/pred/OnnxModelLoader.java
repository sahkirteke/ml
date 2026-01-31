package com.ml.pred;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ml.config.RawIngestionProperties;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.FloatBuffer;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

@Component
public class OnnxModelLoader {

    private static final Logger log = LoggerFactory.getLogger(OnnxModelLoader.class);

    private final Path modelsBaseDir;
    private final ObjectMapper objectMapper;
    private final OrtEnvironment environment;
    private final Map<String, ModelBundle> models = new ConcurrentHashMap<>();
    private final boolean smokeTestEnabled;

    public OnnxModelLoader(RawIngestionProperties properties, ObjectMapper objectMapper) {
        this.modelsBaseDir = properties.getModelsBaseDir().toAbsolutePath().normalize();
        this.objectMapper = objectMapper;
        this.environment = OrtEnvironment.getEnvironment();
        this.smokeTestEnabled = properties.isSmokeTestEnabled();
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
            logOutputInfo(symbol, session);
            if (smokeTestEnabled) {
                runSmokeTest(symbol, meta, session);
            }
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

    private void logOutputInfo(String symbol, OrtSession session) {
        try {
            Map<String, NodeInfo> outputs = session.getOutputInfo();
            for (Map.Entry<String, NodeInfo> entry : outputs.entrySet()) {
                String name = entry.getKey();
                String shape = "unknown";
                if (entry.getValue().getInfo() instanceof TensorInfo tensorInfo) {
                    shape = java.util.Arrays.toString(tensorInfo.getShape());
                }
                log.info("ONNX_OUTPUT_DEBUG symbol={} name={} shape={}", symbol, name, shape);
            }
        } catch (Exception ex) {
            log.info("ONNX_OUTPUT_DEBUG symbol={} error=output_info_failed", symbol, ex);
        }
    }

    private void runSmokeTest(String symbol, ModelMeta meta, OrtSession session) {
        if (meta.getFeatureOrder() == null || meta.getFeatureOrder().isEmpty()) {
            return;
        }
        String inputName = session.getInputNames().iterator().next();
        int rows = 5;
        int cols = meta.getFeatureOrder().size();
        float[] flat = new float[rows * cols];
        Random random = new Random(42L);
        for (int i = 0; i < flat.length; i++) {
            flat[i] = (float) random.nextGaussian();
        }
        try (OnnxTensor tensor = OnnxTensor.createTensor(
                OrtEnvironment.getEnvironment(),
                FloatBuffer.wrap(flat),
                new long[] {rows, cols});
             OrtSession.Result result = session.run(Map.of(inputName, tensor))) {
            String probName = resolveProbOutputName(meta, session);
            float[] probs = extractProbabilities(result, probName, meta.getUpClassIndex());
            if (probs.length == 0) {
                log.warn("ONNX_SMOKE_TEST symbol={} result=empty", symbol);
                return;
            }
            float min = Float.MAX_VALUE;
            float max = -Float.MAX_VALUE;
            for (float value : probs) {
                min = Math.min(min, value);
                max = Math.max(max, value);
            }
            log.info("ONNX_SMOKE_TEST symbol={} min={} max={}", symbol, min, max);
            if (max < 1e-6f || min > 1.0f - 1e-6f) {
                log.warn("WARN_ONNX_SMOKE_DEGENERATE symbol={} min={} max={}", symbol, min, max);
            }
        } catch (Exception ex) {
            log.info("ONNX_SMOKE_TEST symbol={} error=failed", symbol, ex);
        }
    }

    private String resolveProbOutputName(ModelMeta meta, OrtSession session) {
        if (meta.getProbOutputName() != null && session.getOutputNames().contains(meta.getProbOutputName())) {
            return meta.getProbOutputName();
        }
        if (session.getOutputNames().contains("probabilities")) {
            return "probabilities";
        }
        return session.getOutputNames().iterator().next();
    }

    private float[] extractProbabilities(OrtSession.Result result, String outputName, Integer upIndex) {
        int index = upIndex == null ? 1 : upIndex;
        try {
            Object value = result.get(outputName).getValue();
            if (value instanceof float[][] matrix && matrix.length > 0) {
                float[] output = new float[matrix.length];
                for (int i = 0; i < matrix.length; i++) {
                    float[] row = matrix[i];
                    output[i] = row.length > index ? row[index] : row[0];
                }
                return output;
            }
            if (value instanceof float[] vector && vector.length > 0) {
                return new float[] {vector.length > index ? vector[index] : vector[0]};
            }
        } catch (Exception ignored) {
            // ignore
        }
        return new float[0];
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
