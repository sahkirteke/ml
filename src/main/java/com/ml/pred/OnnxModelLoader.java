package com.ml.pred;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.ml.config.RawIngestionProperties;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
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
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

@Component
public class OnnxModelLoader {

    private static final Logger log = LoggerFactory.getLogger(OnnxModelLoader.class);

    private final Path modelsBaseDir;
    private final ObjectMapper objectMapper;
    private final OrtEnvironment environment;
    private static final long MISSING_LOG_INTERVAL_MS = 300_000L;
    private final Map<String, ModelState> models = new ConcurrentHashMap<>();
    private final Map<String, Long> missingLogBySymbol = new ConcurrentHashMap<>();
    private final boolean smokeTestEnabled;

    public OnnxModelLoader(RawIngestionProperties properties, ObjectMapper objectMapper) {
        this.modelsBaseDir = properties.getModelsBaseDir().toAbsolutePath().normalize();
        this.objectMapper = objectMapper;
        this.environment = OrtEnvironment.getEnvironment();
        this.smokeTestEnabled = properties.isSmokeTestEnabled();
        log.info("MODELS_BASE_DIR path={}", modelsBaseDir);
    }

    public Optional<ModelBundle> loadBundle(String symbol) {
        if (symbol == null || symbol.isBlank()) {
            return Optional.empty();
        }
        String key = symbol.toUpperCase();
        ModelState state = models.computeIfAbsent(key, k -> new ModelState());
        return loadIfNeeded(key, state);
    }

    @Scheduled(fixedDelay = 10_000L)
    public void checkAndReload() {
        for (Map.Entry<String, ModelState> entry : models.entrySet()) {
            loadIfNeeded(entry.getKey(), entry.getValue());
        }
    }

    private Optional<ModelBundle> loadIfNeeded(String symbol, ModelState state) {
        Path longDir = modelsBaseDir.resolve(symbol).resolve("current_long");
        Path shortDir = modelsBaseDir.resolve(symbol).resolve("current_short");
        boolean longReady = hasModelFiles(longDir);
        boolean shortReady = hasModelFiles(shortDir);
        if (!longReady && !shortReady) {
            clearVariant(state.longModel);
            clearVariant(state.shortModel);
            Optional<ModelBundle> fallback = loadFallbackCurrent(symbol, state);
            if (fallback.isPresent()) {
                return fallback;
            }
            logMissingModel(symbol, "PRED_SKIP_NO_MODEL symbol=%s longDir=%s shortDir=%s"
                    .formatted(symbol, longDir, shortDir));
            return Optional.empty();
        }
        if (!longReady || !shortReady) {
            if (!longReady) {
                clearVariant(state.longModel);
            }
            if (!shortReady) {
                clearVariant(state.shortModel);
            }
            clearVariant(state.fallbackModel);
            String missing = longReady ? "SHORT" : "LONG";
            logMissingModel(symbol, "PRED_SKIP_PARTIAL_MODEL symbol=%s missing=%s"
                    .formatted(symbol, missing));
            return Optional.empty();
        }
        clearVariant(state.fallbackModel);
        Optional<ModelVariant> longModel = loadVariant(symbol, state.longModel, longDir, "LONG");
        Optional<ModelVariant> shortModel = loadVariant(symbol, state.shortModel, shortDir, "SHORT");
        if (longModel.isEmpty() || shortModel.isEmpty()) {
            return Optional.empty();
        }
        ModelVariant longVariant = longModel.get();
        ModelVariant shortVariant = shortModel.get();
        return Optional.of(new ModelBundle(
                longVariant.session,
                longVariant.meta,
                shortVariant.session,
                shortVariant.meta));
    }

    private boolean hasModelFiles(Path modelDir) {
        return Files.exists(modelDir.resolve("model.onnx")) && Files.exists(modelDir.resolve("model_meta.json"));
    }

    private Optional<ModelVariant> loadVariant(String symbol, ModelVariant variant, Path modelDir, String side) {
        return loadVariant(symbol, variant, modelDir, side, true);
    }

    private Optional<ModelVariant> loadVariant(
            String symbol,
            ModelVariant variant,
            Path modelDir,
            String side,
            boolean logFailures
    ) {
        Path modelPath = modelDir.resolve("model.onnx");
        Path metaPath = modelDir.resolve("model_meta.json");
        try {
            long modelModified = Files.getLastModifiedTime(modelPath).toMillis();
            long metaModified = Files.getLastModifiedTime(metaPath).toMillis();
            if (modelModified == variant.lastModelModified
                    && metaModified == variant.lastMetaModified
                    && variant.session != null
                    && variant.meta != null) {
                return Optional.of(variant);
            }
            ModelMeta meta = readMeta(metaPath);
            closeQuietly(variant.session);
            OrtSession session = environment.createSession(modelPath.toString(), new OrtSession.SessionOptions());
            variant.session = session;
            variant.meta = meta;
            variant.lastModelModified = modelModified;
            variant.lastMetaModified = metaModified;
            log.info("MODEL_LOADED symbol={} side={} modelVersion={} nFeatures={}",
                    symbol,
                    side,
                    resolveModelVersion(meta),
                    meta.getFeatureOrder() == null ? 0 : meta.getFeatureOrder().size());
            logOutputInfo(symbol, session);
            if (smokeTestEnabled) {
                runSmokeTest(symbol, meta, session);
            }
            return Optional.of(variant);
        } catch (Exception ex) {
            if (logFailures) {
                log.info("PRED_SKIP_NO_MODEL symbol={} modelDir={} error=load_failed", symbol, modelDir, ex);
            }
            clearVariant(variant);
            return Optional.empty();
        }
    }

    private void clearVariant(ModelVariant variant) {
        closeQuietly(variant.session);
        variant.session = null;
        variant.meta = null;
        variant.lastModelModified = -1L;
        variant.lastMetaModified = -1L;
    }

    private void logMissingModel(String symbol, String message) {
        long now = System.currentTimeMillis();
        Long last = missingLogBySymbol.get(symbol);
        if (last == null || now - last >= MISSING_LOG_INTERVAL_MS) {
            missingLogBySymbol.put(symbol, now);
            log.info(message);
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

    private String resolveProbOutputName(OrtSession session) {
        if (session.getOutputNames().contains("probabilities")) {
            return "probabilities";
        }
        return session.getOutputNames().iterator().next();
    }

    private String resolveProbOutputName(ModelMeta meta, OrtSession session) {
        if (meta.getProbOutputName() != null && session.getOutputNames().contains(meta.getProbOutputName())) {
            return meta.getProbOutputName();
        }
        return resolveProbOutputName(session);
    }

    private float[] extractProbabilities(OrtSession.Result result, String outputName, Integer upIndex) {
        int index = upIndex == null ? 1 : upIndex;
        try {
            OnnxValue onnxValue = result.get(outputName).orElse(null);
            if (onnxValue == null) {
                return new float[0];
            }
            Object value = onnxValue.getValue();
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

    public record ModelBundle(
            OrtSession longSession,
            ModelMeta longMeta,
            OrtSession shortSession,
            ModelMeta shortMeta
    ) {}

    private static final class ModelVariant {
        private OrtSession session;
        private ModelMeta meta;
        private long lastModelModified = -1L;
        private long lastMetaModified = -1L;
    }

    private static final class ModelState {
        private final ModelVariant longModel = new ModelVariant();
        private final ModelVariant shortModel = new ModelVariant();
        private final ModelVariant fallbackModel = new ModelVariant();
    }

    private Optional<ModelBundle> loadFallbackCurrent(String symbol, ModelState state) {
        Path currentDir = modelsBaseDir.resolve(symbol).resolve("current");
        if (!hasModelFiles(currentDir)) {
            clearVariant(state.fallbackModel);
            return Optional.empty();
        }
        Optional<ModelVariant> fallback = loadVariant(symbol, state.fallbackModel, currentDir, "FALLBACK", false);
        if (fallback.isEmpty()) {
            return Optional.empty();
        }
        ModelVariant variant = fallback.get();
        return Optional.of(new ModelBundle(
                variant.session,
                variant.meta,
                variant.session,
                variant.meta));
    }

    private ModelMeta readMeta(Path metaPath) throws Exception {
        try {
            return objectMapper.readValue(metaPath.toFile(), ModelMeta.class);
        } catch (Exception ex) {
            Map<String, Object> fallback = objectMapper.readValue(metaPath.toFile(), new TypeReference<>() {});
            ModelMeta meta = new ModelMeta();
            Object featureOrder = fallback.get("featureOrder");
            if (featureOrder instanceof List<?> list) {
                meta.setFeatureOrder(list.stream().map(String::valueOf).toList());
            }
            if (fallback.get("probOutputName") != null) {
                meta.setProbOutputName(String.valueOf(fallback.get("probOutputName")));
            }
            if (fallback.get("upClassIndex") instanceof Number number) {
                meta.setUpClassIndex(number.intValue());
            }
            if (fallback.get("modelVersion") != null) {
                meta.setModelVersion(String.valueOf(fallback.get("modelVersion")));
            }
            return meta;
        }
    }

    private String resolveModelVersion(ModelMeta meta) {
        return meta == null ? null : meta.getModelVersion();
    }
}
