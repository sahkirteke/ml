package com.ml.pred;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import com.ml.features.FeatureRecord;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.springframework.stereotype.Component;

@Component
public class OnnxInferenceService {
    public Optional<Double> predict(FeatureRecord record, OnnxModelLoader.ModelBundle model) throws OrtException {
        if (model == null || model.getSession() == null || model.getMeta() == null) {
            return Optional.empty();
        }
        ModelMeta meta = model.getMeta();
        List<String> order = meta.getFeatureOrder();
        if (order == null || order.isEmpty()) {
            return Optional.empty();
        }
        float[] inputs = buildInputs(record, order);
        String inputName = model.getSession().getInputNames().iterator().next();
        try (OnnxTensor tensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(),
                FloatBuffer.wrap(inputs),
                new long[] {1, inputs.length});
             OrtSession.Result result = model.getSession().run(Map.of(inputName, tensor))) {
            return Optional.of(extractProbability(result));
        }
    }

    private float[] buildInputs(FeatureRecord record, List<String> order) {
        float[] values = new float[order.size()];
        for (int i = 0; i < order.size(); i++) {
            Double value = resolveFeature(record, order.get(i));
            values[i] = value == null ? 0.0f : value.floatValue();
        }
        return values;
    }

    private Double resolveFeature(FeatureRecord record, String name) {
        return switch (name) {
            case "ret_1" -> record.getRet_1();
            case "logRet_1" -> record.getLogRet_1();
            case "ret_3" -> record.getRet_3();
            case "ret_12" -> record.getRet_12();
            case "realizedVol_6" -> record.getRealizedVol_6();
            case "realizedVol_24" -> record.getRealizedVol_24();
            case "rangePct" -> record.getRangePct();
            case "bodyPct" -> record.getBodyPct();
            case "upperWickPct" -> record.getUpperWickPct();
            case "lowerWickPct" -> record.getLowerWickPct();
            case "closePos" -> record.getClosePos();
            case "volRatio_12" -> record.getVolRatio_12();
            case "tradeRatio_12" -> record.getTradeRatio_12();
            case "buySellRatio" -> record.getBuySellRatio();
            case "deltaVolNorm" -> record.getDeltaVolNorm();
            case "rsi14" -> record.getRsi14();
            case "atr14" -> record.getAtr14();
            case "ema20DistPct" -> record.getEma20DistPct();
            case "ema200DistPct" -> record.getEma200DistPct();
            default -> null;
        };
    }

    private double extractProbability(OrtSession.Result result) throws OrtException {
        for (Map.Entry<String, OnnxValue> entry : result) {
            OnnxValue value = entry.getValue();
            if (value instanceof OnnxTensor tensor) {
                Object raw = tensor.getValue();
                if (raw instanceof float[][] matrix && matrix.length > 0) {
                    float[] row = matrix[0];
                    if (row.length >= 2) {
                        return row[1];
                    }
                    if (row.length == 1) {
                        return row[0];
                    }
                } else if (raw instanceof float[] vector && vector.length > 0) {
                    return vector.length >= 2 ? vector[1] : vector[0];
                }
            }
        }
        return 0.0d;
    }
}
