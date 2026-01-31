package com.ml.raw;

import com.ml.features.FeatureCalculator;
import com.ml.features.FeatureRecord;
import com.ml.features.FeatureWriter;
import com.ml.features.LabelRecord;
import com.ml.features.LabelWriter;
import com.ml.features.RollingFeatureState;
import com.ml.features.RollingFeatureStateRegistry;
import com.ml.config.RawIngestionProperties;
import com.ml.pred.ModelMeta;
import com.ml.pred.OnnxInferenceService;
import com.ml.pred.OnnxModelLoader;
import com.ml.pred.EvalRecord;
import com.ml.pred.PredRecord;
import com.ml.pred.PredWriter;
import com.ml.ws.BinanceWsClient;
import com.ml.ws.KlineEvent;
import com.ml.ws.KlinePayload;
import com.ml.ws.WsEnvelope;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.Locale;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Mono;

@Service
@Order(2)
public class RawIngestionService implements ApplicationRunner {

    private static final Logger log = LoggerFactory.getLogger(RawIngestionService.class);
    private static final ZoneId ISTANBUL_ZONE = ZoneId.of("Europe/Istanbul");
    private static final DateTimeFormatter ISO_FORMATTER = DateTimeFormatter.ISO_OFFSET_DATE_TIME;

    private final BinanceWsClient wsClient;
    private final RawRecordBuilder recordBuilder;
    private final GzipJsonlAppender appender;
    private final SymbolState symbolState;
    private final RollingFeatureStateRegistry stateRegistry;
    private final FeatureCalculator featureCalculator;
    private final FeatureWriter featureWriter;
    private final LabelWriter labelWriter;
    private final OnnxInferenceService onnxInferenceService;
    private final OnnxModelLoader onnxModelLoader;
    private final PredWriter predWriter;
    private final RawIngestionProperties properties;

    public RawIngestionService(
            BinanceWsClient wsClient,
            RawRecordBuilder recordBuilder,
            GzipJsonlAppender appender,
            SymbolState symbolState,
            RollingFeatureStateRegistry stateRegistry,
            FeatureCalculator featureCalculator,
            FeatureWriter featureWriter,
            LabelWriter labelWriter,
            OnnxInferenceService onnxInferenceService,
            OnnxModelLoader onnxModelLoader,
            PredWriter predWriter,
            RawIngestionProperties properties
    ) {
        this.wsClient = wsClient;
        this.recordBuilder = recordBuilder;
        this.appender = appender;
        this.symbolState = symbolState;
        this.stateRegistry = stateRegistry;
        this.featureCalculator = featureCalculator;
        this.featureWriter = featureWriter;
        this.labelWriter = labelWriter;
        this.onnxInferenceService = onnxInferenceService;
        this.onnxModelLoader = onnxModelLoader;
        this.predWriter = predWriter;
        this.properties = properties;
    }

    @Override
    public void run(ApplicationArguments args) {
        wsClient.stream()
                .filter(this::isFinalKline)
                .groupBy(this::resolveSymbol)
                .flatMap(grouped -> grouped.concatMap(this::processEnvelope)
                        .onErrorResume(ex -> {
                            log.error("RAW_SYMBOL_ERROR symbol={}", grouped.key(), ex);
                            return Mono.empty();
                        }))
                .onErrorContinue((throwable, o) -> log.error("RAW_PIPELINE_ERROR payload={}", o, throwable))
                .subscribe();
    }

    private Mono<Void> processEnvelope(WsEnvelope envelope) {
        KlineEvent event = envelope.getData();
        if (event == null || event.getKline() == null) {
            return Mono.empty();
        }
        String symbol = resolveSymbol(envelope);
        KlinePayload kline = event.getKline();
        long closeTime = kline.getCloseTime() == null ? -1L : kline.getCloseTime();
        if (!symbolState.updateRawIfNewer(symbol, closeTime)) {
            long last = symbolState.getLastRawWrittenCloseTimeMs(symbol);
            log.info("SKIP_DUP symbol={} closeTimeMs={} last={}", symbol, closeTime, last);
            return Mono.empty();
        }
        long receivedAt = System.currentTimeMillis();
        RawRecord record = recordBuilder.build(event, receivedAt);
        if (record == null) {
            return Mono.empty();
        }
        return Mono.fromRunnable(() -> {
            try {
                appender.append(record);
                RollingFeatureState state = stateRegistry.getOrCreate(symbol);
                evaluatePrediction(symbol, state.getLatest(), record);
                LabelRecord labelRecord = buildLabel(state, record);
                if (labelRecord != null) {
                    if (symbolState.updateLabelsIfNewer(symbol, labelRecord.getCloseTimeMs())) {
                        labelWriter.append(labelRecord);
                    } else {
                        log.info("SKIP_DUP symbol={} closeTimeMs={} last={}",
                                symbol,
                                labelRecord.getCloseTimeMs(),
                                symbolState.getLastLabelsCloseTimeMs(symbol));
                    }
                }
                state.add(record);
                symbolState.setPrevClose(symbol, record.getCloseTimeMs(), parseDouble(record.getClosePrice()));
                FeatureRecord featureRecord = featureCalculator.calculate(state);
                if (featureRecord != null) {
                    if (symbolState.updateFeaturesIfNewer(symbol, featureRecord.getCloseTimeMs())) {
                        featureWriter.append(featureRecord);
                        writePrediction(symbol, featureRecord);
                    } else {
                        log.info("SKIP_DUP symbol={} closeTimeMs={} last={}",
                                symbol,
                                featureRecord.getCloseTimeMs(),
                                symbolState.getLastFeaturesCloseTimeMs(symbol));
                    }
                }
            } catch (Exception ex) {
                throw new RuntimeException(ex);
            }
        });
    }

    private void writePrediction(String symbol, FeatureRecord featureRecord) {
        if (!onnxInferenceService.isAvailable()) {
            log.info("PRED_SKIP_NO_MODEL symbol={} closeTimeMs={}", symbol, featureRecord.getCloseTimeMs());
            return;
        }
        if (!featureRecord.isWindowReady()) {
            log.info("PRED_SKIP_NOT_READY symbol={} closeTimeMs={}", symbol, featureRecord.getCloseTimeMs());
            return;
        }
        long lastPred = symbolState.getLastPredCloseTimeMs(symbol);
        if (featureRecord.getCloseTimeMs() <= lastPred) {
            log.info("PRED_SKIP_DUP symbol={} closeTimeMs={} last={}",
                    symbol,
                    featureRecord.getCloseTimeMs(),
                    lastPred);
            return;
        }
        Optional<ModelMeta> meta = onnxModelLoader.getMeta();
        if (meta.isEmpty()) {
            log.info("PRED_SKIP_NO_MODEL symbol={} closeTimeMs={}", symbol, featureRecord.getCloseTimeMs());
            return;
        }
        try {
            Optional<Double> pUpOpt = onnxInferenceService.predict(featureRecord);
            if (pUpOpt.isEmpty()) {
                log.info("PRED_SKIP_NO_MODEL symbol={} closeTimeMs={}", symbol, featureRecord.getCloseTimeMs());
                return;
            }
            double pUp = pUpOpt.get();
            PredRecord predRecord = new PredRecord();
            predRecord.setType("PRED");
            predRecord.setSymbol(symbol);
            predRecord.setTf(featureRecord.getTf());
            predRecord.setCloseTimeMs(featureRecord.getCloseTimeMs());
            predRecord.setCloseTime(formatMs(featureRecord.getCloseTimeMs()));
            predRecord.setFeaturesVersion(featureRecord.getFeaturesVersion());
            predRecord.setModelVersion(meta.get().getModelVersion());
            predRecord.setPUp(pUp);
            predRecord.setDecision(pUp >= 0.5d ? "UP" : "DOWN");
            predRecord.setLoggedAtMs(System.currentTimeMillis());
            predRecord.setLoggedAt(formatMs(predRecord.getLoggedAtMs()));
            double conf = Math.max(pUp, 1.0d - pUp);
            predRecord.setConfidence(conf);
            Double expectedPct = null;
            Double meanRetUp = meta.get().getMeanRetUp();
            Double meanRetDown = meta.get().getMeanRetDown();
            if (meanRetUp != null && meanRetDown != null) {
                double expectedRet = pUp * meanRetUp + (1.0d - pUp) * meanRetDown;
                expectedPct = expectedRet * 100.0d;
                predRecord.setExpectedPct(expectedPct);
            }
            predWriter.append(predRecord);
            log.info("PRED symbol={} closeTimeMs={} pUp={} conf={} expectedPct={} decision={} modelVersion={}",
                    symbol,
                    featureRecord.getCloseTimeMs(),
                    pUp,
                    conf,
                    expectedPct,
                    predRecord.getDecision(),
                    meta.get().getModelVersion());
            symbolState.updatePredIfNewer(symbol, featureRecord.getCloseTimeMs());
            symbolState.setLastPredInfo(symbol, predRecord.getDecision(), pUp);
        } catch (Exception ex) {
            log.info("PRED_SKIP_NO_MODEL symbol={} closeTimeMs={}", symbol, featureRecord.getCloseTimeMs(), ex);
        }
    }

    private void evaluatePrediction(String symbol, RollingFeatureState.Bar previous, RawRecord current) {
        if (previous == null) {
            return;
        }
        long predCloseTimeMs = symbolState.getLastPredCloseTimeMs(symbol);
        if (predCloseTimeMs <= 0 || predCloseTimeMs != previous.getCloseTimeMs()) {
            return;
        }
        long expectedGap = properties.getExpectedGapMs() == null ? 300_000L : properties.getExpectedGapMs();
        long gapMs = current.getCloseTimeMs() - previous.getCloseTimeMs();
        EvalRecord evalRecord = new EvalRecord();
        evalRecord.setType("EVAL");
        evalRecord.setSymbol(symbol);
        evalRecord.setTf(properties.getTf());
        evalRecord.setPredCloseTimeMs(previous.getCloseTimeMs());
        evalRecord.setPredCloseTime(formatMs(previous.getCloseTimeMs()));
        evalRecord.setPredDecision(symbolState.getLastPredDecision(symbol));
        evalRecord.setPredPUp(symbolState.getLastPredPUp(symbol));
        evalRecord.setActualCloseTimeMs(current.getCloseTimeMs());
        evalRecord.setActualCloseTime(formatMs(current.getCloseTimeMs()));
        evalRecord.setEvaluatedAtMs(System.currentTimeMillis());
        evalRecord.setEvaluatedAt(formatMs(evalRecord.getEvaluatedAtMs()));
        if (gapMs != expectedGap) {
            evalRecord.setResult("SKIP_GAP");
            try {
                predWriter.append(evalRecord);
                log.info("EVAL symbol={} predCloseTimeMs={} result=SKIP_GAP",
                        symbol,
                        previous.getCloseTimeMs());
            } catch (Exception ex) {
                log.info("PRED_SKIP_NO_MODEL symbol={} closeTimeMs={}", symbol, current.getCloseTimeMs(), ex);
            }
            return;
        }
        double currentClose = parseDouble(current.getClosePrice());
        double prevClose = previous.getClose();
        if (prevClose == 0.0d || currentClose == 0.0d) {
            return;
        }
        double futureRet = currentClose / prevClose - 1.0d;
        boolean actualUp = futureRet > 0.0d;
        String predDecision = symbolState.getLastPredDecision(symbol);
        String result = "NOK";
        if ("UP".equalsIgnoreCase(predDecision) && actualUp) {
            result = "OK";
        } else if ("DOWN".equalsIgnoreCase(predDecision) && !actualUp) {
            result = "OK";
        }
        evalRecord.setFutureRet_1(futureRet);
        evalRecord.setActualUp(actualUp);
        evalRecord.setResult(result);
        try {
            predWriter.append(evalRecord);
            log.info("EVAL symbol={} predCloseTimeMs={} result={} actualUp={} futureRet_1={}",
                    symbol,
                    previous.getCloseTimeMs(),
                    result,
                    actualUp,
                    futureRet);
        } catch (Exception ex) {
            log.info("PRED_SKIP_NO_MODEL symbol={} closeTimeMs={}", symbol, current.getCloseTimeMs(), ex);
        }
    }

    private LabelRecord buildLabel(RollingFeatureState state, RawRecord current) {
        RollingFeatureState.Bar previous = state.getLatest();
        if (previous == null) {
            return null;
        }
        double prevClose = previous.getClose();
        if (prevClose == 0.0d) {
            return null;
        }
        double currentClose = parseDouble(current.getClosePrice());
        if (currentClose == 0.0d) {
            return null;
        }
        double futureRet = currentClose / prevClose - 1.0d;
        long expectedGap = properties.getExpectedGapMs() == null ? 300_000L : properties.getExpectedGapMs();
        long gapMs = current.getCloseTimeMs() - previous.getCloseTimeMs();
        if (gapMs != expectedGap) {
            log.info("SKIP_GAP_LABEL symbol={} prevCloseTimeMs={} closeTimeMs={} gapMs={} expectedGapMs={}",
                    current.getSymbol(),
                    previous.getCloseTimeMs(),
                    current.getCloseTimeMs(),
                    gapMs,
                    expectedGap);
            return null;
        }
        LabelRecord label = new LabelRecord();
        label.setSymbol(current.getSymbol());
        label.setTf(current.getTf());
        label.setCloseTimeMs(previous.getCloseTimeMs());
        label.setLabelType("next_close_direction");
        label.setFutureRet_1(futureRet);
        label.setLabelUp(futureRet > 0.0d ? 1 : 0);
        return label;
    }

    private double parseDouble(String value) {
        if (value == null || value.isBlank()) {
            return 0.0d;
        }
        try {
            return Double.parseDouble(value);
        } catch (NumberFormatException ex) {
            return 0.0d;
        }
    }

    private boolean isFinalKline(WsEnvelope envelope) {
        if (envelope == null || envelope.getData() == null || envelope.getData().getKline() == null) {
            return false;
        }
        Boolean isFinal = envelope.getData().getKline().getIsFinal();
        return Boolean.TRUE.equals(isFinal);
    }

    private String resolveSymbol(WsEnvelope envelope) {
        String symbol = envelope.getData() == null ? null : envelope.getData().getSymbol();
        if (symbol == null || symbol.isBlank()) {
            return "UNKNOWN";
        }
        return symbol.toUpperCase(Locale.ROOT);
    }

    private String formatMs(long epochMs) {
        return Instant.ofEpochMilli(epochMs)
                .atZone(ISTANBUL_ZONE)
                .format(ISO_FORMATTER);
    }
}
