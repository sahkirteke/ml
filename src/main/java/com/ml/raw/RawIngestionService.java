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
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
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
    private static final double TP_PCT = 0.004d;
    private static final double SL_PCT = 0.002d;
    private static final int MAX_HORIZON_BARS = 7;

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
    private final Map<String, Deque<PendingPrediction>> pendingBySymbol = new ConcurrentHashMap<>();

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
                RollingFeatureState.Bar previous = state.getLatest();
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
                    } else {
                        log.info("SKIP_DUP symbol={} closeTimeMs={} last={}",
                                symbol,
                                featureRecord.getCloseTimeMs(),
                                symbolState.getLastFeaturesCloseTimeMs(symbol));
                    }
                }
                evaluatePendingPredictions(symbol, previous, record);
                if (featureRecord != null) {
                    writePrediction(symbol, featureRecord);
                }
            } catch (Exception ex) {
                throw new RuntimeException(ex);
            }
        });
    }

    private void writePrediction(String symbol, FeatureRecord featureRecord) {
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
        Optional<OnnxModelLoader.ModelBundle> modelOpt = onnxModelLoader.getModel(symbol);
        if (modelOpt.isEmpty()) {
            return;
        }
        ModelMeta modelMeta = modelOpt.get().getMeta();
        if (modelMeta == null) {
            return;
        }
        logPredInputDebug(symbol, featureRecord, modelMeta);
        try {
            Optional<Double> pUpOpt = onnxInferenceService.predict(featureRecord, modelOpt.get());
            if (pUpOpt.isEmpty()) {
                return;
            }
            double pUp = pUpOpt.get();
            double minConfidence = resolveMinConfidence(modelMeta);
            double minAbsExpectedPct = resolveMinAbsExpectedPct(modelMeta);
            double minAbsEdge = resolveMinAbsEdge(modelMeta);
            PredRecord predRecord = new PredRecord();
            predRecord.setType("PRED");
            predRecord.setSymbol(symbol);
            predRecord.setTf(featureRecord.getTf());
            predRecord.setCloseTimeMs(featureRecord.getCloseTimeMs());
            predRecord.setCloseTime(formatMs(featureRecord.getCloseTimeMs()));
            predRecord.setFeaturesVersion(featureRecord.getFeaturesVersion());
            predRecord.setModelVersion(modelMeta.getModelVersion());
            predRecord.setPUp(pUp);
            predRecord.setLoggedAtMs(System.currentTimeMillis());
            predRecord.setLoggedAt(formatMs(predRecord.getLoggedAtMs()));
            double conf = Math.max(pUp, 1.0d - pUp);
            predRecord.setConfidence(conf);
            double edgeAbs = Math.abs(pUp - 0.5d);
            predRecord.setEdgeAbs(edgeAbs);
            Double expectedPct = null;
            Double expectedBp = null;
            Double meanRetUp = modelMeta.getMeanRetUp();
            Double meanRetDown = modelMeta.getMeanRetDown();
            if (meanRetUp != null && meanRetDown != null) {
                double expectedRet = pUp * meanRetUp + (1.0d - pUp) * meanRetDown;
                expectedPct = expectedRet;
                predRecord.setExpectedPct(expectedPct);
                expectedBp = expectedPct * 10000.0d;
                predRecord.setExpectedBp(expectedBp);
            }
            predRecord.setMinConfidence(minConfidence);
            predRecord.setMinAbsExpectedPct(minAbsExpectedPct);
            predRecord.setMinAbsEdge(minAbsEdge);
            boolean lowExpected = expectedPct != null && Math.abs(expectedPct) < minAbsExpectedPct;
            boolean lowEdge = edgeAbs < minAbsEdge;
            String decisionReason = "DIRECT";
            String failedGate = "NONE";
            String decision;
            if (conf < minConfidence) {
                decision = "NO_TRADE";
                decisionReason = "LOW_CONF";
                failedGate = "CONFIDENCE";
            } else if (lowExpected || lowEdge) {
                decision = "NO_TRADE";
                if (lowExpected && lowEdge) {
                    decisionReason = "LOW_EXPECTED|LOW_EDGE";
                    failedGate = "EXPECTED+EDGE";
                } else if (lowExpected) {
                    decisionReason = "LOW_EXPECTED";
                    failedGate = "EXPECTED";
                } else {
                    decisionReason = "LOW_EDGE";
                    failedGate = "EDGE";
                }
            } else {
                decision = pUp >= 0.5d ? "UP" : "DOWN";
            }
            predRecord.setDirection(decision);
            predRecord.setDecisionReason(decisionReason);
            predRecord.setFailedGate(failedGate);
            double entryPrice = parseDouble(featureRecord.getClosePrice());
            if (entryPrice > 0.0d && !"NO_TRADE".equalsIgnoreCase(decision)) {
                if ("UP".equalsIgnoreCase(decision)) {
                    predRecord.setEntryPrice(entryPrice);
                    predRecord.setTpPrice(entryPrice * (1.0d + TP_PCT));
                    predRecord.setSlPrice(entryPrice * (1.0d - SL_PCT));
                } else {
                    predRecord.setEntryPrice(entryPrice);
                    predRecord.setTpPrice(entryPrice * (1.0d - TP_PCT));
                    predRecord.setSlPrice(entryPrice * (1.0d + SL_PCT));
                }
            }
            predWriter.append(predRecord);
            log.info(
                    "PRED symbol={} closeTimeMs={} pUp={} conf={} edgeAbs={} expectedPct={} expectedBp={} "
                            + "minConfidence={} minAbsExpectedPct={} minAbsEdge={} failedGate={} decision={} "
                            + "reason={} modelVersion={}",
                    symbol,
                    featureRecord.getCloseTimeMs(),
                    pUp,
                    conf,
                    edgeAbs,
                    expectedPct,
                    expectedBp,
                    minConfidence,
                    minAbsExpectedPct,
                    minAbsEdge,
                    failedGate,
                    predRecord.getDirection(),
                    decisionReason,
                    modelMeta.getModelVersion());
            symbolState.updatePredIfNewer(symbol, featureRecord.getCloseTimeMs());
            symbolState.setLastPredInfo(symbol, predRecord.getDirection(), pUp);
            if ("UP".equalsIgnoreCase(decision) || "DOWN".equalsIgnoreCase(decision)) {
                enqueuePending(symbol, featureRecord, predRecord);
            }
        } catch (Exception ex) {
            log.info("PRED_SKIP_NO_MODEL symbol={} closeTimeMs={}", symbol, featureRecord.getCloseTimeMs(), ex);
        }
    }

    private void evaluatePendingPredictions(String symbol, RollingFeatureState.Bar previous, RawRecord current) {
        if (previous == null) {
            return;
        }
        Deque<PendingPrediction> pending = pendingBySymbol.computeIfAbsent(symbol, key -> new ArrayDeque<>());
        if (pending.isEmpty()) {
            return;
        }
        long expectedGap = properties.getExpectedGapMs() == null ? 300_000L : properties.getExpectedGapMs();
        long gapMs = current.getCloseTimeMs() - previous.getCloseTimeMs();
        if (gapMs != expectedGap) {
            pending.clear();
            return;
        }
        double barHigh = parseDouble(current.getHighPrice());
        double barLow = parseDouble(current.getLowPrice());
        if (barHigh == 0.0d || barLow == 0.0d) {
            return;
        }
        pending.removeIf(item -> {
            item.incrementBarsWaited();
            if (item.getBarsWaited() > MAX_HORIZON_BARS) {
                return true;
            }
            boolean hitTp;
            boolean hitSl;
            if ("UP".equalsIgnoreCase(item.getDirection())) {
                hitTp = barHigh >= item.getTpPrice();
                hitSl = barLow <= item.getSlPrice();
            } else {
                hitTp = barLow <= item.getTpPrice();
                hitSl = barHigh >= item.getSlPrice();
            }
            if (!(hitTp || hitSl)) {
                return false;
            }
            String event = hitSl ? "SL_HIT" : "TP_HIT";
            String result = hitSl ? "NOK" : "OK";
            EvalRecord evalRecord = new EvalRecord();
            evalRecord.setType("EVAL");
            evalRecord.setSymbol(symbol);
            evalRecord.setTf(properties.getTf());
            evalRecord.setPredCloseTimeMs(item.getPredCloseTimeMs());
            evalRecord.setEventCloseTimeMs(current.getCloseTimeMs());
            evalRecord.setDirection(item.getDirection());
            evalRecord.setEntryPrice(item.getEntryPrice());
            evalRecord.setTpPrice(item.getTpPrice());
            evalRecord.setSlPrice(item.getSlPrice());
            evalRecord.setResult(result);
            evalRecord.setEvent(event);
            evalRecord.setLoggedAtMs(System.currentTimeMillis());
            evalRecord.setLoggedAt(formatMs(evalRecord.getLoggedAtMs()));
            try {
                predWriter.append(evalRecord);
                log.info("EVAL symbol={} predCloseTimeMs={} result={} event={}",
                        symbol,
                        item.getPredCloseTimeMs(),
                        result,
                        event);
            } catch (Exception ex) {
                log.info("PRED_SKIP_NO_MODEL symbol={} closeTimeMs={}", symbol, current.getCloseTimeMs(), ex);
            }
            return true;
        });
    }

    private void enqueuePending(String symbol, FeatureRecord featureRecord, PredRecord predRecord) {
        double entryPrice = predRecord.getEntryPrice() == null ? 0.0d : predRecord.getEntryPrice();
        Double tpPrice = predRecord.getTpPrice();
        Double slPrice = predRecord.getSlPrice();
        if (entryPrice <= 0.0d || tpPrice == null || slPrice == null) {
            return;
        }
        PendingPrediction pending = new PendingPrediction(
                featureRecord.getCloseTimeMs(),
                entryPrice,
                predRecord.getDirection(),
                tpPrice,
                slPrice
        );
        pendingBySymbol.computeIfAbsent(symbol, key -> new ArrayDeque<>()).addLast(pending);
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

    private void logPredInputDebug(String symbol, FeatureRecord featureRecord, ModelMeta meta) {
        long count = symbolState.incrementPredDebugCount(symbol);
        if (count > 3) {
            return;
        }
        if (meta.getFeatureOrder() == null || meta.getFeatureOrder().isEmpty()) {
            return;
        }
        StringBuilder builder = new StringBuilder();
        builder.append("PRED_INPUT_DEBUG symbol=").append(symbol)
                .append(" closeTimeMs=").append(featureRecord.getCloseTimeMs());
        int limit = Math.min(8, meta.getFeatureOrder().size());
        boolean hasNaN = false;
        for (int i = 0; i < meta.getFeatureOrder().size(); i++) {
            String name = meta.getFeatureOrder().get(i);
            Double value = resolveFeatureValue(featureRecord, name);
            if (value != null && value.isNaN()) {
                hasNaN = true;
            }
            if (i < limit) {
                builder.append(" f").append(i).append("=")
                        .append(name)
                        .append(":")
                        .append(value == null ? "null" : value);
            }
        }
        builder.append(" hasNaN=").append(hasNaN);
        log.info(builder.toString());
    }

    private Double resolveFeatureValue(FeatureRecord record, String name) {
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

    private double resolveMinConfidence(ModelMeta meta) {
        double fallback = properties.getDecision() == null || properties.getDecision().getMinConfidence() == null
                ? 0.55d
                : properties.getDecision().getMinConfidence();
        if (meta.getDecisionPolicy() != null && meta.getDecisionPolicy().getMinConfidence() != null) {
            return meta.getDecisionPolicy().getMinConfidence();
        }
        return fallback;
    }

    private double resolveMinAbsExpectedPct(ModelMeta meta) {
        double fallback = properties.getDecision() == null || properties.getDecision().getMinAbsExpectedPct() == null
                ? 0.002d
                : properties.getDecision().getMinAbsExpectedPct();
        if (meta.getDecisionPolicy() != null && meta.getDecisionPolicy().getMinAbsExpectedPct() != null) {
            return meta.getDecisionPolicy().getMinAbsExpectedPct();
        }
        return fallback;
    }

    private double resolveMinAbsEdge(ModelMeta meta) {
        double fallback = properties.getDecision() == null || properties.getDecision().getMinAbsEdge() == null
                ? 0.05d
                : properties.getDecision().getMinAbsEdge();
        if (meta.getDecisionPolicy() != null && meta.getDecisionPolicy().getMinAbsEdge() != null) {
            return meta.getDecisionPolicy().getMinAbsEdge();
        }
        return fallback;
    }

    private String formatMs(long epochMs) {
        return Instant.ofEpochMilli(epochMs)
                .atZone(ISTANBUL_ZONE)
                .format(ISO_FORMATTER);
    }

    private static final class PendingPrediction {
        private final long predCloseTimeMs;
        private final double entryPrice;
        private final String direction;
        private final double tpPrice;
        private final double slPrice;
        private int barsWaited;

        private PendingPrediction(long predCloseTimeMs, double entryPrice, String direction, double tpPrice, double slPrice) {
            this.predCloseTimeMs = predCloseTimeMs;
            this.entryPrice = entryPrice;
            this.direction = direction;
            this.tpPrice = tpPrice;
            this.slPrice = slPrice;
            this.barsWaited = 0;
        }

        private long getPredCloseTimeMs() {
            return predCloseTimeMs;
        }

        private double getEntryPrice() {
            return entryPrice;
        }

        private String getDirection() {
            return direction;
        }

        private double getTpPrice() {
            return tpPrice;
        }

        private double getSlPrice() {
            return slPrice;
        }

        private int getBarsWaited() {
            return barsWaited;
        }

        private void incrementBarsWaited() {
            barsWaited += 1;
        }
    }
}
