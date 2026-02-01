package com.ml.raw;

import com.ml.features.FeatureCalculator;
import com.ml.features.FeatureRecord;
import com.ml.features.FeatureWriter;
import com.ml.features.LabelRecord;
import com.ml.features.LabelWriter;
import com.ml.features.RollingFeatureState;
import com.ml.features.RollingFeatureStateRegistry;
import com.ml.config.RawIngestionProperties;
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
    private static final double P_TRADE = 0.55d;
    private static final int HORIZON_BARS = 7;

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
    private final Map<String, PendingTrade> pendingBySymbol = new ConcurrentHashMap<>();

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
                evaluatePendingWithBar(symbol, previous, record);
                if (featureRecord != null && !pendingBySymbol.containsKey(symbol)) {
                    writePrediction(symbol, featureRecord);
                }
            } catch (Exception ex) {
                throw new RuntimeException(ex);
            }
        });
    }

    private void writePrediction(String symbol, FeatureRecord featureRecord) {
        if (!featureRecord.isWindowReady()) {
            return;
        }
        long lastPred = symbolState.getLastPredCloseTimeMs(symbol);
        if (featureRecord.getCloseTimeMs() <= lastPred) {
            return;
        }
        Optional<OnnxModelLoader.ModelPair> modelOpt = onnxModelLoader.getModelPair(symbol);
        if (modelOpt.isEmpty()) {
            return;
        }
        OnnxModelLoader.ModelPair modelPair = modelOpt.get();
        try {
            Optional<Double> pLongOpt = onnxInferenceService.predict(featureRecord, modelPair.getLongModel());
            Optional<Double> pShortOpt = onnxInferenceService.predict(featureRecord, modelPair.getShortModel());
            if (pLongOpt.isEmpty() || pShortOpt.isEmpty()) {
                return;
            }
            double pLongHit = pLongOpt.get();
            double pShortHit = pShortOpt.get();
            double best = Math.max(pLongHit, pShortHit);
            if (best < P_TRADE) {
                return;
            }
            String direction = pLongHit > pShortHit ? "UP" : "DOWN";
            double pHit = direction.equals("UP") ? pLongHit : pShortHit;
            PredRecord predRecord = new PredRecord();
            predRecord.setType("PRED");
            predRecord.setSymbol(symbol);
            predRecord.setTf(featureRecord.getTf());
            predRecord.setCloseTimeMs(featureRecord.getCloseTimeMs());
            predRecord.setCloseTime(formatMs(featureRecord.getCloseTimeMs()));
            predRecord.setHorizonBars(HORIZON_BARS);
            predRecord.setTpPct(TP_PCT);
            predRecord.setSlPct(SL_PCT);
            predRecord.setDirection(direction);
            predRecord.setPHit(pHit);
            predRecord.setPTrade(P_TRADE);
            predRecord.setLoggedAtMs(System.currentTimeMillis());
            predRecord.setLoggedAt(formatMs(predRecord.getLoggedAtMs()));
            double conf = Math.max(pHit, 1.0d - pHit);
            predRecord.setConfidence(conf);
            double entryPrice = parseDouble(featureRecord.getClosePrice());
            if (entryPrice <= 0.0d) {
                return;
            }
            predRecord.setEntryPrice(entryPrice);
            if ("UP".equalsIgnoreCase(direction)) {
                predRecord.setTpPrice(entryPrice * (1.0d + TP_PCT));
                predRecord.setSlPrice(entryPrice * (1.0d - SL_PCT));
            } else {
                predRecord.setTpPrice(entryPrice * (1.0d - TP_PCT));
                predRecord.setSlPrice(entryPrice * (1.0d + SL_PCT));
            }
            predWriter.append(predRecord);
            log.info(
                    "PRED symbol={} closeTimeMs={} direction={} pHit={} pTrade={} conf={}",
                    symbol,
                    featureRecord.getCloseTimeMs(),
                    direction,
                    pHit,
                    P_TRADE,
                    conf,
                    conf);
            symbolState.updatePredIfNewer(symbol, featureRecord.getCloseTimeMs());
            symbolState.setLastPredInfo(symbol, predRecord.getDirection(), pHit);
            enqueuePending(symbol, featureRecord, predRecord);
        } catch (Exception ex) {
            log.info("PRED_SKIP_NO_MODEL symbol={} closeTimeMs={}", symbol, featureRecord.getCloseTimeMs(), ex);
        }
    }

    private void evaluatePendingWithBar(String symbol, RollingFeatureState.Bar previous, RawRecord current) {
        if (previous == null) {
            return;
        }
        PendingTrade pending = pendingBySymbol.get(symbol);
        if (pending == null) {
            return;
        }
        long expectedGap = properties.getExpectedGapMs() == null ? 300_000L : properties.getExpectedGapMs();
        long gapMs = current.getCloseTimeMs() - previous.getCloseTimeMs();
        if (gapMs != expectedGap) {
            pendingBySymbol.remove(symbol);
            return;
        }
        double barHigh = parseDouble(current.getHighPrice());
        double barLow = parseDouble(current.getLowPrice());
        if (barHigh == 0.0d || barLow == 0.0d) {
            return;
        }
        boolean hitTp;
        boolean hitSl;
        if ("UP".equalsIgnoreCase(pending.getDirection())) {
            hitTp = barHigh >= pending.getTpPrice();
            hitSl = barLow <= pending.getSlPrice();
        } else {
            hitTp = barLow <= pending.getTpPrice();
            hitSl = barHigh >= pending.getSlPrice();
        }
        if (!(hitTp || hitSl)) {
            pending.decrementBarsLeft();
            if (pending.getBarsLeft() <= 0) {
                pendingBySymbol.remove(symbol);
            }
            return;
        }
        String event = hitSl ? "SL_HIT" : "TP_HIT";
        String result = hitSl ? "NOK" : "OK";
        if (hitTp && hitSl) {
            event = "SL_HIT_SAME_BAR";
            result = "NOK";
        }
        EvalRecord evalRecord = new EvalRecord();
        evalRecord.setType("EVAL");
        evalRecord.setSymbol(symbol);
        evalRecord.setTf(properties.getTf());
        evalRecord.setPredCloseTimeMs(pending.getPredCloseTimeMs());
        evalRecord.setEventCloseTimeMs(current.getCloseTimeMs());
        evalRecord.setDirection(pending.getDirection());
        evalRecord.setEntryPrice(pending.getEntryPrice());
        evalRecord.setTpPrice(pending.getTpPrice());
        evalRecord.setSlPrice(pending.getSlPrice());
        evalRecord.setResult(result);
        evalRecord.setEvent(event);
        evalRecord.setLoggedAtMs(System.currentTimeMillis());
        evalRecord.setLoggedAt(formatMs(evalRecord.getLoggedAtMs()));
        try {
            predWriter.append(evalRecord);
            log.info("EVAL symbol={} predCloseTimeMs={} result={} event={}",
                    symbol,
                    pending.getPredCloseTimeMs(),
                    result,
                    event);
        } catch (Exception ex) {
            log.info("PRED_SKIP_NO_MODEL symbol={} closeTimeMs={}", symbol, current.getCloseTimeMs(), ex);
        }
        pendingBySymbol.remove(symbol);
    }

    private void enqueuePending(String symbol, FeatureRecord featureRecord, PredRecord predRecord) {
        double entryPrice = predRecord.getEntryPrice() == null ? 0.0d : predRecord.getEntryPrice();
        Double tpPrice = predRecord.getTpPrice();
        Double slPrice = predRecord.getSlPrice();
        if (entryPrice <= 0.0d || tpPrice == null || slPrice == null) {
            return;
        }
        PendingTrade pending = new PendingTrade(
                featureRecord.getCloseTimeMs(),
                entryPrice,
                predRecord.getDirection(),
                tpPrice,
                slPrice,
                HORIZON_BARS
        );
        pendingBySymbol.put(symbol, pending);
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

    private static final class PendingTrade {
        private final long predCloseTimeMs;
        private final double entryPrice;
        private final String direction;
        private final double tpPrice;
        private final double slPrice;
        private int barsLeft;

        private PendingTrade(
                long predCloseTimeMs,
                double entryPrice,
                String direction,
                double tpPrice,
                double slPrice,
                int barsLeft
        ) {
            this.predCloseTimeMs = predCloseTimeMs;
            this.entryPrice = entryPrice;
            this.direction = direction;
            this.tpPrice = tpPrice;
            this.slPrice = slPrice;
            this.barsLeft = barsLeft;
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

        private int getBarsLeft() {
            return barsLeft;
        }

        private void decrementBarsLeft() {
            barsLeft -= 1;
        }
    }
}
