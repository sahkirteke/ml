package com.ml.raw;

import com.ml.features.FeatureCalculator;
import com.ml.features.FeatureRecord;
import com.ml.features.FeatureWriter;
import com.ml.features.LabelRecord;
import com.ml.features.LabelWriter;
import com.ml.features.RollingFeatureState;
import com.ml.features.RollingFeatureStateRegistry;
import com.ml.config.RawIngestionProperties;
import com.ml.ws.BinanceWsClient;
import com.ml.ws.KlineEvent;
import com.ml.ws.KlinePayload;
import com.ml.ws.WsEnvelope;
import java.util.Locale;
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

    private final BinanceWsClient wsClient;
    private final RawRecordBuilder recordBuilder;
    private final GzipJsonlAppender appender;
    private final SymbolState symbolState;
    private final RollingFeatureStateRegistry stateRegistry;
    private final FeatureCalculator featureCalculator;
    private final FeatureWriter featureWriter;
    private final LabelWriter labelWriter;
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
        if (!symbolState.updateIfNewer(symbol, closeTime)) {
            long last = symbolState.getLastCloseTimeMs(symbol);
            log.info("RAW_SKIP_DUP symbol={} closeTimeMs={} last={}", symbol, closeTime, last);
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
                LabelRecord labelRecord = buildLabel(state, record);
                if (labelRecord != null
                        && symbolState.updateLabelsIfNewer(symbol, labelRecord.getCloseTimeMs())) {
                    labelWriter.append(labelRecord);
                }
                state.add(record);
                FeatureRecord featureRecord = featureCalculator.calculate(state);
                if (featureRecord != null
                        && symbolState.updateFeaturesIfNewer(symbol, featureRecord.getCloseTimeMs())) {
                    featureWriter.append(featureRecord);
                }
            } catch (Exception ex) {
                throw new RuntimeException(ex);
            }
        });
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
        LabelRecord label = new LabelRecord();
        label.setSymbol(current.getSymbol());
        label.setCloseTimeMs(previous.getCloseTimeMs());
        label.setFutureCloseTimeMs(current.getCloseTimeMs());
        label.setFutureRet_1(futureRet);
        label.setLabelUp(futureRet > 0.0d);
        label.setExpectedGapMs(expectedGap);
        label.setGapMs(gapMs);
        label.setLabelValid(gapMs == expectedGap);
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
}
