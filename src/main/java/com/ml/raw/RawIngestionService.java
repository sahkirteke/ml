package com.ml.raw;

import com.ml.ws.BinanceWsClient;
import com.ml.ws.KlineEvent;
import com.ml.ws.KlinePayload;
import com.ml.ws.WsEnvelope;
import jakarta.annotation.PostConstruct;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

@Service
public class RawIngestionService {

    private static final Logger log = LoggerFactory.getLogger(RawIngestionService.class);

    private final BinanceWsClient wsClient;
    private final RawRecordBuilder recordBuilder;
    private final GzipJsonlAppender appender;
    private final SymbolState symbolState;

    public RawIngestionService(
            BinanceWsClient wsClient,
            RawRecordBuilder recordBuilder,
            GzipJsonlAppender appender,
            SymbolState symbolState
    ) {
        this.wsClient = wsClient;
        this.recordBuilder = recordBuilder;
        this.appender = appender;
        this.symbolState = symbolState;
    }

    @PostConstruct
    public void start() {
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
            } catch (Exception ex) {
                throw new RuntimeException(ex);
            }
        });
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
