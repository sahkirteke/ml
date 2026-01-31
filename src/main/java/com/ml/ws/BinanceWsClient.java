package com.ml.ws;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ml.config.RawIngestionProperties;
import java.net.URI;
import java.time.Duration;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.socket.client.ReactorNettyWebSocketClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.publisher.Sinks;
import reactor.util.retry.Retry;

@Component
public class BinanceWsClient {

    private static final Logger log = LoggerFactory.getLogger(BinanceWsClient.class);
    private static final String BASE_URL = "wss://fstream.binance.com";

    private final ReactorNettyWebSocketClient client;
    private final ObjectMapper objectMapper;
    private final RawIngestionProperties properties;

    public BinanceWsClient(ObjectMapper objectMapper, RawIngestionProperties properties) {
        this.client = new ReactorNettyWebSocketClient();
        this.objectMapper = objectMapper;
        this.properties = properties;
    }

    public Flux<WsEnvelope> stream() {
        return Flux.defer(() -> {
                    URI uri = buildUri(properties.getSymbols());
                    Sinks.Many<WsEnvelope> sink = Sinks.many().multicast().onBackpressureBuffer();
                    Mono<Void> session = client.execute(uri, webSocketSession -> webSocketSession.receive()
                            .map(message -> message.getPayloadAsText())
                            .flatMap(this::decode)
                            .doOnNext(env -> sink.emitNext(env, Sinks.EmitFailureHandler.FAIL_FAST))
                            .doOnError(err -> sink.emitError(err, Sinks.EmitFailureHandler.FAIL_FAST))
                            .doOnComplete(() -> sink.emitComplete(Sinks.EmitFailureHandler.FAIL_FAST))
                            .then());
                    session.subscribe();
                    return sink.asFlux();
                })
                .doOnSubscribe(sub -> log.info("WS_CONNECT url={}", buildUri(properties.getSymbols())))
                .retryWhen(Retry.backoff(Long.MAX_VALUE, Duration.ofSeconds(2))
                        .maxBackoff(Duration.ofSeconds(30))
                        .doBeforeRetry(signal -> log.warn("WS_RECONNECT attempt={} reason={}",
                                signal.totalRetries() + 1,
                                signal.failure() == null ? "unknown" : signal.failure().getMessage())));
    }

    private URI buildUri(List<String> symbols) {
        String tf = properties.getTf();
        String streams = symbols.stream()
                .filter(symbol -> symbol != null && !symbol.isBlank())
                .map(symbol -> symbol.toLowerCase(Locale.ROOT) + "@kline_" + tf)
                .collect(Collectors.joining("/"));
        return URI.create(BASE_URL + "/stream?streams=" + streams);
    }

    private Mono<WsEnvelope> decode(String payload) {
        if (payload == null || payload.isBlank()) {
            return Mono.empty();
        }
        try {
            WsEnvelope envelope = objectMapper.readValue(payload, WsEnvelope.class);
            if (envelope == null || envelope.getData() == null || envelope.getData().getKline() == null) {
                return Mono.empty();
            }
            return Mono.just(envelope);
        } catch (Exception ex) {
            log.warn("WS_PARSE_FAIL payload={}", payload, ex);
            return Mono.empty();
        }
    }
}
