package com.ml.raw;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import org.springframework.stereotype.Component;

@Component
public class SymbolState {

    private final Map<String, AtomicLong> lastCloseBySymbol = new ConcurrentHashMap<>();

    public long getLastCloseTimeMs(String symbol) {
        AtomicLong value = lastCloseBySymbol.get(symbol);
        return value == null ? -1L : value.get();
    }

    public boolean updateIfNewer(String symbol, long closeTimeMs) {
        AtomicLong current = lastCloseBySymbol.computeIfAbsent(symbol, key -> new AtomicLong(-1L));
        while (true) {
            long existing = current.get();
            if (closeTimeMs <= existing) {
                return false;
            }
            if (current.compareAndSet(existing, closeTimeMs)) {
                return true;
            }
        }
    }
}
