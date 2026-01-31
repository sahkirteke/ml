package com.ml.raw;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import org.springframework.stereotype.Component;

@Component
public class SymbolState {

    private final Map<String, AtomicLong> lastCloseBySymbol = new ConcurrentHashMap<>();
    private final Map<String, AtomicLong> lastFeaturesCloseBySymbol = new ConcurrentHashMap<>();
    private final Map<String, AtomicLong> lastLabelsCloseBySymbol = new ConcurrentHashMap<>();

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

    public long getLastFeaturesCloseTimeMs(String symbol) {
        AtomicLong value = lastFeaturesCloseBySymbol.get(symbol);
        return value == null ? -1L : value.get();
    }

    public boolean updateFeaturesIfNewer(String symbol, long closeTimeMs) {
        AtomicLong current = lastFeaturesCloseBySymbol.computeIfAbsent(symbol, key -> new AtomicLong(-1L));
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

    public long getLastLabelsCloseTimeMs(String symbol) {
        AtomicLong value = lastLabelsCloseBySymbol.get(symbol);
        return value == null ? -1L : value.get();
    }

    public boolean updateLabelsIfNewer(String symbol, long closeTimeMs) {
        AtomicLong current = lastLabelsCloseBySymbol.computeIfAbsent(symbol, key -> new AtomicLong(-1L));
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
