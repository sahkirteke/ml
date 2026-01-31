package com.ml.raw;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import org.springframework.stereotype.Component;

@Component
public class SymbolState {

    private final Map<String, AtomicLong> lastRawWrittenCloseBySymbol = new ConcurrentHashMap<>();
    private final Map<String, AtomicLong> lastLabelWrittenCloseBySymbol = new ConcurrentHashMap<>();
    private final Map<String, AtomicLong> prevCloseTimeBySymbol = new ConcurrentHashMap<>();
    private final Map<String, AtomicLong> prevClosePriceBySymbol = new ConcurrentHashMap<>();
    private final Map<String, AtomicLong> lastFeaturesCloseBySymbol = new ConcurrentHashMap<>();
    private final Map<String, AtomicLong> lastPredWrittenCloseBySymbol = new ConcurrentHashMap<>();
    private final Map<String, String> lastPredDecisionBySymbol = new ConcurrentHashMap<>();
    private final Map<String, AtomicLong> lastPredPUpBySymbol = new ConcurrentHashMap<>();

    public long getLastRawWrittenCloseTimeMs(String symbol) {
        AtomicLong value = lastRawWrittenCloseBySymbol.get(symbol);
        return value == null ? -1L : value.get();
    }

    public boolean updateRawIfNewer(String symbol, long closeTimeMs) {
        AtomicLong current = lastRawWrittenCloseBySymbol.computeIfAbsent(symbol, key -> new AtomicLong(-1L));
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
        AtomicLong value = lastLabelWrittenCloseBySymbol.get(symbol);
        return value == null ? -1L : value.get();
    }

    public boolean updateLabelsIfNewer(String symbol, long closeTimeMs) {
        AtomicLong current = lastLabelWrittenCloseBySymbol.computeIfAbsent(symbol, key -> new AtomicLong(-1L));
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

    public long getLastPredCloseTimeMs(String symbol) {
        AtomicLong value = lastPredWrittenCloseBySymbol.get(symbol);
        return value == null ? -1L : value.get();
    }

    public boolean updatePredIfNewer(String symbol, long closeTimeMs) {
        AtomicLong current = lastPredWrittenCloseBySymbol.computeIfAbsent(symbol, key -> new AtomicLong(-1L));
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

    public String getLastPredDecision(String symbol) {
        return lastPredDecisionBySymbol.get(symbol);
    }

    public Double getLastPredPUp(String symbol) {
        AtomicLong value = lastPredPUpBySymbol.get(symbol);
        return value == null ? null : Double.longBitsToDouble(value.get());
    }

    public void setLastPredInfo(String symbol, String decision, double pUp) {
        lastPredDecisionBySymbol.put(symbol, decision);
        lastPredPUpBySymbol.computeIfAbsent(symbol, key -> new AtomicLong(Double.doubleToLongBits(pUp)))
                .set(Double.doubleToLongBits(pUp));
    }

    public long getPrevCloseTimeMs(String symbol) {
        AtomicLong value = prevCloseTimeBySymbol.get(symbol);
        return value == null ? -1L : value.get();
    }

    public double getPrevClosePrice(String symbol) {
        AtomicLong value = prevClosePriceBySymbol.get(symbol);
        return value == null ? 0.0d : Double.longBitsToDouble(value.get());
    }

    public void setPrevClose(String symbol, long closeTimeMs, double closePrice) {
        prevCloseTimeBySymbol.computeIfAbsent(symbol, key -> new AtomicLong(-1L)).set(closeTimeMs);
        prevClosePriceBySymbol.computeIfAbsent(symbol, key -> new AtomicLong(0L))
                .set(Double.doubleToLongBits(closePrice));
    }
}
