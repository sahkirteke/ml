package com.ml.features;

import com.ml.raw.RawRecord;
import java.util.ArrayList;
import java.util.List;

public class RollingFeatureState {

    private final String symbol;
    private final int maxBars;
    private final List<Bar> bars = new ArrayList<>();

    public RollingFeatureState(String symbol, int maxBars) {
        this.symbol = symbol;
        this.maxBars = maxBars;
    }

    public String getSymbol() {
        return symbol;
    }

    public int size() {
        return bars.size();
    }

    public List<Bar> getBars() {
        return bars;
    }

    public void add(RawRecord record) {
        if (record == null) {
            return;
        }
        Bar bar = new Bar(
                record.getOpenTimeMs(),
                record.getCloseTimeMs(),
                record.getEventTimeMs(),
                parseDouble(record.getOpenPrice()),
                parseDouble(record.getHighPrice()),
                parseDouble(record.getLowPrice()),
                parseDouble(record.getClosePrice()),
                record.getClosePrice(),
                parseDouble(record.getVolume()),
                record.getTradeCount(),
                parseDouble(record.getBuySellRatio()),
                parseDouble(record.getDeltaBaseVol())
        );
        bars.add(bar);
        if (bars.size() > maxBars) {
            bars.remove(0);
        }
    }

    public Bar getLatest() {
        if (bars.isEmpty()) {
            return null;
        }
        return bars.get(bars.size() - 1);
    }

    public Bar getPrevious() {
        if (bars.size() < 2) {
            return null;
        }
        return bars.get(bars.size() - 2);
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

    public static final class Bar {
        private final long openTimeMs;
        private final long closeTimeMs;
        private final long eventTimeMs;
        private final double open;
        private final double high;
        private final double low;
        private final double close;
        private final String closePrice;
        private final double volume;
        private final long tradeCount;
        private final double buySellRatio;
        private final double deltaBaseVol;

        public Bar(long openTimeMs,
                   long closeTimeMs,
                   long eventTimeMs,
                   double open,
                   double high,
                   double low,
                   double close,
                   String closePrice,
                   double volume,
                   long tradeCount,
                   double buySellRatio,
                   double deltaBaseVol) {
            this.openTimeMs = openTimeMs;
            this.closeTimeMs = closeTimeMs;
            this.eventTimeMs = eventTimeMs;
            this.open = open;
            this.high = high;
            this.low = low;
            this.close = close;
            this.closePrice = closePrice;
            this.volume = volume;
            this.tradeCount = tradeCount;
            this.buySellRatio = buySellRatio;
            this.deltaBaseVol = deltaBaseVol;
        }

        public long getOpenTimeMs() {
            return openTimeMs;
        }

        public long getCloseTimeMs() {
            return closeTimeMs;
        }

        public long getEventTimeMs() {
            return eventTimeMs;
        }

        public double getOpen() {
            return open;
        }

        public double getHigh() {
            return high;
        }

        public double getLow() {
            return low;
        }

        public double getClose() {
            return close;
        }

        public String getClosePrice() {
            return closePrice;
        }

        public double getVolume() {
            return volume;
        }

        public long getTradeCount() {
            return tradeCount;
        }

        public double getBuySellRatio() {
            return buySellRatio;
        }

        public double getDeltaBaseVol() {
            return deltaBaseVol;
        }
    }
}
