package com.ml.features;

import com.ml.config.RawIngestionProperties;
import com.ml.features.RollingFeatureState.Bar;
import java.util.List;
import org.springframework.stereotype.Component;

@Component
public class FeatureCalculator {

    private final RawIngestionProperties properties;

    public FeatureCalculator(RawIngestionProperties properties) {
        this.properties = properties;
    }

    public FeatureRecord calculate(RollingFeatureState state) {
        Bar current = state.getLatest();
        if (current == null) {
            return null;
        }
        List<Bar> bars = state.getBars();
        int size = bars.size();
        double close = current.getClose();
        double eps = properties.getEps().doubleValue();

        FeatureRecord record = new FeatureRecord();
        record.setSymbol(state.getSymbol());
        record.setTf(properties.getTf());
        record.setCloseTimeMs(current.getCloseTimeMs());
        record.setClosePrice(close);
        record.setFeaturesVersion(properties.getFeaturesVersion());

        Double ret1 = null;
        Double logRet1 = null;
        if (size >= 2) {
            double prevClose = bars.get(size - 2).getClose();
            if (prevClose != 0.0d) {
                ret1 = close / prevClose - 1.0d;
                logRet1 = Math.log(close / prevClose);
            }
        }
        record.setRet_1(ret1);
        record.setLogRet_1(logRet1);

        record.setRet_3(computeReturn(bars, 3));
        record.setRet_12(computeReturn(bars, 12));
        record.setRealizedVol_6(computeRealizedVol(bars, 6));
        record.setRealizedVol_24(computeRealizedVol(bars, 24));

        double high = current.getHigh();
        double low = current.getLow();
        double open = current.getOpen();
        double range = Math.max(high - low, 0.0d);
        record.setRangePct(range / safeDenom(close, eps));
        record.setBodyPct(Math.abs(close - open) / safeDenom(close, eps));
        record.setUpperWickPct(Math.max(0.0d, high - Math.max(open, close)) / safeDenom(close, eps));
        record.setLowerWickPct(Math.max(0.0d, Math.min(open, close) - low) / safeDenom(close, eps));
        record.setClosePos((close - low) / Math.max(range, eps));

        record.setVolRatio_12(computeRatio(bars, 12, Bar::getVolume));
        record.setTradeRatio_12(computeRatio(bars, 12, bar -> (double) bar.getTradeCount()));
        record.setBuySellRatio(current.getBuySellRatio());
        record.setDeltaVolNorm(current.getDeltaBaseVol() / Math.max(current.getVolume(), eps));

        Double rsi14 = computeRsi(bars, 14);
        Double atr14 = computeAtr(bars, 14);
        Double ema20 = computeEma(bars, 20);
        Double ema200 = computeEma(bars, 200);
        record.setRsi14(rsi14);
        record.setAtr14(atr14);
        record.setEma20DistPct(ema20 == null ? null : (close - ema20) / ema20);
        record.setEma200DistPct(ema200 == null ? null : (close - ema200) / ema200);

        boolean windowReady = size >= 200
                && record.getRet_12() != null
                && record.getRealizedVol_24() != null
                && record.getVolRatio_12() != null
                && record.getTradeRatio_12() != null
                && rsi14 != null
                && atr14 != null;
        record.setWindowReady(windowReady);
        return record;
    }

    private Double computeReturn(List<Bar> bars, int lag) {
        int size = bars.size();
        if (size < lag + 1) {
            return null;
        }
        double current = bars.get(size - 1).getClose();
        double past = bars.get(size - 1 - lag).getClose();
        if (past == 0.0d) {
            return null;
        }
        return current / past - 1.0d;
    }

    private Double computeRealizedVol(List<Bar> bars, int window) {
        int size = bars.size();
        if (size < window + 1) {
            return null;
        }
        double[] logRets = new double[window];
        for (int i = 0; i < window; i++) {
            double close = bars.get(size - window + i).getClose();
            double prev = bars.get(size - window + i - 1).getClose();
            if (prev == 0.0d) {
                return null;
            }
            logRets[i] = Math.log(close / prev);
        }
        double mean = 0.0d;
        for (double value : logRets) {
            mean += value;
        }
        mean /= window;
        double variance = 0.0d;
        for (double value : logRets) {
            double diff = value - mean;
            variance += diff * diff;
        }
        variance /= window;
        return Math.sqrt(variance);
    }

    private Double computeRatio(List<Bar> bars, int window, ToDouble extractor) {
        int size = bars.size();
        if (size < window) {
            return null;
        }
        double sum = 0.0d;
        for (int i = size - window; i < size; i++) {
            sum += extractor.value(bars.get(i));
        }
        double sma = sum / window;
        if (sma == 0.0d) {
            return null;
        }
        double current = extractor.value(bars.get(size - 1));
        return current / sma;
    }

    private Double computeEma(List<Bar> bars, int period) {
        int size = bars.size();
        if (size < period) {
            return null;
        }
        double sum = 0.0d;
        for (int i = 0; i < period; i++) {
            sum += bars.get(i).getClose();
        }
        double ema = sum / period;
        double k = 2.0d / (period + 1.0d);
        for (int i = period; i < size; i++) {
            double close = bars.get(i).getClose();
            ema = (close - ema) * k + ema;
        }
        return ema;
    }

    private Double computeRsi(List<Bar> bars, int period) {
        int size = bars.size();
        if (size < period + 1) {
            return null;
        }
        double gainSum = 0.0d;
        double lossSum = 0.0d;
        for (int i = 1; i <= period; i++) {
            double change = bars.get(i).getClose() - bars.get(i - 1).getClose();
            if (change >= 0.0d) {
                gainSum += change;
            } else {
                lossSum += -change;
            }
        }
        double avgGain = gainSum / period;
        double avgLoss = lossSum / period;
        for (int i = period + 1; i < size; i++) {
            double change = bars.get(i).getClose() - bars.get(i - 1).getClose();
            double gain = Math.max(change, 0.0d);
            double loss = Math.max(-change, 0.0d);
            avgGain = (avgGain * (period - 1) + gain) / period;
            avgLoss = (avgLoss * (period - 1) + loss) / period;
        }
        if (avgLoss == 0.0d) {
            return avgGain == 0.0d ? 0.0d : 100.0d;
        }
        double rs = avgGain / avgLoss;
        return 100.0d - (100.0d / (1.0d + rs));
    }

    private Double computeAtr(List<Bar> bars, int period) {
        int size = bars.size();
        if (size < period + 1) {
            return null;
        }
        double trSum = 0.0d;
        for (int i = 1; i <= period; i++) {
            trSum += trueRange(bars.get(i), bars.get(i - 1).getClose());
        }
        double atr = trSum / period;
        for (int i = period + 1; i < size; i++) {
            double tr = trueRange(bars.get(i), bars.get(i - 1).getClose());
            atr = (atr * (period - 1) + tr) / period;
        }
        return atr;
    }

    private double trueRange(Bar bar, double prevClose) {
        double highLow = bar.getHigh() - bar.getLow();
        double highClose = Math.abs(bar.getHigh() - prevClose);
        double lowClose = Math.abs(bar.getLow() - prevClose);
        return Math.max(highLow, Math.max(highClose, lowClose));
    }

    private double safeDenom(double value, double eps) {
        return Math.abs(value) < eps ? eps : value;
    }

    @FunctionalInterface
    private interface ToDouble {
        double value(Bar bar);
    }
}
