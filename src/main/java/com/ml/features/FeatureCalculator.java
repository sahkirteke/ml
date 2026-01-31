package com.ml.features;

import com.ml.config.RawIngestionProperties;
import com.ml.features.RollingFeatureState.Bar;
import java.util.ArrayList;
import java.util.List;
import org.springframework.stereotype.Component;

@Component
public class FeatureCalculator {

    private static final int BB_PERIOD = 20;
    private static final double BB_STD_DEV = 2.0d;
    private static final int EMA_FAST = 12;
    private static final int EMA_SLOW = 26;
    private static final int EMA_SIGNAL = 9;
    private static final int RSI_PERIOD = 9;
    private static final int ADX_PERIOD = 14;
    private static final int VOL_RATIO_PERIOD = 12;

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
        FeatureRecord record = new FeatureRecord();
        record.setSymbol(state.getSymbol());
        record.setOpenTimeMs(current.getOpenTimeMs());
        record.setCloseTimeMs(current.getCloseTimeMs());
        record.setEventTimeMs(current.getEventTimeMs() == 0L ? null : current.getEventTimeMs());
        record.setFeaturesVersion(properties.getFeaturesVersion());

        boolean windowReady = true;
        List<Double> x = new ArrayList<>();

        Double ret1 = computeReturn(bars, 1);
        Double ret3 = computeReturn(bars, 3);
        Double ret6 = computeReturn(bars, 6);
        Double bbPercent = computeBbPercentB(bars);
        Double bbWidth = computeBbWidth(bars);
        Double ema20Dist = computeEmaDist(bars, 20);
        Double rsi9 = computeRsi(bars, RSI_PERIOD);
        Double adx14 = computeAdx(bars, ADX_PERIOD);
        Double macdDelta = computeMacdDelta(bars);
        Double volRatio = computeVolumeRatio(bars, VOL_RATIO_PERIOD);

        windowReady &= ret1 != null && ret3 != null && ret6 != null;
        windowReady &= bbPercent != null && bbWidth != null;
        windowReady &= ema20Dist != null && rsi9 != null && adx14 != null;
        windowReady &= macdDelta != null && volRatio != null;

        x.add(orZero(ret1));
        x.add(orZero(ret3));
        x.add(orZero(ret6));
        x.add(orZero(bbPercent));
        x.add(orZero(bbWidth));
        x.add(orZero(ema20Dist));
        x.add(orZero(rsi9));
        x.add(orZero(adx14));
        x.add(orZero(macdDelta));
        x.add(orZero(volRatio));

        record.setWindowReady(windowReady);
        record.setX(x);
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

    private Double computeBbPercentB(List<Bar> bars) {
        int size = bars.size();
        if (size < BB_PERIOD) {
            return null;
        }
        Stats stats = computeStats(bars, BB_PERIOD);
        if (stats == null || stats.stdDev == 0.0d) {
            return null;
        }
        double close = bars.get(size - 1).getClose();
        double lower = stats.mean - BB_STD_DEV * stats.stdDev;
        double upper = stats.mean + BB_STD_DEV * stats.stdDev;
        double denom = upper - lower;
        if (denom == 0.0d) {
            return null;
        }
        return (close - lower) / denom;
    }

    private Double computeBbWidth(List<Bar> bars) {
        int size = bars.size();
        if (size < BB_PERIOD) {
            return null;
        }
        Stats stats = computeStats(bars, BB_PERIOD);
        if (stats == null || stats.mean == 0.0d) {
            return null;
        }
        double upper = stats.mean + BB_STD_DEV * stats.stdDev;
        double lower = stats.mean - BB_STD_DEV * stats.stdDev;
        return (upper - lower) / stats.mean;
    }

    private Double computeEmaDist(List<Bar> bars, int period) {
        Double ema = computeEma(bars, period);
        if (ema == null || ema == 0.0d) {
            return null;
        }
        double close = bars.get(bars.size() - 1).getClose();
        return (close - ema) / ema;
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

    private Double computeAdx(List<Bar> bars, int period) {
        int size = bars.size();
        if (size < period + 1) {
            return null;
        }
        double[] tr = new double[size - 1];
        double[] plusDm = new double[size - 1];
        double[] minusDm = new double[size - 1];
        for (int i = 1; i < size; i++) {
            Bar current = bars.get(i);
            Bar prev = bars.get(i - 1);
            double upMove = current.getHigh() - prev.getHigh();
            double downMove = prev.getLow() - current.getLow();
            plusDm[i - 1] = (upMove > downMove && upMove > 0) ? upMove : 0.0d;
            minusDm[i - 1] = (downMove > upMove && downMove > 0) ? downMove : 0.0d;
            tr[i - 1] = trueRange(current, prev.getClose());
        }
        double atr = 0.0d;
        double plusDmSum = 0.0d;
        double minusDmSum = 0.0d;
        for (int i = 0; i < period; i++) {
            atr += tr[i];
            plusDmSum += plusDm[i];
            minusDmSum += minusDm[i];
        }
        atr /= period;
        double plusDi = atr == 0.0d ? 0.0d : 100.0d * (plusDmSum / period) / atr;
        double minusDi = atr == 0.0d ? 0.0d : 100.0d * (minusDmSum / period) / atr;
        double dx = (plusDi + minusDi) == 0.0d ? 0.0d : 100.0d * Math.abs(plusDi - minusDi) / (plusDi + minusDi);
        double adx = dx;
        for (int i = period; i < tr.length; i++) {
            atr = (atr * (period - 1) + tr[i]) / period;
            plusDmSum = (plusDmSum * (period - 1) + plusDm[i]) / period;
            minusDmSum = (minusDmSum * (period - 1) + minusDm[i]) / period;
            plusDi = atr == 0.0d ? 0.0d : 100.0d * plusDmSum / atr;
            minusDi = atr == 0.0d ? 0.0d : 100.0d * minusDmSum / atr;
            dx = (plusDi + minusDi) == 0.0d ? 0.0d : 100.0d * Math.abs(plusDi - minusDi) / (plusDi + minusDi);
            adx = (adx * (period - 1) + dx) / period;
        }
        return adx;
    }

    private Double computeMacdDelta(List<Bar> bars) {
        int size = bars.size();
        if (size < EMA_SLOW + EMA_SIGNAL) {
            return null;
        }
        List<Double> fastSeries = computeEmaSeries(bars, EMA_FAST);
        List<Double> slowSeries = computeEmaSeries(bars, EMA_SLOW);
        if (fastSeries.isEmpty() || slowSeries.isEmpty()) {
            return null;
        }
        int offset = EMA_SLOW - EMA_FAST;
        List<Double> macdSeries = new ArrayList<>();
        for (int i = 0; i < slowSeries.size(); i++) {
            macdSeries.add(fastSeries.get(i + offset) - slowSeries.get(i));
        }
        if (macdSeries.size() < EMA_SIGNAL) {
            return null;
        }
        double signal = computeEmaFromSeries(macdSeries, EMA_SIGNAL);
        double macd = macdSeries.get(macdSeries.size() - 1);
        return macd - signal;
    }

    private double computeEmaFromSeries(List<Double> series, int period) {
        double sum = 0.0d;
        for (int i = 0; i < period; i++) {
            sum += series.get(i);
        }
        double ema = sum / period;
        double k = 2.0d / (period + 1.0d);
        for (int i = period; i < series.size(); i++) {
            double value = series.get(i);
            ema = (value - ema) * k + ema;
        }
        return ema;
    }

    private List<Double> computeEmaSeries(List<Bar> bars, int period) {
        int size = bars.size();
        if (size < period) {
            return List.of();
        }
        double sum = 0.0d;
        for (int i = 0; i < period; i++) {
            sum += bars.get(i).getClose();
        }
        double ema = sum / period;
        double k = 2.0d / (period + 1.0d);
        List<Double> series = new ArrayList<>();
        series.add(ema);
        for (int i = period; i < size; i++) {
            double close = bars.get(i).getClose();
            ema = (close - ema) * k + ema;
            series.add(ema);
        }
        return series;
    }

    private Double computeVolumeRatio(List<Bar> bars, int period) {
        int size = bars.size();
        if (size < period) {
            return null;
        }
        double sum = 0.0d;
        for (int i = size - period; i < size; i++) {
            sum += bars.get(i).getVolume();
        }
        double sma = sum / period;
        if (sma == 0.0d) {
            return null;
        }
        double current = bars.get(size - 1).getVolume();
        return current / sma;
    }

    private Stats computeStats(List<Bar> bars, int period) {
        int size = bars.size();
        if (size < period) {
            return null;
        }
        double sum = 0.0d;
        for (int i = size - period; i < size; i++) {
            sum += bars.get(i).getClose();
        }
        double mean = sum / period;
        double variance = 0.0d;
        for (int i = size - period; i < size; i++) {
            double diff = bars.get(i).getClose() - mean;
            variance += diff * diff;
        }
        variance /= period;
        return new Stats(mean, Math.sqrt(variance));
    }

    private double trueRange(Bar bar, double prevClose) {
        double highLow = bar.getHigh() - bar.getLow();
        double highClose = Math.abs(bar.getHigh() - prevClose);
        double lowClose = Math.abs(bar.getLow() - prevClose);
        return Math.max(highLow, Math.max(highClose, lowClose));
    }

    private double orZero(Double value) {
        return value == null ? 0.0d : value;
    }

    private static final class Stats {
        private final double mean;
        private final double stdDev;

        private Stats(double mean, double stdDev) {
            this.mean = mean;
            this.stdDev = stdDev;
        }
    }
}
