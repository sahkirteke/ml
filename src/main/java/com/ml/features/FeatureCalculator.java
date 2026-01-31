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
        FeatureRecord record = new FeatureRecord();
        record.setSymbol(state.getSymbol());
        record.setTf(properties.getTf());
        record.setCloseTimeMs(current.getCloseTimeMs());
        record.setClosePrice(current.getClosePrice());
        record.setFeaturesVersion(properties.getFeaturesVersion());

        Double ret1 = computeReturn(bars, 1);
        Double logRet1 = computeLogReturn(bars, 1);
        Double ret3 = computeReturn(bars, 3);
        Double ret12 = computeReturn(bars, 12);
        Double realizedVol6 = computeRealizedVol(bars, 6);
        Double realizedVol24 = computeRealizedVol(bars, 24);

        double eps = properties.getEps().doubleValue();
        double range = Math.max(current.getHigh() - current.getLow(), 0.0d);
        Double rangePct = range / safeDenom(current.getClose(), eps);
        Double bodyPct = Math.abs(current.getClose() - current.getOpen()) / safeDenom(current.getClose(), eps);
        Double upperWickPct = Math.max(0.0d, current.getHigh() - Math.max(current.getOpen(), current.getClose()))
                / safeDenom(current.getClose(), eps);
        Double lowerWickPct = Math.max(0.0d, Math.min(current.getOpen(), current.getClose()) - current.getLow())
                / safeDenom(current.getClose(), eps);
        Double closePos = (current.getClose() - current.getLow()) / Math.max(range, eps);

        Double volRatio12 = computeRatio(bars, 12, Bar::getVolume);
        Double tradeRatio12 = computeRatio(bars, 12, bar -> (double) bar.getTradeCount());
        Double buySellRatio = current.getBuySellRatio();
        Double deltaVolNorm = current.getDeltaBaseVol() / Math.max(current.getVolume(), eps);

        Double rsi14 = computeRsi(bars, 14);
        Double atr14 = computeAtr(bars, 14);
        Double ema20 = computeEma(bars, 20);
        Double ema200 = computeEma(bars, 200);
        Double ema20Dist = ema20 == null ? null : (current.getClose() - ema20) / ema20;
        Double ema200Dist = ema200 == null ? null : (current.getClose() - ema200) / ema200;

        record.setRet_1(ret1);
        record.setLogRet_1(logRet1);
        record.setRet_3(ret3);
        record.setRet_12(ret12);
        record.setRealizedVol_6(realizedVol6);
        record.setRealizedVol_24(realizedVol24);
        record.setRangePct(rangePct);
        record.setBodyPct(bodyPct);
        record.setUpperWickPct(upperWickPct);
        record.setLowerWickPct(lowerWickPct);
        record.setClosePos(closePos);
        record.setVolRatio_12(volRatio12);
        record.setTradeRatio_12(tradeRatio12);
        record.setBuySellRatio(buySellRatio);
        record.setDeltaVolNorm(deltaVolNorm);
        record.setRsi14(rsi14);
        record.setAtr14(atr14);
        record.setEma20DistPct(ema20Dist);
        record.setEma200DistPct(ema200Dist);

        boolean windowReady = bars.size() >= 200
                && ret12 != null
                && realizedVol24 != null
                && volRatio12 != null
                && tradeRatio12 != null
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

    private Double computeLogReturn(List<Bar> bars, int lag) {
        int size = bars.size();
        if (size < lag + 1) {
            return null;
        }
        double current = bars.get(size - 1).getClose();
        double past = bars.get(size - 1 - lag).getClose();
        if (past == 0.0d || current == 0.0d) {
            return null;
        }
        return Math.log(current / past);
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
