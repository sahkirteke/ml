package com.ml.raw;

import com.ml.config.RawIngestionProperties;
import com.ml.ws.KlineEvent;
import com.ml.ws.KlinePayload;
import java.math.BigDecimal;
import java.math.RoundingMode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class RawRecordBuilder {

    private static final Logger log = LoggerFactory.getLogger(RawRecordBuilder.class);

    private final RawIngestionProperties properties;

    public RawRecordBuilder(RawIngestionProperties properties) {
        this.properties = properties;
    }

    public RawRecord build(KlineEvent event, long receivedAtMs) {
        if (event == null || event.getKline() == null) {
            return null;
        }
        KlinePayload kline = event.getKline();
        if (event.getSymbol() == null || kline.getCloseTime() == null || kline.getOpenTime() == null) {
            return null;
        }
        RawRecord record = new RawRecord();
        record.setSymbol(event.getSymbol().toUpperCase());
        record.setTf(properties.getTf());
        record.setEventTimeMs(safeLong(event.getEventTime()));
        record.setOpenTimeMs(kline.getOpenTime());
        record.setCloseTimeMs(kline.getCloseTime());
        record.setOpenPrice(kline.getOpenPrice());
        record.setHighPrice(kline.getHighPrice());
        record.setLowPrice(kline.getLowPrice());
        record.setClosePrice(kline.getClosePrice());
        record.setVolume(kline.getVolume());
        record.setQuoteVolume(kline.getQuoteVolume());
        record.setTradeCount(safeLong(kline.getTradeCount()));
        record.setTakerBuyBaseVol(kline.getTakerBuyBaseVol());
        record.setTakerBuyQuoteVol(kline.getTakerBuyQuoteVol());
        record.setFinal(Boolean.TRUE.equals(kline.getIsFinal()));
        record.setReceivedAtMs(receivedAtMs);
        applyDerived(record);
        return record;
    }

    private void applyDerived(RawRecord record) {
        try {
            BigDecimal volume = toDecimal(record.getVolume());
            BigDecimal takerBuy = toDecimal(record.getTakerBuyBaseVol());
            BigDecimal eps = properties.getEps();
            BigDecimal sellBase = volume.subtract(takerBuy);
            if (sellBase.signum() < 0) {
                sellBase = BigDecimal.ZERO;
            }
            BigDecimal denom = sellBase.compareTo(eps) < 0 ? eps : sellBase;
            BigDecimal ratio = takerBuy.divide(denom, 18, RoundingMode.HALF_UP);
            BigDecimal delta = takerBuy.subtract(sellBase);
            record.setSellBaseVol(sellBase.toPlainString());
            record.setBuySellRatio(ratio.toPlainString());
            record.setDeltaBaseVol(delta.toPlainString());
        } catch (Exception ex) {
            log.warn("RAW_DERIVE_FAIL symbol={} closeTimeMs={}", record.getSymbol(), record.getCloseTimeMs(), ex);
            record.setSellBaseVol("0");
            record.setBuySellRatio("0");
            record.setDeltaBaseVol("0");
        }
    }

    private BigDecimal toDecimal(String value) {
        if (value == null || value.isBlank()) {
            return BigDecimal.ZERO;
        }
        return new BigDecimal(value);
    }

    private long safeLong(Long value) {
        return value == null ? 0L : value;
    }
}
