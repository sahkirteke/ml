package com.ml.features;

import com.ml.raw.DailyPartitionResolver;
import com.ml.raw.GzipJsonlAppender;
import java.nio.file.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class FeatureWriter {

    private static final Logger log = LoggerFactory.getLogger(FeatureWriter.class);

    private final GzipJsonlAppender appender;
    private final DailyPartitionResolver partitionResolver;

    public FeatureWriter(GzipJsonlAppender appender, DailyPartitionResolver partitionResolver) {
        this.appender = appender;
        this.partitionResolver = partitionResolver;
    }

    public Path append(FeatureRecord record) {
        if (record == null) {
            return null;
        }
        try {
            Path file = appender.appendFeature(record);
            String partition = partitionResolver.resolveDate(record.getCloseTimeMs());
            log.debug("FEATURE_WRITE symbol={} closeTimeMs={} partition={} windowReady={}",
                    record.getSymbol(),
                    record.getCloseTimeMs(),
                    partition,
                    record.isWindowReady());
            return file;
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }
}
