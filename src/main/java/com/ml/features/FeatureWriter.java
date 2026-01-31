package com.ml.features;

import com.ml.raw.GzipJsonlAppender;
import java.nio.file.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class FeatureWriter {

    private static final Logger log = LoggerFactory.getLogger(FeatureWriter.class);

    private final GzipJsonlAppender appender;

    public FeatureWriter(GzipJsonlAppender appender) {
        this.appender = appender;
    }

    public Path append(FeatureRecord record) {
        if (record == null) {
            return null;
        }
        try {
            Path file = appender.appendFeature(record);
            log.info("FEATURE_WRITE symbol={} closeTimeMs={} file={} windowReady={}",
                    record.getSymbol(),
                    record.getCloseTimeMs(),
                    file,
                    record.isWindowReady());
            return file;
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }
}
