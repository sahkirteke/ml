package com.ml.features;

import com.ml.raw.DailyPartitionResolver;
import com.ml.raw.GzipJsonlAppender;
import java.nio.file.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

@Component
public class LabelWriter {

    private static final Logger log = LoggerFactory.getLogger(LabelWriter.class);

    private final GzipJsonlAppender appender;
    private final DailyPartitionResolver partitionResolver;

    public LabelWriter(GzipJsonlAppender appender, DailyPartitionResolver partitionResolver) {
        this.appender = appender;
        this.partitionResolver = partitionResolver;
    }

    public Path append(LabelRecord record) {
        if (record == null) {
            return null;
        }
        try {
            Path file = appender.appendLabel(record);
            String partition = partitionResolver.resolveDate(record.getCloseTimeMs());
            log.debug("LABEL_WRITE symbol={} closeTimeMs={} partition={} labelUp={}",
                    record.getSymbol(),
                    record.getCloseTimeMs(),
                    partition,
                    record.getLabelUp());
            return file;
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }
}
