package com.ml.raw;

import com.ml.config.RawIngestionProperties;
import java.time.Instant;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import org.springframework.stereotype.Component;

@Component
public class DailyPartitionResolver {

    private static final DateTimeFormatter FORMATTER = DateTimeFormatter.BASIC_ISO_DATE;

    private final RawIngestionProperties properties;

    public DailyPartitionResolver(RawIngestionProperties properties) {
        this.properties = properties;
    }

    public String resolveDate(long closeTimeMs) {
        LocalDate date = Instant.ofEpochMilli(closeTimeMs)
                .atZone(properties.getPartitionZone())
                .toLocalDate();
        return FORMATTER.format(date);
    }
}
