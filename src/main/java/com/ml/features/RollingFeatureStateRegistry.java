package com.ml.features;

import com.ml.config.RawIngestionProperties;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.springframework.stereotype.Component;

@Component
public class RollingFeatureStateRegistry {

    private final Map<String, RollingFeatureState> states = new ConcurrentHashMap<>();
    private final RawIngestionProperties properties;

    public RollingFeatureStateRegistry(RawIngestionProperties properties) {
        this.properties = properties;
    }

    public RollingFeatureState getOrCreate(String symbol) {
        int maxBars = properties.getWarmupBars();
        return states.computeIfAbsent(symbol, key -> new RollingFeatureState(key, maxBars));
    }
}
