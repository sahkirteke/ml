package com.ml.raw;

import org.springframework.boot.ApplicationArguments;
import org.springframework.boot.ApplicationRunner;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;

@Component
@Order(1)
public class WarmupRunner implements ApplicationRunner {

    private final RawWarmupLoader warmupLoader;

    public WarmupRunner(RawWarmupLoader warmupLoader) {
        this.warmupLoader = warmupLoader;
    }

    @Override
    public void run(ApplicationArguments args) {
        warmupLoader.warmup();
    }
}
