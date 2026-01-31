package com.ml;

import com.ml.config.RawIngestionProperties;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableScheduling
@EnableConfigurationProperties(RawIngestionProperties.class)
public class MlApplication {

    public static void main(String[] args) {
        SpringApplication.run(MlApplication.class, args);
    }
}
