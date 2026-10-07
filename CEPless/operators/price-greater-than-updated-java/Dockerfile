FROM eclipse-temurin:8-jre

# THRESHOLD of the updated PriceGreaterThan; operator updates can't pass env vars
ENV PRICE_THRESHOLD=100000

COPY target/java-example-operator-1.0-SNAPSHOT-jar-with-dependencies.jar /app.jar

ENTRYPOINT ["java", "-jar", "/app.jar"]