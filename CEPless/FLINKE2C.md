# FlinkE2C comparison jobs

Uses the CEPless integration from the `flinke2c` repo (see its `CEPLESS.md`): a Flink `DataStream`
can send elements to a CEPless operator via `.serverless(...)` / `.serverlessFilter(...)`.

## No drop primitive

CEPless always returns exactly one response per request (`EventManager.process(item)` ->
`operator.process(item)` -> `send(result)`), so an operator can't drop an input. Like
`java-fraud-detection`, a filter prefixes a flag to every row and the Flink job drops rows.

## Queries

### PriceGreaterThan (filter)

`Q1CeplessPriceGreaterThanJob`. `operators/price-greater-than-java/`

Build: `./build.sh <docker-hub-user> price-greater-than`. Run:
`flink run -c org.example.flinke2c.Q1CeplessPriceGreaterThanJob flinke2c-cepless-job-1.0-SNAPSHOT.jar
--bids.host <host> --bids.port <port> --sink.host <host> --sink.port <port> --operator
price-greater-than`.

### Imputation

`Q1CeplessImputationJob`, CEPless equivalent of `q1_flinke2c_parallel.sql` (which offloads to a
Rust runtime with 2 workers, while the CEPless operator here runs with parallelism 1).

`operators/imputation-java/Operator.java` ports `ImputationFunction`'s algorithm.

Row layout adds `channel`/`url` (needed for the distance metric):
`auction,bidder,price,channel,url,dateTime,extra,latency_ts`, `price` empty when `NULL`.

Build: `./build.sh <docker-hub-user> imputation`. Run: same as above with
`-c org.example.flinke2c.Q1CeplessImputationJob --operator imputation`.

### Currency conversion

`Q1CeplessCurrencyConversionJob`, the actual nexmark_q1 query (`price * 0.908`, USD to EUR).

Build: `./build.sh <docker-hub-user> currency-conversion`. Run: same as above with
`-c org.example.flinke2c.Q1CeplessCurrencyConversionJob --operator currency-conversion`.