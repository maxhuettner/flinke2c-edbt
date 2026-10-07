package org.example.flinke2c;

import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.functions.sink.legacy.SinkFunction;
import org.apache.flink.streaming.api.functions.source.legacy.SourceFunction;
import org.apache.flink.table.data.DecimalData;
import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.StringData;
import org.apache.flink.table.data.TimestampData;
import org.apache.flink.table.types.logical.BigIntType;
import org.apache.flink.table.types.logical.DecimalType;
import org.apache.flink.table.types.logical.RowType;
import org.apache.flink.table.types.logical.RowType.RowField;
import org.apache.flink.table.types.logical.TimestampType;
import org.apache.flink.table.types.logical.VarCharType;
import org.apache.flink.types.RowKind;
import org.apache.flink.util.ParameterTool;
import org.example.udf.BinaryTcpSink;
import org.example.udf.BinaryTcpSourceFunction;

import java.math.BigDecimal;
import java.util.Arrays;
import java.util.List;

/**
 * nexmark_q1 as a DataStream job with PriceGreaterThan offloaded to the CEPless operator
 * {@code price-greater-than}. The operator tags each row "true"/"false", the job filters locally.
 *
 * <p>Args: --bids.host --bids.port --sink.host --sink.port [--operator]
 */
public class Q1CeplessPriceGreaterThanJob {

	public static void main(String[] args) throws Exception {
		ParameterTool params = ParameterTool.fromArgs(args);

		String bidsHost = params.getRequired("bids.host");
		int bidsPort = params.getInt("bids.port");
		String sinkHost = params.getRequired("sink.host");
		int sinkPort = params.getInt("sink.port");
		String operatorName = params.get("operator", "price-greater-than");

		StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
		env.getConfig().setGlobalJobParameters(params);
		// same session settings as the nexmark_q1 SQL
		// 'pipeline.object-reuse' = 'true'
		env.getConfig().enableObjectReuse();
		// 'pipeline.operator-chaining.enabled' = 'false'
		env.disableOperatorChaining();

		RowType bidsRowType = buildBidsRowType();
		RowType sinkRowType = buildSinkRowType();

		SourceFunction<RowData> source =
				new BinaryTcpSourceFunction(bidsHost, bidsPort, bidsRowType);
		SinkFunction<RowData> sink =
				new BinaryTcpSink(sinkHost, sinkPort, sinkRowType);

		DataStream<RowData> bids = env.addSource(source).name("bids-tcp-source");

		// auction,bidder,price,dateTime,extra,latency_ts joined by SEP (price at index 2)
		DataStream<String> rows = bids.map(bid -> toCsv(bid)).name("to-nexmark-q1-row").returns(String.class);

		// CEPless operators can't drop events, so the operator prefixes "true"/"false" and we filter here
		DataStream<String> annotated = rows.serverless(operatorName).name("cepless-" + operatorName);

		DataStream<String> filtered = annotated
				.filter(row -> row.startsWith(TRUE_PREFIX))
				.name("keep-price-greater-than")
				.map(row -> row.substring(TRUE_PREFIX.length()))
				.name("strip-cepless-annotation")
				.returns(String.class);

		DataStream<RowData> sinkRows = filtered.map(Q1CeplessPriceGreaterThanJob::fromCsv).name("from-csv-row").returns(RowData.class);

		sinkRows.addSink(sink).name("nexmark_q1-tcp-sink");

		env.execute("nexmark_q1 (CEPless PriceGreaterThan)");
	}

	// SOH instead of ',' since 'extra' can contain commas
	private static final String SEP = "";
	private static final String TRUE_PREFIX = "true" + SEP;

	private static String toCsv(RowData bid) {
		long auction = bid.getLong(0);
		long bidder = bid.getLong(1);
		// price can be NULL; empty field -> operator returns "false"
		String price = bid.isNullAt(2) ? "" : Long.toString(bid.getLong(2));
		// skip channel (3) and url (4)
		long dateTimeMillis = bid.getTimestamp(5, 3).getMillisecond();
		String extra = bid.getString(6).toString();
		long latencyTs = bid.getLong(7);
		return auction + SEP + bidder + SEP + price + SEP + dateTimeMillis + SEP + extra + SEP + latencyTs;
	}

	private static RowData fromCsv(String csv) {
		String[] parts = csv.split(SEP, 6);
		long auction = Long.parseLong(parts[0]);
		long bidder = Long.parseLong(parts[1]);
		long price = Long.parseLong(parts[2]);
		long dateTimeMillis = Long.parseLong(parts[3]);
		String extra = parts[4];
		long latencyTs = Long.parseLong(parts[5]);

		GenericRowData row = new GenericRowData(6);
		row.setRowKind(RowKind.INSERT);
		row.setField(0, auction);
		row.setField(1, bidder);
		// DECIMAL(23,3) is too wide for fromUnscaledLong, use fromBigDecimal
		row.setField(2, DecimalData.fromBigDecimal(BigDecimal.valueOf(price).setScale(3), 23, 3));
		row.setField(3, TimestampData.fromEpochMillis(dateTimeMillis));
		row.setField(4, StringData.fromString(extra));
		row.setField(5, latencyTs);
		return row;
	}

	private static RowType buildBidsRowType() {
		List<RowField> fields = Arrays.asList(
				new RowField("auction", new BigIntType()),
				new RowField("bidder", new BigIntType()),
				new RowField("price", new BigIntType()),
				new RowField("channel", new VarCharType(VarCharType.MAX_LENGTH)),
				new RowField("url", new VarCharType(VarCharType.MAX_LENGTH)),
				new RowField("dateTime", new TimestampType(3)),
				new RowField("extra", new VarCharType(VarCharType.MAX_LENGTH)),
				new RowField("latency_ts", new BigIntType()));
		return new RowType(fields);
	}

	private static RowType buildSinkRowType() {
		List<RowField> fields = Arrays.asList(
				new RowField("auction", new BigIntType()),
				new RowField("bidder", new BigIntType()),
				new RowField("price", new DecimalType(23, 3)),
				new RowField("dateTime", new TimestampType(3)),
				new RowField("extra", new VarCharType(VarCharType.MAX_LENGTH)),
				new RowField("latency_ts", new BigIntType()));
		return new RowType(fields);
	}
}
