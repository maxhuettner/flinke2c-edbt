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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.List;

/**
 * q1_flinke2c_parallel.sql as a DataStream job with ImputationFunction offloaded to the CEPless
 * operator {@code imputation} (KNN imputation of missing prices). 1:1, nothing is dropped.
 *
 * <p>Args: --bids.host --bids.port --sink.host --sink.port [--operator]
 */
public class Q1CeplessImputationJob {

	private static final Logger LOG = LoggerFactory.getLogger(Q1CeplessImputationJob.class);

	public static void main(String[] args) throws Exception {
		ParameterTool params = ParameterTool.fromArgs(args);

		String bidsHost = params.getRequired("bids.host");
		int bidsPort = params.getInt("bids.port");
		String sinkHost = params.getRequired("sink.host");
		int sinkPort = params.getInt("sink.port");
		String operatorName = params.get("operator", "imputation");

		StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
		env.getConfig().setGlobalJobParameters(params);
		// same session settings as q1_flinke2c_parallel.sql
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

		// auction,bidder,price,channel,url,dateTime,extra,latency_ts joined by SEP
		// (channel/url are used by the KNN distance), NULL price -> empty field
		DataStream<String> rows = bids.map(Q1CeplessImputationJob::toCsv).name("to-imputation-row").returns(String.class);

		// same row shape back, only price possibly imputed
		DataStream<String> imputed = rows.serverless(operatorName).name("cepless-" + operatorName);

		// drop malformed responses (e.g. stray empty strings) so fromCsv doesn't crash the job
		DataStream<String> wellFormed = imputed
				.filter(row -> row.split(SEP, 8).length == 8)
				.name("drop-malformed-cepless-response");

		DataStream<RowData> sinkRows = wellFormed.map(Q1CeplessImputationJob::fromCsv).name("from-csv-row").returns(RowData.class);

		sinkRows.addSink(sink).name("nexmark_q1-tcp-sink");

		env.execute("nexmark_q1 (CEPless Imputation)");
	}

	// SOH instead of ',' since channel/url/extra can contain commas
	private static final String SEP = "";

	private static final java.util.concurrent.atomic.AtomicInteger TO_CSV_SAMPLE_COUNT =
			new java.util.concurrent.atomic.AtomicInteger();

	private static String toCsv(RowData bid) {
		long auction = bid.getLong(0);
		long bidder = bid.getLong(1);
		String price = bid.isNullAt(2) ? "" : Long.toString(bid.getLong(2));
		String channel = bid.isNullAt(3) ? "" : bid.getString(3).toString();
		String url = bid.isNullAt(4) ? "" : bid.getString(4).toString();
		long dateTimeMillis = bid.isNullAt(5) ? 0L : bid.getTimestamp(5, 3).getMillisecond();
		String extra = bid.isNullAt(6) ? "" : bid.getString(6).toString();
		long latencyTs = bid.isNullAt(7) ? 0L : bid.getLong(7);
		String result = auction + SEP + bidder + SEP + price + SEP + channel + SEP + url + SEP
				+ dateTimeMillis + SEP + extra + SEP + latencyTs;
		if (TO_CSV_SAMPLE_COUNT.getAndIncrement() < 10) {
			LOG.info("toCsv sample: raw={} length={}", escapeForLog(result), result.length());
		}
		return result;
	}

	private static RowData fromCsv(String csv) {
		String[] parts = csv.split(SEP, 8);
		try {
			long auction = Long.parseLong(parts[0]);
			long bidder = Long.parseLong(parts[1]);
			// operator always returns a price, so a parse failure here is a real error
			BigDecimal price = new BigDecimal(parts[2]).setScale(3, RoundingMode.HALF_UP);
			// skip channel (3) and url (4)
			long dateTimeMillis = Long.parseLong(parts[5]);
			String extra = parts[6];
			long latencyTs = Long.parseLong(parts[7]);

			GenericRowData row = new GenericRowData(6);
			row.setRowKind(RowKind.INSERT);
			row.setField(0, auction);
			row.setField(1, bidder);
			row.setField(2, DecimalData.fromBigDecimal(price, 23, 3));
			row.setField(3, TimestampData.fromEpochMillis(dateTimeMillis));
			row.setField(4, StringData.fromString(extra));
			row.setField(5, latencyTs);
			return row;
		} catch (RuntimeException e) {
			LOG.error(
					"fromCsv failed to parse a CEPless response. raw={} length={} parts.length={} parts={}",
					escapeForLog(csv), csv.length(), parts.length, java.util.Arrays.toString(escapeForLog(parts)));
			throw e;
		}
	}

	private static String escapeForLog(String s) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (c < 0x20 || c > 0x7e) {
				sb.append(String.format("\\u%04x", (int) c));
			} else {
				sb.append(c);
			}
		}
		return sb.toString();
	}

	private static String[] escapeForLog(String[] parts) {
		String[] out = new String[parts.length];
		for (int i = 0; i < parts.length; i++) {
			out[i] = escapeForLog(parts[i]);
		}
		return out;
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
