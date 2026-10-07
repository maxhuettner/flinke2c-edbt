package org.example.udf;

import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.streaming.api.functions.sink.legacy.RichSinkFunction;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.types.logical.RowType;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.Socket;

/** BinaryTcpSink from my-flink-udfs without the DynamicTableSink wrapper, for addSink(...). */
public class BinaryTcpSink extends RichSinkFunction<RowData> {
	private static final long serialVersionUID = 1L;
	private static final int BUFFER_SIZE = 256 * 1024;

	private final String address;
	private final int port;
	private final RowType rowType;

	private transient DataOutputStream out;
	private transient TcpBinaryCodec.Encoder encoder;
	private transient Socket socket;

	public BinaryTcpSink(String address, int port, RowType rowType) {
		this.address = address;
		this.port = port;
		this.rowType = rowType;
	}

	@Override
	public void open(OpenContext openContext) throws Exception {
		this.encoder = new TcpBinaryCodec.Encoder(rowType);
		this.socket = new Socket(address, port);
		this.socket.setTcpNoDelay(true);
		this.socket.setSendBufferSize(BUFFER_SIZE);
		this.out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream(), BUFFER_SIZE));
	}

	@Override
	public void invoke(RowData row, Context context) throws IOException {
		int len = encoder.encode(row);
		byte[] buffer = encoder.getBuffer();
		out.writeInt(len);
		out.write(buffer, 0, len);
	}

	@Override
	public void close() throws Exception {
		if (out != null) {
			out.flush();
			out.close();
			out = null;
		}
		if (socket != null) {
			socket.close();
			socket = null;
		}
	}
}
