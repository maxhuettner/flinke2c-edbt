package org.example.udf;

import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.streaming.api.functions.source.legacy.RichParallelSourceFunction;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.types.logical.RowType;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.IOException;
import java.net.Socket;

/** BinaryTcpSourceFunction from my-flink-udfs without the table source wrapper, for addSource(...). */
public class BinaryTcpSourceFunction extends RichParallelSourceFunction<RowData> {

	private static final int BUFFER_SIZE = 256 * 1024;

	private final String address;
	private final int port;
	private final RowType rowType;

	private transient volatile boolean running;
	private transient Socket socket;
	private transient DataInputStream in;
	private transient TcpBinaryCodec.Decoder decoder;
	private transient byte[] payloadBuffer;

	public BinaryTcpSourceFunction(String address, int port, RowType rowType) {
		this.address = address;
		this.port = port;
		this.rowType = rowType;
	}

	@Override
	public void open(OpenContext openContext) throws Exception {
		this.decoder = new TcpBinaryCodec.Decoder(rowType);
		this.socket = new Socket(address, port);
		this.socket.setTcpNoDelay(true);
		this.socket.setReceiveBufferSize(BUFFER_SIZE);
		this.in = new DataInputStream(new BufferedInputStream(socket.getInputStream(), BUFFER_SIZE));
		this.payloadBuffer = new byte[BUFFER_SIZE];
		this.running = true;
	}

	@Override
	public void run(SourceContext<RowData> ctx) throws Exception {
		try {
			while (running) {
				int length;
				try {
					length = in.readInt();
				} catch (EOFException eof) {
					running = false;
					break;
				} catch (IOException ioe) {
					if (!running) {
						break;
					}
					throw ioe;
				}

				if (length <= 0) {
					continue;
				}

				ensurePayloadCapacity(length);
				in.readFully(payloadBuffer, 0, length);
				RowData row = decoder.decode(payloadBuffer, length);

				synchronized (ctx.getCheckpointLock()) {
					ctx.collect(row);
				}
			}
		} finally {
			closeQuietly();
		}
	}

	private void ensurePayloadCapacity(int length) {
		if (length <= payloadBuffer.length) {
			return;
		}
		int newSize = payloadBuffer.length;
		while (newSize < length) {
			newSize *= 2;
		}
		payloadBuffer = new byte[newSize];
	}

	@Override
	public void cancel() {
		running = false;
		closeQuietly();
	}

	@Override
	public void close() {
		closeQuietly();
	}

	private void closeQuietly() {
		if (in != null) {
			try {
				in.close();
			} catch (IOException ignored) {
				// best effort
			}
			in = null;
		}
		if (socket != null) {
			try {
				socket.close();
			} catch (IOException ignored) {
				// best effort
			}
			socket = null;
		}
	}
}
