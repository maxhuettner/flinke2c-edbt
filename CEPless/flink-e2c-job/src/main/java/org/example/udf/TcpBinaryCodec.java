package org.example.udf;

import org.apache.flink.table.data.GenericRowData;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.data.StringData;
import org.apache.flink.table.data.TimestampData;
import org.apache.flink.table.types.logical.LogicalType;
import org.apache.flink.table.types.logical.LogicalTypeRoot;
import org.apache.flink.table.types.logical.RowType;
import org.apache.flink.types.RowKind;

import java.io.IOException;
import java.util.List;

/**
 * Ultra-simple binary codec - no null bitmap, no metadata
 * Just raw values in schema order for maximum performance
 */
final class TcpBinaryCodec {

    static final class Encoder {
        private final FieldInfo[] fields;
        private final int fieldCount;
        private byte[] buffer;

        Encoder(RowType rowType) {
            this.fields = buildFieldInfo(rowType);
            this.fieldCount = fields.length;
            this.buffer = new byte[1024];
        }

        byte[] getBuffer() {
            return buffer;
        }

        int encode(RowData row) throws IOException {
            // Write null bitmap first
            int nullBitmapSize = (fieldCount + 7) / 8;
            ensureCapacity(nullBitmapSize);

            // Initialize null bitmap to 0
            for (int i = 0; i < nullBitmapSize; i++) {
                buffer[i] = 0;
            }

            int pos = nullBitmapSize;

            for (int i = 0; i < fieldCount; i++) {
                FieldInfo info = fields[i];

                // Check if field is null
                if (row.isNullAt(i)) {
                    // Set bit in null bitmap (bit = 1 means null)
                    int bytePos = i / 8;
                    int bitPos = i % 8;
                    buffer[bytePos] |= (1 << bitPos);
                    continue; // Skip encoding this field
                }

                switch (info.type) {
                    case BIGINT:
                        ensureCapacity(pos + 8);
                        writeLongBE(buffer, pos, row.getLong(i));
                        pos += 8;
                        break;

                    case DECIMAL: {
                        // Encode DECIMAL as BIGINT (multiply by 10^precision to preserve decimals)
                        ensureCapacity(pos + 8);
                        long unscaledValue = row.getDecimal(i, info.precision, info.scale).toBigDecimal().unscaledValue().longValue();
                        writeLongBE(buffer, pos, unscaledValue);
                        pos += 8;
                        break;
                    }

                    case VARCHAR: {
                        byte[] bytes = row.getString(i).toBytes();
                        ensureCapacity(pos + 4 + bytes.length);
                        writeIntBE(buffer, pos, bytes.length);
                        pos += 4;
                        System.arraycopy(bytes, 0, buffer, pos, bytes.length);
                        pos += bytes.length;
                        break;
                    }

                    case TIMESTAMP_WITHOUT_TIME_ZONE:
                    case TIMESTAMP_WITH_LOCAL_TIME_ZONE: {
                        ensureCapacity(pos + 8);
                        writeLongBE(buffer, pos, row.getTimestamp(i, 3).getMillisecond());
                        pos += 8;
                        break;
                    }

                    default:
                        throw new IOException("Unsupported type: " + info.type);
                }
            }

            return pos;
        }

        private void ensureCapacity(int required) {
            if (required <= buffer.length) {
                return;
            }
            int newSize = buffer.length;
            while (newSize < required) {
                newSize *= 2;
            }
            buffer = java.util.Arrays.copyOf(buffer, newSize);
        }
    }

    static final class Decoder {
        private final FieldInfo[] fields;
        private final int fieldCount;

        Decoder(RowType rowType) {
            this.fields = buildFieldInfo(rowType);
            this.fieldCount = fields.length;
        }

        RowData decode(byte[] payload, int length) throws IOException {
            // Read null bitmap first
            int nullBitmapSize = (fieldCount + 7) / 8;
            require(length, nullBitmapSize);

            int pos = nullBitmapSize;
            GenericRowData row = new GenericRowData(fieldCount);
            row.setRowKind(RowKind.INSERT);

            for (int i = 0; i < fieldCount; i++) {
                FieldInfo info = fields[i];

                // Check if field is null from bitmap
                int bytePos = i / 8;
                int bitPos = i % 8;
                boolean isNull = (payload[bytePos] & (1 << bitPos)) != 0;

                if (isNull) {
                    row.setField(i, null);
                    continue; // Skip decoding this field
                }

                switch (info.type) {
                    case BIGINT:
                        require(length, pos + 8);
                        row.setField(i, readLongBE(payload, pos));
                        pos += 8;
                        break;

                    case DECIMAL: {
                        // Decode DECIMAL from BIGINT (unscaled value)
                        require(length, pos + 8);
                        long unscaledValue = readLongBE(payload, pos);
                        row.setField(i, org.apache.flink.table.data.DecimalData.fromBigDecimal(
                            new java.math.BigDecimal(java.math.BigInteger.valueOf(unscaledValue), info.scale),
                            info.precision,
                            info.scale
                        ));
                        pos += 8;
                        break;
                    }

                    case VARCHAR: {
                        int strLen = readLength(payload, length, pos);
                        pos += 4;
                        byte[] bytes = new byte[strLen];
                        System.arraycopy(payload, pos, bytes, 0, strLen);
                        pos += strLen;
                        row.setField(i, StringData.fromBytes(bytes));
                        break;
                    }

                    case TIMESTAMP_WITHOUT_TIME_ZONE:
                    case TIMESTAMP_WITH_LOCAL_TIME_ZONE:
                        require(length, pos + 8);
                        row.setField(i, TimestampData.fromEpochMillis(readLongBE(payload, pos)));
                        pos += 8;
                        break;

                    default:
                        throw new IOException("Unsupported type: " + info.type);
                }
            }

            return row;
        }
    }

    private static FieldInfo[] buildFieldInfo(RowType rowType) {
        List<RowType.RowField> fields = rowType.getFields();
        FieldInfo[] infos = new FieldInfo[fields.size()];
        for (int i = 0; i < fields.size(); i++) {
            LogicalType type = fields.get(i).getType();
            int precision = 0;
            int scale = 0;
            if (type.getTypeRoot() == LogicalTypeRoot.DECIMAL) {
                org.apache.flink.table.types.logical.DecimalType decimalType =
                    (org.apache.flink.table.types.logical.DecimalType) type;
                precision = decimalType.getPrecision();
                scale = decimalType.getScale();
            }
            infos[i] = new FieldInfo(type.getTypeRoot(), precision, scale);
        }
        return infos;
    }

    private static int readLength(byte[] payload, int length, int pos) throws IOException {
        require(length, pos + 4);
        int len = readIntBE(payload, pos);
        if (len < 0 || pos + 4 + len > length) {
            throw new IOException("Invalid length: " + len);
        }
        return len;
    }

    private static void require(int length, int required) throws IOException {
        if (required > length) {
            throw new IOException("Truncated frame: need " + required + " but only have " + length);
        }
    }

    private static void writeIntBE(byte[] buffer, int pos, int value) {
        buffer[pos] = (byte) (value >>> 24);
        buffer[pos + 1] = (byte) (value >>> 16);
        buffer[pos + 2] = (byte) (value >>> 8);
        buffer[pos + 3] = (byte) value;
    }

    private static void writeLongBE(byte[] buffer, int pos, long value) {
        buffer[pos] = (byte) (value >>> 56);
        buffer[pos + 1] = (byte) (value >>> 48);
        buffer[pos + 2] = (byte) (value >>> 40);
        buffer[pos + 3] = (byte) (value >>> 32);
        buffer[pos + 4] = (byte) (value >>> 24);
        buffer[pos + 5] = (byte) (value >>> 16);
        buffer[pos + 6] = (byte) (value >>> 8);
        buffer[pos + 7] = (byte) value;
    }

    private static int readIntBE(byte[] buffer, int pos) {
        return ((buffer[pos] & 0xFF) << 24)
            | ((buffer[pos + 1] & 0xFF) << 16)
            | ((buffer[pos + 2] & 0xFF) << 8)
            | (buffer[pos + 3] & 0xFF);
    }

    private static long readLongBE(byte[] buffer, int pos) {
        return ((long) (buffer[pos] & 0xFF) << 56)
            | ((long) (buffer[pos + 1] & 0xFF) << 48)
            | ((long) (buffer[pos + 2] & 0xFF) << 40)
            | ((long) (buffer[pos + 3] & 0xFF) << 32)
            | ((long) (buffer[pos + 4] & 0xFF) << 24)
            | ((long) (buffer[pos + 5] & 0xFF) << 16)
            | ((long) (buffer[pos + 6] & 0xFF) << 8)
            | ((long) (buffer[pos + 7] & 0xFF));
    }

    private static final class FieldInfo {
        private final LogicalTypeRoot type;
        private final int precision;
        private final int scale;

        private FieldInfo(LogicalTypeRoot type, int precision, int scale) {
            this.type = type;
            this.precision = precision;
            this.scale = scale;
        }
    }
}
