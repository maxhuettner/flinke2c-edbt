package operator;

import java.math.BigDecimal;
import java.math.RoundingMode;

/**
 * CEPless version of CurrencyConversionFunction: auction,bidder,price,dateTime,extra,latency_ts
 * in and out, price * 0.908. Empty price stays empty.
 */
public class Operator implements OperatorProcessingInterface {

	private static final BigDecimal CONVERSION_FACTOR = new BigDecimal("0.908");

	// SOH, fields can contain commas
	private static final String SEP = "";

	@Override
	public String process(String item) {
		try {
			String[] parts = item.split(SEP, 6);
			long auction = Long.parseLong(parts[0]);
			long bidder = Long.parseLong(parts[1]);
			String priceField = parts[2];
			String price = priceField.isEmpty()
					? ""
					: new BigDecimal(priceField).multiply(CONVERSION_FACTOR).setScale(3, RoundingMode.HALF_UP).toString();
			long dateTimeMillis = Long.parseLong(parts[3]);
			String extra = parts[4];
			String latencyTs = parts[5];

			return auction + SEP + bidder + SEP + price + SEP + dateTimeMillis + SEP + extra + SEP + latencyTs;
		} catch (Exception e) {
			// malformed input, return as is (job drops it)
			System.out.println("Wire-format error processing event '" + item + "': " + e);
			return item;
		}
	}
}
