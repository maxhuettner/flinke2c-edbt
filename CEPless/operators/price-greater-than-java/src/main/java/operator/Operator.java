package operator;

import java.math.BigDecimal;

/**
 * CEPless version of PriceGreaterThan. Prefixes each row with "true"/"false" (CEPless operators
 * can't drop events), the job filters on it. PRICE_FIELD_INDEX / PRICE_THRESHOLD env vars override defaults.
 */
public class Operator implements OperatorProcessingInterface {

	// SOH, fields can contain commas
	private static final String SEP = "";

	private static final int PRICE_FIELD_INDEX = envInt("PRICE_FIELD_INDEX", 2);
	private static final BigDecimal THRESHOLD = envDecimal("PRICE_THRESHOLD", "1000");

	@Override
	public String process(String item) {
		try {
			String[] parts = item.split(SEP);
			BigDecimal price = new BigDecimal(parts[PRICE_FIELD_INDEX].trim());
			boolean keep = price.compareTo(THRESHOLD) > 0;
			return (keep ? "true" : "false") + SEP + item;
		} catch (Exception e) {
			System.out.println("Failed to parse price from event '" + item + "': " + e.getMessage());
			return "false" + SEP + item;
		}
	}

	private static int envInt(String name, int defaultValue) {
		String value = System.getenv(name);
		return value == null ? defaultValue : Integer.parseInt(value);
	}

	private static BigDecimal envDecimal(String name, String defaultValue) {
		String value = System.getenv(name);
		return new BigDecimal(value == null ? defaultValue : value);
	}
}
