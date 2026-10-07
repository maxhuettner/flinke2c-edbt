package operator;

import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.charset.StandardCharsets;

/**
 * CEPless version of ImputationFunction: auction,bidder,price,channel,url,dateTime,extra,latency_ts
 * in and out. Missing prices are replaced with a KNN estimate from recent bids.
 * History is in-process only (not checkpointed, not shared between instances).
 */
public class Operator implements OperatorProcessingInterface {

	private static final int HISTORY_SIZE = 5_000;
	private static final int SEARCH_LIMIT = 512;
	private static final int K = 10;

	private static final double EPS = 1e-6;
	private static final double W_BIDDER = 0.25;
	private static final double W_TIME = 1.0;
	private static final double W_STR = 0.25;

	private static final BigDecimal DEFAULT_PRICE = new BigDecimal("0.000");

	// SOH, fields can contain commas
	private static final String SEP = "";

	private final BoundedRing history = new BoundedRing(HISTORY_SIZE);

	@Override
	public String process(String item) {
		try {
			String[] parts = item.split(SEP, 8);
			long auction = Long.parseLong(parts[0]);
			long bidder = Long.parseLong(parts[1]);
			String priceField = parts[2];
			String channel = parts[3];
			String url = parts[4];
			long dateTimeMillis = Long.parseLong(parts[5]);
			String extra = parts[6];
			String latencyTs = parts[7];

			double tsSeconds = dateTimeMillis / 1000.0;
			int channelHash = hashOrZero(channel);
			int urlHash = hashOrZero(url);
			int extraHash = hashOrZero(extra);

			BigDecimal price;
			if (!priceField.isEmpty()) {
				double priceDouble = Double.parseDouble(priceField);
				history.add(new Obs(bidder, tsSeconds, channelHash, urlHash, extraHash, priceDouble));
				price = new BigDecimal(priceField);
			} else {
				double imputed = knnImputePrice(bidder, tsSeconds, channelHash, urlHash, extraHash);
				price = Double.isNaN(imputed)
						? DEFAULT_PRICE
						: BigDecimal.valueOf(imputed).setScale(3, RoundingMode.HALF_UP);
			}

			return auction + SEP + bidder + SEP + price + SEP + channel + SEP + url + SEP
					+ dateTimeMillis + SEP + extra + SEP + latencyTs;
		} catch (Exception e) {
			// malformed input, return as is (job drops it)
			System.out.println("Wire-format error processing event '" + item + "': " + e);
			return item;
		}
	}

	private double knnImputePrice(long bidderId, double tsSeconds, int channelHash, int urlHash, int extraHash) {
		Obs[] snap = history.snapshotLast(Math.min(SEARCH_LIMIT, HISTORY_SIZE));
		if (snap.length == 0) {
			return Double.NaN;
		}

		double[] bestDist = new double[K];
		double[] bestPrice = new double[K];
		int found = 0;

		for (Obs o : snap) {
			if (o == null) {
				continue;
			}

			double dist = distance(bidderId, tsSeconds, channelHash, urlHash, extraHash, o);

			if (found < K) {
				bestDist[found] = dist;
				bestPrice[found] = o.priceDouble;
				found++;
			} else {
				int worstIdx = 0;
				double worst = bestDist[0];
				for (int j = 1; j < K; j++) {
					if (bestDist[j] > worst) {
						worst = bestDist[j];
						worstIdx = j;
					}
				}
				if (dist < worst) {
					bestDist[worstIdx] = dist;
					bestPrice[worstIdx] = o.priceDouble;
				}
			}
		}

		if (found == 0) {
			return Double.NaN;
		}

		double num = 0.0;
		double den = 0.0;
		for (int i = 0; i < found; i++) {
			double w = 1.0 / (bestDist[i] + EPS);
			num += bestPrice[i] * w;
			den += w;
		}
		return den == 0.0 ? Double.NaN : (num / den);
	}

	private static double distance(long bidderId, double tsSeconds, int channelHash, int urlHash, int extraHash, Obs o) {
		double s = 0.0;
		s += W_BIDDER * (bidderId == o.bidderId ? 0.0 : 1.0);
		double dt = (tsSeconds - o.tsSeconds);
		s += W_TIME * (dt * dt) * 1e-8;
		s += W_STR * (channelHash == o.channelHash ? 0.0 : 1.0);
		s += W_STR * (urlHash == o.urlHash ? 0.0 : 1.0);
		s += W_STR * (extraHash == o.extraHash ? 0.0 : 1.0);
		return s;
	}

	private static int hashOrZero(String s) {
		if (s == null || s.trim().isEmpty()) {
			return 0;
		}
		return murmurLikeHash(s);
	}

	private static int murmurLikeHash(String s) {
		byte[] data = s.getBytes(StandardCharsets.UTF_8);
		int h = 0x9747b28c;
		for (byte b : data) {
			h ^= b;
			h *= 0x5bd1e995;
			h ^= (h >>> 15);
		}
		return h;
	}

	/** A single recorded (non-null-price) bid observation, used as a KNN candidate. */
	private static final class Obs {
		final long bidderId;
		final double tsSeconds;
		final int channelHash;
		final int urlHash;
		final int extraHash;
		final double priceDouble;

		Obs(long bidderId, double tsSeconds, int channelHash, int urlHash, int extraHash, double priceDouble) {
			this.bidderId = bidderId;
			this.tsSeconds = tsSeconds;
			this.channelHash = channelHash;
			this.urlHash = urlHash;
			this.extraHash = extraHash;
			this.priceDouble = priceDouble;
		}
	}

	/** Fixed-capacity ring buffer of the most recent observations, newest-first snapshots. */
	private static final class BoundedRing {
		private final int capacity;
		private final Obs[] buffer;
		private final Object lock = new Object();
		private int start = 0;
		private int size = 0;

		BoundedRing(int capacity) {
			this.capacity = capacity;
			this.buffer = new Obs[capacity];
		}

		void add(Obs obs) {
			synchronized (lock) {
				if (size < capacity) {
					buffer[(start + size) % capacity] = obs;
					size++;
				} else {
					buffer[start] = obs;
					start = (start + 1) % capacity;
				}
			}
		}

		Obs[] snapshotLast(int limit) {
			synchronized (lock) {
				int n = Math.min(size, limit);
				Obs[] out = new Obs[n];
				for (int i = 0; i < n; i++) {
					int idx = (start + size - 1 - i + capacity) % capacity;
					out[i] = buffer[idx];
				}
				return out;
			}
		}
	}
}
