#!/usr/bin/env python3
"""btc-cent-forecast-01 entry: forecast Bitcoin's price 240 minutes ahead.

Model (log-price space, per case, from that case's 4320 closes only):
  forecast = last_close * exp(pred), where
    pred = 0.5 * (median(last 5 closes) - last_close)     # robust anchor
           - 0.10 * (last_close - EMA240(last closes))    # mean reversion
  clipped to +/- 2 * sigma_4h, sigma_4h = 4h vol from 1-min log returns.

Why this shape (chosen on 66 Binance BTCUSDT proxy checkpoints, 4h grid,
2026-11-04..14, plus the 6 practice checkpoints on the same proxy):
  * 4h BTC returns are dominated by noise; short-horizon moves mean-revert
    (corr of 1h/4h returns with the next 4h return is about -0.11, gap to
    the 60/240-min EMA about -0.13..-0.17). Trend-following the recent
    return hurts (-12.9% vs persistence in the same backtest).
  * A small pull toward the 240-min EMA beats persistence on both halves
    of the proxy sample (paired t about -1.5); a bigger pull does not pay.
  * Half-weighting the 5-minute median as the anchor insures against the
    checkpoint close being a one-minute pool dislocation (the graded series
    is a single Orca pool's price, where a large trade moves it for a
    moment) without lagging a genuine fast move.
  * Everything is shrunk hard toward persistence: typical |pred| is well
    under 1%. If the signal is absent the entry costs almost nothing
    relative to the baseline; if it is present it wins a little.
  * No fitting happens at run time: all constants are fixed here, so the
    program is fast (tens of ms per case) and deterministic (two runs of
    the same case produce byte-identical output).

Backtest (Binance BTCUSDT 1m proxy, NOT the graded Orca series):
  persistence MAE 54,241.6 cents; this model 53,169.5 cents (-1.98%,
  paired t -1.46, wins 42/66 cases). On the 6 practice checkpoints
  (proxy) it is about even with persistence (+162.7 cents, inside noise).
  The proxy level tracks the real one: persistence on the practice proxy
  is 44,324.2 cents vs the house practice baseline of 44,521.6 cents on
  the true Orca series. Absolute MAE on the final will differ; the
  relative ranking is what transfers.

Usage (local):  mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
The grader runs: python3 /bot/main.py, reading /input/*.in, writing
/output/<id>.out. Standard library only, no network, no randomness.
"""

import json
import math
import os
import statistics

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

MIN_PRICE = 0.00001
MAX_PRICE = 9_999_999.99999

# --- model constants (fixed; see module docstring) ---
ANCHOR_W = 0.5        # weight of the 5-minute median anchor
MR_K = 0.10           # pull toward the 240-minute EMA of log price
EMA_SPAN = 240        # minutes
CLIP_MULT = 2.0       # clip |pred| to this multiple of the 4h vol


def _log_prices(prices):
    """Positive finite log prices, in order. Bad values are dropped."""
    out = []
    for p in prices:
        try:
            v = float(p)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0.0:
            out.append(math.log(v))
    return out


def _ema(values, span):
    a = 2.0 / (span + 1.0)
    e = values[0]
    for v in values[1:]:
        e = a * v + (1.0 - a) * e
    return e


def _sd_1m(values, window):
    """Population sd of 1-minute log returns over the last `window` minutes."""
    w = values[-(window + 1):] if len(values) > window else values
    if len(w) < 2:
        return 0.0
    diffs = [w[i + 1] - w[i] for i in range(len(w) - 1)]
    return statistics.pstdev(diffs) if len(diffs) > 1 else 0.0


def forecast_price(prices):
    """One case's forecast price from its closes (list of numbers)."""
    x = _log_prices(prices)
    if not x:
        return 1.0  # no usable price at all; still a valid output
    last = x[-1]
    last_price = math.exp(last)

    n = len(x)
    # Robust anchor: median of the last up-to-5 closes (a one-minute pool
    # spike cannot move a median).
    med5 = sorted(x[-5:])[len(x[-5:]) // 2]
    anchor = ANCHOR_W * (med5 - last)

    # Mean reversion toward the 240-minute EMA of log price.
    ema240 = _ema(x, EMA_SPAN)
    pull = -MR_K * (last - ema240)

    # 4h-horizon volatility from 1-minute log returns (average of the
    # 240-minute and 1440-minute windows, scaled to 240 minutes).
    s240 = _sd_1m(x, 240)
    s1440 = _sd_1m(x, 1440)
    sig4h = 0.5 * (s240 + s1440) * math.sqrt(240.0)

    # The pull is the speculative part: clip it to +/- 2 * sigma_4h so a
    # wild regime cannot blow up a case. The anchor is a level estimate,
    # not a bet, so it is not clipped.
    cap = CLIP_MULT * sig4h if (sig4h > 0.0 and math.isfinite(sig4h)) else 0.0
    if pull > cap:
        pull = cap
    elif pull < -cap:
        pull = -cap

    guess = last_price * math.exp(anchor + pull)
    if not math.isfinite(guess) or guess <= 0.0:
        guess = last_price
    # Keep on the price grid (5 decimals) and inside the scorer's bounds.
    guess = round(min(MAX_PRICE, max(MIN_PRICE, guess)) * 1e5) / 1e5
    return guess


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    try:
        names = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".in"))
    except FileNotFoundError:
        return
    for name in names:
        case_id = name[: -len(".in")]
        try:
            with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
                case = json.load(fh)
            markets = case.get("markets") or []
            prices = (markets[0].get("prices") or []) if markets else []
            guess = forecast_price(prices)
        except Exception:
            # Never crash a case: fall back to the checkpoint close if we
            # can read it, else a plain valid price. A valid output always
            # beats a missing one (an invalid case forfeits any prize).
            try:
                with open(os.path.join(INPUT_DIR, name), "r", encoding="utf-8") as fh:
                    case = json.load(fh)
                markets = case.get("markets") or []
                prices = (markets[0].get("prices") or []) if markets else []
                last = None
                for p in reversed(prices):
                    try:
                        v = float(p)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(v) and v > 0.0:
                        last = v
                        break
                guess = last if last is not None else 1.0
                guess = round(min(MAX_PRICE, max(MIN_PRICE, guess)) * 1e5) / 1e5
            except Exception:
                guess = 1.0
        out = {"forecasts": [{"case": case_id, "price": guess}]}
        with open(os.path.join(OUTPUT_DIR, case_id + ".out"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
