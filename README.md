# btc-cent-forecast-01 — Bitcoin 4-hour-ahead forecaster

Entry for the [make.fun](https://make.fun) bounty
**"Forecast Bitcoin's price 4 hours ahead, 12 times over 48 hours"**
(`btc-cent-forecast-01`). One program, `main.py`, Python 3.13, standard
library only, no network, no randomness.

## What it does

For each case it reads the 4,320 one-minute closes of the 72 hours up to
the checkpoint (from `$MAKEFUN_INPUT_DIR`, else `/input/`) and writes one
guess of the price 240 minutes later (to `$MAKEFUN_OUTPUT_DIR`, else
`/output/`), as `{"forecasts": [{"case": "<id>", "price": <number>}]}`.

The model works in log-price space and is deliberately small and heavily
shrunk toward persistence (the house baseline):

```
forecast = last_close * exp( 0.5 * (median(last 5 closes) - last_close)
                             - 0.10 * (last_close - EMA240) )
```

with the mean-reversion pull clipped to ±2× the 4h volatility implied by
recent 1-minute returns. Rationale:

- 4h BTC moves are mostly noise. In a 66-checkpoint backtest on Binance
  BTCUSDT 1-minute closes (4h grid, 2026-11-04–14), short-horizon
  returns are *negatively* correlated with the next 4h return
  (mean reversion), while trend-following the recent return makes things
  much worse (−12.9% vs persistence).
- A small pull toward the 240-minute EMA beats persistence on both halves
  of that sample; a bigger pull does not pay. The 5-minute median anchor
  insures against the checkpoint close being a one-minute dislocation in
  the graded series (a single Orca pool's price, where a large trade
  moves it for a moment).
- No fitting happens at run time — every constant is fixed — so the
  program is deterministic (two runs give byte-identical output) and runs
  in tens of milliseconds per case, far under the 8-second cap.

Backtest (Binance BTCUSDT proxy — the graded series is Orca cbBTC/USDC,
so absolute MAE differs; relative ranking is what transfers):

| model | MAE, cents (66 checkpoints) |
|---|---|
| persistence (house baseline) | 54,241.6 |
| this model | 53,169.5 (−1.98%, paired t −1.46, wins 42/66) |

On the 6 kit practice checkpoints (same proxy) it is about even with
persistence (+162.7 cents — inside noise), and the proxy level tracks the
real one: persistence on the practice proxy is 44,324.2 cents vs the
house practice baseline of 44,521.6 cents on the true Orca series.

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 /bot/main.py` with `/input` and `/output`.
The backtest/validation scripts live outside this directory
(`~/workspace/makefun/backtest*.py`, `fetch_binance.py`) and are not part
of the entry.

## License

MIT — see `LICENSE`.
