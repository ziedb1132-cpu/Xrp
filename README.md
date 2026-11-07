# Zied XRP AI

Independent XRP-USDT Spot market research and paper-signal engine.

## Operating rules
- XRP-USDT Spot only; research signals are simulated. A separate live OKX Spot adapter exists and remains disabled unless `XRP_LIVE_TRADING_ENABLED=true` is explicitly configured.
- One pending or open paper signal at a time. A signal becomes open only when its AI-selected entry condition is reached.
- AI entry methods: MARKET, BREAKOUT, RETEST, SUPPORT_BOUNCE. Outcome-based entry-method learning requires at least 30 compatible net winners and 30 net losers, then updates incrementally.
- The 30-layer classifier and learned TP/SL policy require at least 30 compatible wins and 30 compatible losses, valid price-excursion labels, and a chronological train/validation split containing both outcomes. A conservative out-of-sample overfit guard checks validation log-loss, Brier score, balanced accuracy and train/validation generalization gap before saving or loading a checkpoint; rejected candidates never replace the last approved model. Checkpoints without the current guard approval are not loaded.
- Standard new signals use ATR/AI-bounded TP from 2% to 6% and SL from 1% to 3%, with a minimum TP:SL distance ratio of 1.5:1. Existing open signals retain their original TP/SL.
- A separate XRP scalping profile checks completed 1m candles with 5m trend confirmation, RSI, short momentum, volume and order-book spread/imbalance. It requires a tight spread (at most 0.06%) and at least five independent confirmations.
- New scalp signals use TP 0.90%–1.50% and SL 0.40%–0.80%, with at least 1.5:1 gross distance ratio and the configured estimated round-trip fee/slippage cost considered. The scalp profile has its own trailing activation (50% of target distance) and stop gap (0.25%); the standard strategy and existing open positions are unchanged.
- The existing paper signal lifecycle records scalping results separately in `scalping_performance`. The live adapter can mirror a newly opened scalp paper signal to OKX when live execution is enabled, available USDT exists, and all existing single-position/OCO safety checks pass. Paper results and actual OKX fills remain separate; a paper estimate is not an exchange fill.
- New-signal trailing activates after 30% of the TP distance is reached, moves the stop to fee-aware break-even, then trails 0.70% below the high-water price. Stops only move upward. Pre-existing open signals retain their legacy trailing behavior.
- No Time Exit. Paper signals close only through TP or SL. Historical backtests also leave trades unresolved if neither level is reached before the data ends.
- Backtests use ATR-derived exits, matching trailing logic, configured fees/slippage, chronological signal evaluation, and conservative stop-first handling when an OHLC candle touches both TP and SL.
- Missing order-book data or an excessive spread blocks new entries. Market regime classification requires multi-timeframe trend agreement.
- All results are paper estimates, net of configured simulated fees and slippage; they are not a guarantee of future results.

## Market analysis
- OKX public market data for XRP-USDT: 5m, 15m, 1h, 4h candles, ticker, and order book.
- Stable 1,205-dimensional feature schema, eight dynamically ranked strategy families, regime context, and XRP/Ripple news sentiment.
- AI model weights, state, history, strategy snapshot and summary are independent from the Solana repository.

## Persistence
- Workflow executes 100 cycles, saves state after every cycle, then dispatches the next run.
- State is restored between workflow runs and is never reset by a normal start.
- Files above 90 MiB are split into chunks and verified with SHA-256 on restore.
- `xrp_summary.json` is the compact dashboard feed, including net expectancy, profit factor, average win/loss, drawdown, trailing results, AI-vs-pre-AI performance and per-strategy backtest exit counts; `xrp_state.json` retains detailed history.

## Live execution and safety
- Live execution is opt-in and separate from paper-signal generation. It uses OKX Spot cash mode only; no margin or leverage.
- Each eligible new XRP-USDT live BUY uses 100% of the currently available USDT balance, rounded down to the nearest 0.01 USDT. Funds already reserved by OKX are excluded. Fees, price movement, exchange minimums, and precision can still cause an order to be rejected or filled differently than estimated.
- Only one live XRP position is allowed at a time. The adapter requires verified TP/SL OCO protection and halts for manual reconciliation if protection cannot be confirmed.
- A halt caused specifically by `No available USDT balance.` is checked automatically each cycle. Once Trading-account USDT is available, the adapter verifies there are no open XRP spot orders, pending protective algos, or existing XRP holdings before clearing that balance-only halt. It does not replay the signal that failed for lack of funds; it waits for a new signal. Any other halt reason, or any existing XRP/order exposure, remains halted for manual reconciliation.
- Full-balance allocation can expose the entire available USDT balance to loss. This is research software, not investment advice. Model metrics and backtests are experimental and do not guarantee future results.
