# Zied XRP AI

Independent XRP-USDT Spot market research and paper-signal engine.

## Operating rules
- XRP-USDT Spot only; simulated BUY/WAIT signals only. No exchange orders or API credentials.
- One pending or open paper signal at a time. A signal becomes open only when its AI-selected entry condition is reached.
- AI entry methods: MARKET, BREAKOUT, RETEST, SUPPORT_BOUNCE. Entry-method learning starts after 10 closed signals and continues incrementally.
- AI TP/SL exit policy and the 30-layer classifier remain gated until at least 10 closed signals have compatible, finite XRP feature snapshots.
- No Time Exit. Signals close only through TP or SL; trailing stop can tighten but never loosen.
- All results are paper estimates, net of configured simulated fees and slippage.

## Market analysis
- OKX public market data for XRP-USDT: 5m, 15m, 1h, 4h candles, ticker, and order book.
- Stable 1,205-dimensional feature schema, eight dynamically ranked strategy families, regime context, and XRP/Ripple news sentiment.
- AI model weights, state, history, strategy snapshot and summary are independent from the Solana repository.

## Persistence
- Workflow executes 100 cycles, saves state after every cycle, then dispatches the next run.
- State is restored between workflow runs and is never reset by a normal start.
- Files above 90 MiB are split into chunks and verified with SHA-256 on restore.
- `xrp_summary.json` is the compact dashboard feed; `xrp_state.json` retains detailed history.

## Safety
All signals are simulated. This is research software, not investment advice. Model metrics and backtests are experimental and do not guarantee future results.
