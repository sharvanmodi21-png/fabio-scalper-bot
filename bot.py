"""btc_scalp_bot – A minimal implementation of Fabio Valentini's scalping strategy

Features
--------
* Supports **live** trading on Binance (BTC/USDT) and **paper** (simulation) mode.
* Paper mode is the default – the bot will *not* send real orders unless you change `mode` to "live" in `config.json`.
* Uses Binance REST (via **ccxt**) for account info and order placement.
* Uses a WebSocket depth stream to approximate order‑flow / footprint data.
* Implements a three‑loss daily stop and a simple risk‑per‑trade calculation.
* Logs all decisions to the file specified in `config.json`.

Important – Do not run this bot with real funds until you have thoroughly back‑tested the logic
and understand the risks. The implementation below is a **skeleton** that demonstrates the
core workflow; you will likely need to tune thresholds, LVN/POC detection, and the entry/exit
conditions to match the full Valentini methodology.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd
import websocket

# ---------------------------------------------------------------------------
# Configuration & Logging
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).with_name("config.json")
if not CONFIG_PATH.is_file():
    print("Missing config.json – aborting.")
    sys.exit(1)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

MODE = cfg.get("mode", "paper").lower()  # "live" or "paper"
RISK_PCT = float(cfg.get("risk_percent", 0.25)) / 100.0
VOLUME_PROFILE_BINS = int(cfg.get("volume_profile_bins", 50))
LVN_PERCENTILE = float(cfg.get("lvn_percentile", 5)) / 100.0
MAX_LOSSES = int(cfg.get("max_consecutive_losses", 3))
LOG_FILE = cfg.get("log_file", "bot.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("btc_scalp_bot")

# ---------------------------------------------------------------------------
# Helper utilities – Paper trading state
# ---------------------------------------------------------------------------
class PaperWallet:
    """Simple in‑memory wallet for paper mode.

    Balances are stored in USDT (quote) and BTC (base). The wallet starts with a
    configurable amount of USDT (default $1,000) and zero BTC.
    """

    def __init__(self, initial_usdt: float = 1_000.0):
        self.usdt = initial_usdt
        self.btc = 0.0
        self.initial_capital = initial_usdt  # for daily loss limit calculations
        self.position: Optional[Dict] = None  # Holds the active trade details
        self.consecutive_losses = 0
        self.daily_loss = 0.0  # cumulative loss for the current day
        # Tracking stats for summary
        self.trade_count = 0
        self.cumulative_profit = 0.0

    def reset_daily(self):
        self.consecutive_losses = 0
        self.daily_loss = 0.0
        logger.info("Daily loss counter reset.")

    def equity(self) -> float:
        """Current equity in USDT (USDT + BTC * last_price)."""
        price = getattr(self, "_last_price", 0.0)
        return self.usdt + self.btc * price

    def update_price(self, price: float):
        self._last_price = price

    def open_position(self, side: str, size_usdt: float, entry_price: float, sl: float, tp: float):
        """Record a simulated position.

        side – "long" or "short"
        size_usdt – USD amount risked (not full notional)
        """
        qty = size_usdt / entry_price
        self.position = {
            "side": side,
            "entry_price": entry_price,
            "size_usdt": size_usdt,
            "qty": qty,
            "sl": sl,
            "tp": tp,
            "opened_at": datetime.now(timezone.utc).replace(tzinfo=None),
        }
        # Deduct the risked amount from cash (margin placeholder)
        self.usdt -= size_usdt
        self.trade_count += 1
        logger.info(f"[Paper] Opened {side} {qty:.6f} BTC @ {entry_price:.2f} (SL={sl:.2f} TP={tp:.2f})")

    def close_position(self, price: float):
        if not self.position:
            return
        side = self.position["side"]
        qty = self.position["qty"]
        pnl = (price - self.position["entry_price"]) * qty
        if side == "short":
            pnl = -pnl
        self.usdt += self.position["size_usdt"] + pnl  # return used margin + PnL
        self.btc += 0  # no actual BTC holds in paper mode
        profit = pnl
        self.cumulative_profit += profit
        logger.info(f"[Paper] Closed {side} @ {price:.2f}, PnL = {profit:.2f} USDT")
        # Update loss counters
        if profit < 0:
            loss_amount = -profit
            self.daily_loss += loss_amount
            self.consecutive_losses += 1
            logger.info(f"[Paper] Consecutive losses: {self.consecutive_losses}, Daily loss: {self.daily_loss:.2f} USDT")
        else:
            self.consecutive_losses = 0
        self.position = None
    def reset_daily(self):
        self.consecutive_losses = 0
        logger.info("Daily loss counter reset.")

    def equity(self) -> float:
        """Current equity in USDT (USDT + BTC * last_price)."""
        price = self._last_price if hasattr(self, "_last_price") else 0.0
        return self.usdt + self.btc * price

    def update_price(self, price: float):
        self._last_price = price

    def open_position(self, side: str, size_usdt: float, entry_price: float, sl: float, tp: float):
        """Record a simulated position.

        side – "long" or "short"
        size_usdt – USD amount risked (not full notional)
        """
        qty = size_usdt / entry_price
        self.position = {
            "side": side,
            "entry_price": entry_price,
            "size_usdt": size_usdt,
            "qty": qty,
            "sl": sl,
            "tp": tp,
            "opened_at": datetime.now(timezone.utc).replace(tzinfo=None),
        }
        # Adjust cash for the used margin (full notional for simplicity)
        self.usdt -= size_usdt
        logger.info(f"[Paper] Opened {side} {qty:.6f} BTC @ {entry_price:.2f} (SL={sl:.2f} TP={tp:.2f})")

    def close_position(self, price: float):
        if not self.position:
            return
        side = self.position["side"]
        qty = self.position["qty"]
        pnl = (price - self.position["entry_price"]) * qty
        if side == "short":
            pnl = -pnl
        self.usdt += self.position["size_usdt"] + pnl  # return used margin + PnL
        self.btc += 0  # no actual BTC holds in paper mode
        profit = pnl
        logger.info(f"[Paper] Closed {side} @ {price:.2f}, PnL = {profit:.2f} USDT")
        # Update loss counter
        if profit < 0:
            self.consecutive_losses += 1
            logger.info(f"[Paper] Consecutive losses: {self.consecutive_losses}")
        else:
            self.consecutive_losses = 0
        self.position = None

# ---------------------------------------------------------------------------
# Core Strategy Helpers
# ---------------------------------------------------------------------------
def fetch_kline(exchange: ccxt.binance, symbol: str, timeframe: str = "30m", limit: int = 100) -> pd.DataFrame:
    """Return recent klines as a DataFrame with columns: timestamp, open, high, low, close, volume."""
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df

def detect_market_state(df: pd.DataFrame) -> str:
    """Very simple market‑state detector.

    Returns "balance" if price is inside the previous range, otherwise "imbalance".
    This is a placeholder – the full strategy would use volume profile analysis.
    """
    recent = df.iloc[-2:]
    prev_low, prev_high = recent.iloc[0][["low", "high"]]
    cur_close = df.iloc[-1]["close"]
    if prev_low <= cur_close <= prev_high:
        return "balance"
    return "imbalance"

def compute_depth_lvns(depth: Dict) -> Tuple[List[float], float]:
    """Identify LVNs (price levels with minimal cumulative depth) and POC.

    depth – dict with 'bids' and 'asks' each as list of [price, amount].
    Returns (list_of_lvn_prices, poc_price). Handles empty depth gracefully.
    """
    # If no depth data yet, return empty LVN list and a neutral POC (0)
    if not depth.get('bids') and not depth.get('asks'):
        return [], 0.0
    # Combine bids and asks into a single price‑volume series
    price_vol = {}
    for side in ("bids", "asks"):
        for price, amt in depth[side]:
            price_vol[price] = price_vol.get(price, 0) + amt
    # Convert to sorted arrays
    prices = np.array(sorted(price_vol.keys()))
    volumes = np.array([price_vol[p] for p in prices])
    # POC = price with max volume
    poc_price = float(prices[np.argmax(volumes)])
    # LVN detection – simple heuristic: volume < 5% of max volume
    threshold = 0.05 * volumes.max()
    lvns = [float(p) for p, v in zip(prices, volumes) if v < threshold]
    return lvns, poc_price

def should_enter_trade(state: str, lvns: List[float], price: float, depth: Dict) -> bool:
    """Determine if entry conditions are met.

    - Price must be at (or very near) an LVN.
    - Depth must show absorption on the opposite side and a spike of market‑order flow.
    This is a highly simplified version; replace with more sophisticated footprint checks.
    """
    if state != "imbalance":
        return False
    # Proximity check (within 0.1% of LVN)
    near_lvn = any(abs(price - lvn) / lvn < 0.001 for lvn in lvns)
    if not near_lvn:
        return False
    # Simple absorption test: large bid depth supporting price
    bid_depth = sum(amt for p, amt in depth["bids"] if p >= price)
    ask_depth = sum(amt for p, amt in depth["asks"] if p <= price)
    # If bid depth is > 3× ask depth we consider it absorption for a long trade
    return bid_depth > 3 * ask_depth

# ---------------------------------------------------------------------------
# Main Bot Class
# ---------------------------------------------------------------------------
class ScalpingBot:
    SYMBOL = "BTC/USDT"
    WS_ENDPOINT = "wss://data-stream.binance.vision/ws/btcusdt@depth20"

    def __init__(self, config: dict):
        self.config = config
        # For paper mode we can use public endpoints without authentication.
        if MODE == "paper":
            self.exchange = ccxt.binance({"enableRateLimit": True})
        else:
            self.exchange = ccxt.binance({
                "apiKey": config.get("apiKey"),
                "secret": config.get("secret"),
                "enableRateLimit": True,
            })
        self.paper_wallet = PaperWallet() if MODE == "paper" else None
        self.last_price: float = 0.0
        self.depth: Dict = {"bids": [], "asks": []}
        self.consecutive_losses = 0
        self.daily_reset_time = datetime.now(timezone.utc).replace(tzinfo=None).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        self.last_summary_time = datetime.now(timezone.utc).replace(tzinfo=None)
        # Daily loss limit (2% of initial capital by default)
        self.daily_loss_limit = (self.config.get("daily_loss_percent", 2) / 100.0) * (self.paper_wallet.initial_capital if self.paper_wallet else 0)
        self.last_summary_time = datetime.now(timezone.utc).replace(tzinfo=None)

    # -----------------------------------------------------------------------
    # WebSocket handling (depth stream)
    # -----------------------------------------------------------------------
    def on_message(self, ws, message):
        import json as _json
        data = _json.loads(message)
        # Binance depth event provides arrays of [price, qty]
        self.depth["bids"] = [(float(p), float(q)) for p, q in data.get("b", [])]
        self.depth["asks"] = [(float(p), float(q)) for p, q in data.get("a", [])]
        # Update last known price (midpoint of best bid/ask)
        if self.depth["bids"] and self.depth["asks"]:
            best_bid = self.depth["bids"][0][0]
            best_ask = self.depth["asks"][0][0]
            self.last_price = (best_bid + best_ask) / 2.0
        # Forward to main loop via a flag
        self.tick_received = True

    def on_error(self, ws, error):
        logger.error(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        logger.info("WebSocket closed, reconnecting in 5s...")
        time.sleep(5)
        self.start_ws()

    def start_ws(self):
        self.ws = websocket.WebSocketApp(
            self.WS_ENDPOINT,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        # Run in a background thread so the main loop can continue
        import threading
        t = threading.Thread(target=self.ws.run_forever, daemon=True)
        t.start()
        # Give it a moment to connect
        time.sleep(1)

    # -----------------------------------------------------------------------
    # Trading actions (live vs. paper)
    # -----------------------------------------------------------------------
    def place_order(self, side: str, qty: float, price: float) -> dict:
        """Place a limit order (live mode) or simulate (paper mode)."""
        if MODE == "paper":
            # Simulate instant fill for simplicity
            return {"id": "paper-order", "side": side, "qty": qty, "price": price, "filled": qty}
        # Live order – use ccxt create_limit_order
        try:
            order = self.exchange.create_limit_order(self.SYMBOL, side, qty, price)
            logger.info(f"Live order placed: {order['id']}")
            return order
        except Exception as e:
            logger.error(f"Failed to place live order: {e}")
            return {}

    def close_live_position(self, position_id: str):
        # Placeholder – in a real bot you would track the order ID.
        pass

    # -----------------------------------------------------------------------
    # Core loop
    # -----------------------------------------------------------------------
    def run(self):
        logger.info(f"Starting bot in {MODE.upper()} mode.")
        self.start_ws()
        while True:
            # Restrict trading to Mon‑Fri, 6 pm‑10 pm IST (12:30‑16:30 UTC)
            now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
            now_ist = now_utc + timedelta(hours=5, minutes=30)
            can_trade_time = now_ist.weekday() < 5 and 18 <= now_ist.hour < 22
            # If outside allowed window, skip entry logic but continue processing depth
            if not can_trade_time:
                # Still update price for equity tracking
                if self.paper_wallet:
                    self.paper_wallet.update_price(self.last_price)
                time.sleep(30)  # pause longer when market is closed
                continue
            # Daily reset of loss counter
            if datetime.now(timezone.utc).replace(tzinfo=None) >= self.daily_reset_time:
                if self.paper_wallet:
                    self.paper_wallet.reset_daily()
                self.daily_reset_time = datetime.now(timezone.utc).replace(tzinfo=None).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                # Update summary time after daily reset
                self.last_summary_time = datetime.now(timezone.utc).replace(tzinfo=None)
                # Daily loss limit (2% of initial capital by default)
                self.daily_loss_limit = (self.config.get("daily_loss_percent", 2) / 100.0) * (self.paper_wallet.initial_capital if self.paper_wallet else 0)
                self.last_summary_time = datetime.now(timezone.utc).replace(tzinfo=None)

            # Wait for a new depth tick
            if not getattr(self, "tick_received", False):
                time.sleep(0.1)
                continue
            self.tick_received = False
            # Pull recent higher‑timeframe klines for market‑state detection
            try:
                df = fetch_kline(self.exchange, self.SYMBOL, timeframe="30m", limit=5)
            except Exception as e:
                logger.error(f"Failed to fetch klines: {e}")
                continue
            market_state = detect_market_state(df)

            # Compute LVNs and POC from current depth snapshot
            lvns, poc = compute_depth_lvns(self.depth)

            # Check existing position
            if MODE == "paper":
                wallet = self.paper_wallet
            else:
                # In live mode you would query open positions via the exchange
                wallet = None  # placeholder

            # If we have an open paper position, evaluate SL/TP
            if MODE == "paper" and wallet.position:
                pos = wallet.position
                if pos["side"] == "long" and (self.last_price <= pos["sl"] or self.last_price >= pos["tp"]):
                    wallet.close_position(self.last_price)
                    continue
                if pos["side"] == "short" and (self.last_price >= pos["sl"] or self.last_price <= pos["tp"]):
                    wallet.close_position(self.last_price)
                    continue

            # Enforce three‑loss rule (paper mode only for now)
            if MODE == "paper" and wallet.consecutive_losses >= MAX_LOSSES:
                logger.info("Three consecutive losses reached – pausing trading for today.")
                time.sleep(60)  # sleep and re‑check later
                continue

            # ENTRY LOGIC
            if should_enter_trade(market_state, lvns, self.last_price, self.depth):
                # Determine side – simplified: go long in imbalanced up‑move, short otherwise
                side = "long" if market_state == "imbalance" else "short"
                # Risk calculation – amount of USDT to risk
                if MODE == "paper":
                    equity = wallet.equity()
                else:
                    # Live: fetch balance from exchange (placeholder)
                    equity = 10_000  # replace with real balance fetch
                risk_amount = equity * RISK_PCT
                # Simple stop: 1 tick (0.1% of price) for demo purposes
                tick = self.last_price * 0.001
                sl = self.last_price - tick if side == "long" else self.last_price + tick
                # Take profit = POC (from depth) – could be refined
                tp = poc
                # Compute quantity based on risk (approximate, ignoring fees)
                qty = risk_amount / abs(self.last_price - sl)
                # Place order
                order = self.place_order("buy" if side == "long" else "sell", qty, self.last_price)
                if MODE == "paper":
                    wallet.open_position(side, risk_amount, self.last_price, sl, tp)
                else:
                    # In live mode you would store the order ID and monitor fill status
                    pass
                logger.info(f"Entered {side.upper()} position at {self.last_price:.2f}, SL={sl:.2f}, TP={tp:.2f}")

            # Throttle loop to avoid hammering the API (depth updates arrive ~100 ms)
            time.sleep(0.2)

if __name__ == "__main__":
    bot = ScalpingBot(cfg)
    bot.run()
