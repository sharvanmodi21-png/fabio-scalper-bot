import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import ccxt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Load configuration (same as bot)
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)

MODE = cfg.get("mode", "paper")
RISK_PCT = float(cfg.get("risk_percent", 0.5)) / 100.0
VOLUME_PROFILE_BINS = int(cfg.get("volume_profile_bins", 50))
LVN_PERCENTILE = float(cfg.get("lvn_percentile", 5)) / 100.0
DAILY_LOSS_PCT = float(cfg.get("daily_loss_percent", 2)) / 100.0
MAX_CONSECUTIVE_LOSSES = int(cfg.get("max_consecutive_losses", 3))
LOG_FILE = cfg.get("log_file", "bot.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("backtest")

# ---------------------------------------------------------------------------
# Helper: build volume profile from depth snapshot
# ---------------------------------------------------------------------------
def build_volume_profile(depth: Dict, price_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return bucket centre prices and aggregated volumes.

    depth – {"bids": [(price, amount), ...], "asks": [(price, amount), ...]}
    price_range – (low, high) defining the impulse leg range.
    """
    low, high = price_range
    if low >= high:
        return np.array([]), np.array([])

    bins = np.linspace(low, high, VOLUME_PROFILE_BINS + 1)
    bucket_vol = np.zeros(VOLUME_PROFILE_BINS)
    for side in ("bids", "asks"):
        for price, amt in depth.get(side, []):
            idx = np.searchsorted(bins, price, side="right") - 1
            if 0 <= idx < VOLUME_PROFILE_BINS:
                bucket_vol[idx] += amt
    bucket_prices = (bins[:-1] + bins[1:]) / 2.0
    return bucket_prices, bucket_vol

# ---------------------------------------------------------------------------
# Helper: compute LVNs and POC from depth
# ---------------------------------------------------------------------------
def compute_depth_lvns(depth: Dict) -> Tuple[List[float], float]:
    """Identify LVNs and POC using a fixed‑range volume profile.

    The price range is derived from the most recent 30‑minute candle. If that
    cannot be fetched (e.g., during back‑test), a fallback wide range is used.
    """
    # Attempt to fetch the last completed 30‑min candle for BTC/USDT
    try:
        exchange = ccxt.binance({"enableRateLimit": True})
        recent = exchange.fetch_ohlcv("BTC/USDT", timeframe="30m", limit=2)
        # recent[-2] is the completed candle (the last one may be still forming)
        _, low, high, _, _ = recent[-2]
    except Exception:
        # Fallback: use a broad range that should encompass typical BTC price
        low, high = 20000.0, 40000.0

    bucket_prices, bucket_vol = build_volume_profile(depth, (float(low), float(high)))
    if bucket_vol.size == 0:
        return [], 0.0

    poc_idx = int(np.argmax(bucket_vol))
    poc_price = float(bucket_prices[poc_idx])
    max_vol = bucket_vol.max()
    threshold = LVN_PERCENTILE * max_vol
    lvns = [float(p) for p, v in zip(bucket_prices, bucket_vol) if v <= threshold]
    return lvns, poc_price

# ---------------------------------------------------------------------------
# In‑memory wallet for back‑testing (mirrors the paper wallet logic)
# ---------------------------------------------------------------------------
class BacktestWallet:
    def __init__(self, initial_usdt: float = 10_000.0):
        self.usdt = initial_usdt
        self.btc = 0.0
        self.initial_capital = initial_usdt
        self.position: Optional[Dict] = None
        self.consecutive_losses = 0
        self.daily_loss = 0.0
        self.trade_count = 0
        self.cumulative_profit = 0.0
        self._last_price = 0.0
        self.daily_loss_limit = DAILY_LOSS_PCT * self.initial_capital
        self.daily_reset_time = None  # will be set on first price update

    def equity(self) -> float:
        return self.usdt + self.btc * self._last_price

    def update_price(self, price: float):
        self._last_price = price
        now = datetime.utcnow()
        if not self.daily_reset_time:
            # Set the first daily reset to the next UTC midnight
            self.daily_reset_time = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
        if now >= self.daily_reset_time:
            self.reset_daily()
            self.daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)

    def reset_daily(self):
        self.consecutive_losses = 0
        self.daily_loss = 0.0
        logger.info("Daily counters reset")

    def open_position(self, side: str, size_usdt: float, entry_price: float, sl: float, tp: float):
        qty = size_usdt / entry_price
        self.position = {
            "side": side,
            "entry_price": entry_price,
            "size_usdt": size_usdt,
            "qty": qty,
            "sl": sl,
            "tp": tp,
            "opened_at": datetime.utcnow(),
        }
        self.usdt -= size_usdt
        self.trade_count += 1
        logger.info(f"Opened {side.upper()} {qty:.6f} BTC @ {entry_price:.2f} (SL={sl:.2f}, TP={tp:.2f})")

    def close_position(self, price: float) -> float:
        if not self.position:
            return 0.0
        side = self.position["side"]
        qty = self.position["qty"]
        pnl = (price - self.position["entry_price"]) * qty
        if side == "short":
            pnl = -pnl
        # Return margin + profit
        self.usdt += self.position["size_usdt"] + pnl
        self.btc = 0.0
        self.position = None
        self.cumulative_profit += pnl
        logger.info(f"Closed {side.upper()} @ {price:.2f}, PnL = {pnl:.2f} USDT")
        if pnl < 0:
            self.consecutive_losses += 1
            self.daily_loss += abs(pnl)
        else:
            self.consecutive_losses = 0
        return pnl

# ---------------------------------------------------------------------------
# Trade entry logic (simplified placeholder – replace with the real bot logic)
# ---------------------------------------------------------------------------
def should_enter_trade(wallet: BacktestWallet, depth: Dict) -> Optional[Dict]:
    """Return a dict with trade details or None.

    This placeholder uses a very naive rule: if there is *any* LVN below the current
    price, go long; if there is an LVN above, go short. In a real back‑test you would
    reuse the exact entry conditions from the live bot.
    """
    if wallet.position:
        return None
    lvns, poc = compute_depth_lvns(depth)
    if not lvns:
        return None
    price = wallet._last_price
    # Simple heuristic – enter long if price > POC and there is an LVN below price
    if price > poc and any(lvn < price for lvn in lvns):
        # Compute risk based on a fixed % of equity
        risk_usdt = wallet.equity() * RISK_PCT
        sl = price * 0.999  # 0.1% stop‑loss placeholder
        tp = price * 1.01   # 1% target placeholder
        return {"side": "long", "size_usdt": risk_usdt, "entry_price": price, "sl": sl, "tp": tp}
    # Enter short if price < POC and LVN above price
    if price < poc and any(lvn > price for lvn in lvns):
        risk_usdt = wallet.equity() * RISK_PCT
        sl = price * 1.001
        tp = price * 0.99
        return {"side": "short", "size_usdt": risk_usdt, "entry_price": price, "sl": sl, "tp": tp}
    return None

# ---------------------------------------------------------------------------
# Load historical depth snapshots (CSV files)
# Expected format per CSV: timestamp,price,side,amount
#   side = "bid" or "ask"
# ---------------------------------------------------------------------------
def load_depth_snapshots(folder: str) -> List[Tuple[datetime, Dict]]:
    snapshots: List[Tuple[datetime, Dict]] = []
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith('.csv'):
            continue
        path = os.path.join(folder, fname)
        df = pd.read_csv(path)
        # Expect columns: timestamp (ISO), price, side, amount
        depth: Dict = {"bids": [], "asks": []}
        for _, row in df.iterrows():
            ts = datetime.fromisoformat(row["timestamp"])
            price = float(row["price"])
            amt = float(row["amount"])
            if row["side"].lower() == "bid":
                depth["bids"].append((price, amt))
            else:
                depth["asks"].append((price, amt))
        # Use the timestamp of the first row as the snapshot time (all rows share it)
        if not df.empty:
            snapshot_time = datetime.fromisoformat(df.iloc[0]["timestamp"])
            snapshots.append((snapshot_time, depth))
    return snapshots

# ---------------------------------------------------------------------------
# Main back‑test driver
# ---------------------------------------------------------------------------
def run_backtest():
    logger.info("Starting back‑test")
    wallet = BacktestWallet()
    # Load depth data – user should place CSVs under historical_depth/
    depth_folder = os.path.join(os.path.dirname(__file__), "historical_depth")
    snapshots = load_depth_snapshots(depth_folder)
    if not snapshots:
        logger.error("No depth snapshots found in %s", depth_folder)
        return

    trade_records: List[Dict] = []

    for ts, depth in snapshots:
        # Convert UTC to IST for trading window check
        now_ist = ts + timedelta(hours=5, minutes=30)
        can_trade = now_ist.weekday() < 5 and 18 <= now_ist.hour < 22
        # Update last price – use mid‑price of the best bid/ask if available
        if depth["bids"] and depth["asks"]:
            best_bid = max(p for p, _ in depth["bids"])
            best_ask = min(p for p, _ in depth["asks"])
            mid_price = (best_bid + best_ask) / 2.0
        else:
            mid_price = wallet._last_price
        wallet.update_price(mid_price)

        # Daily loss limit & consecutive‑loss stop
        if wallet.daily_loss >= wallet.daily_loss_limit or wallet.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            can_trade = False

        # Exit position if stop‑loss or take‑profit hit
        if wallet.position:
            side = wallet.position["side"]
            sl = wallet.position["sl"]
            tp = wallet.position["tp"]
            if side == "long" and (mid_price <= sl or mid_price >= tp):
                pnl = wallet.close_position(mid_price)
                trade_records.append({
                    "timestamp": ts.isoformat(),
                    "side": side,
                    "entry": wallet.position["entry_price"] if wallet.position else None,
                    "exit": mid_price,
                    "pnl": pnl,
                })
            elif side == "short" and (mid_price >= sl or mid_price <= tp):
                pnl = wallet.close_position(mid_price)
                trade_records.append({
                    "timestamp": ts.isoformat(),
                    "side": side,
                    "entry": wallet.position["entry_price"] if wallet.position else None,
                    "exit": mid_price,
                    "pnl": pnl,
                })

        if can_trade and not wallet.position:
            decision = should_enter_trade(wallet, depth)
            if decision:
                wallet.open_position(
                    side=decision["side"],
                    size_usdt=decision["size_usdt"],
                    entry_price=decision["entry_price"],
                    sl=decision["sl"],
                    tp=decision["tp"],
                )
                trade_records.append({
                    "timestamp": ts.isoformat(),
                    "side": decision["side"],
                    "entry": decision["entry_price"],
                    "sl": decision["sl"],
                    "tp": decision["tp"],
                    "size_usdt": decision["size_usdt"],
                })

    # -----------------------------------------------------------------------
    # Export results to Excel
    # -----------------------------------------------------------------------
    if not trade_records:
        logger.warning("No trades were generated during the back‑test")
    else:
        df_trades = pd.DataFrame(trade_records)
        # Daily summary
        df_trades["date"] = pd.to_datetime(df_trades["timestamp"]).dt.date
        daily = df_trades.groupby("date").agg(
            trades="timestamp",
            profit=("pnl", "sum"),
        ).reset_index()
        daily["trades"] = daily["trades"].astype(int)

        # Overall stats
        total_trades = len(df_trades)
        win_rate = (df_trades["pnl"] > 0).mean() * 100 if total_trades else 0
        avg_pnl = df_trades["pnl"].mean() if total_trades else 0
        max_drawdown = (df_trades["pnl"].cumsum().cummax() - df_trades["pnl"].cumsum()).max()
        stats = {
            "total_trades": total_trades,
            "win_rate_%": round(win_rate, 2),
            "avg_pnl": round(avg_pnl, 2),
            "max_drawdown": round(max_drawdown, 2),
            "final_equity_usdt": round(wallet.equity(), 2),
        }
        df_stats = pd.DataFrame([stats])

        excel_path = os.path.join(os.path.dirname(__file__), "backtest_results.xlsx")
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df_trades.to_excel(writer, sheet_name="Trades", index=False)
            daily.to_excel(writer, sheet_name="Daily", index=False)
            df_stats.to_excel(writer, sheet_name="Stats", index=False)
        logger.info("Back‑test completed. Results written to %s", excel_path)

if __name__ == "__main__":
    run_backtest()
