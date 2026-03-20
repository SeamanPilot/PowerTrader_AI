"""
pt_paper_trader.py  —  Paper Trading Bot for PowerTrader AI

Mirrors the logic of pt_trader.py (DCA, trailing profit margin, neural signals)
without touching the real Robinhood API.

Prices are fetched from Yahoo Finance (yfinance), which is already a dependency.
Account state (virtual cash + simulated positions) is persisted to hub_data so the
GUI (pt_hub.py) can display paper-trading performance alongside live-trading data.

Usage:
    python pt_paper_trader.py

Configuration:
    gui_settings.json keys read by this file:
        paper_initial_cash   (float, default 10_000)   — virtual starting balance
        coins                                           — which symbols to trade
        trade_start_level, start_allocation_pct,
        dca_multiplier, dca_levels, max_dca_buys_per_24h,
        pm_start_pct_no_dca, pm_start_pct_with_dca, trailing_gap_pct
        main_neural_dir                                 — for reading neural signal files
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
import uuid
from typing import Any, Dict, Optional

import colorama
from colorama import Fore, Style

# ── yfinance for real-time crypto prices ────────────────────────────────────
try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    print("[PaperTrader] WARNING: yfinance not installed. Install it with: pip install yfinance")

colorama.init(autoreset=True)

# ─────────────────────────────────────────────────────────────────────────────
# HUB DATA PATHS  (paper_ prefix so they don't collide with live trader files)
# ─────────────────────────────────────────────────────────────────────────────
HUB_DATA_DIR = os.environ.get(
    "POWERTRADER_HUB_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "hub_data"),
)
os.makedirs(HUB_DATA_DIR, exist_ok=True)

PAPER_TRADER_STATUS_PATH       = os.path.join(HUB_DATA_DIR, "paper_trader_status.json")
PAPER_TRADE_HISTORY_PATH       = os.path.join(HUB_DATA_DIR, "paper_trade_history.jsonl")
PAPER_PNL_LEDGER_PATH          = os.path.join(HUB_DATA_DIR, "paper_pnl_ledger.json")
PAPER_ACCOUNT_VALUE_HISTORY_PATH = os.path.join(HUB_DATA_DIR, "paper_account_value_history.jsonl")
PAPER_ACCOUNT_STATE_PATH       = os.path.join(HUB_DATA_DIR, "paper_account.json")

# ─────────────────────────────────────────────────────────────────────────────
# GUI SETTINGS  (shared with pt_trader.py)
# ─────────────────────────────────────────────────────────────────────────────
_GUI_SETTINGS_PATH = os.environ.get("POWERTRADER_GUI_SETTINGS") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "gui_settings.json",
)

_gui_settings_cache: dict = {
    "mtime": None,
    "coins": ["BTC", "ETH", "XRP", "BNB", "DOGE"],
    "main_neural_dir": None,
    "trade_start_level": 3,
    "start_allocation_pct": 0.005,
    "dca_multiplier": 2.0,
    "dca_levels": [-2.5, -5.0, -10.0, -20.0, -30.0, -40.0, -50.0],
    "max_dca_buys_per_24h": 2,
    "pm_start_pct_no_dca": 5.0,
    "pm_start_pct_with_dca": 2.5,
    "trailing_gap_pct": 0.5,
    "paper_initial_cash": 10_000.0,
}


def _load_gui_settings() -> dict:
    """Load (and cache by mtime) gui_settings.json."""
    try:
        if not os.path.isfile(_GUI_SETTINGS_PATH):
            return dict(_gui_settings_cache)

        mtime = os.path.getmtime(_GUI_SETTINGS_PATH)
        if _gui_settings_cache["mtime"] == mtime:
            return dict(_gui_settings_cache)

        with open(_GUI_SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f) or {}

        def _float(key: str, default: float) -> float:
            try:
                return float(str(data.get(key, default)).replace("%", "").strip())
            except Exception:
                return default

        def _int(key: str, default: int) -> int:
            try:
                return int(float(data.get(key, default)))
            except Exception:
                return default

        coins = data.get("coins") or list(_gui_settings_cache["coins"])
        coins = [str(c).strip().upper() for c in coins if str(c).strip()]
        if not coins:
            coins = list(_gui_settings_cache["coins"])

        main_neural_dir = data.get("main_neural_dir") or None
        if isinstance(main_neural_dir, str):
            main_neural_dir = main_neural_dir.strip() or None

        dca_levels = data.get("dca_levels", _gui_settings_cache["dca_levels"])
        if not isinstance(dca_levels, list) or not dca_levels:
            dca_levels = list(_gui_settings_cache["dca_levels"])
        parsed = []
        for v in dca_levels:
            try:
                parsed.append(float(v))
            except Exception:
                pass
        if parsed:
            dca_levels = parsed

        _gui_settings_cache.update(
            mtime=mtime,
            coins=coins,
            main_neural_dir=main_neural_dir,
            trade_start_level=max(1, min(_int("trade_start_level", 3), 7)),
            start_allocation_pct=max(0.0, _float("start_allocation_pct", 0.005)),
            dca_multiplier=max(0.0, _float("dca_multiplier", 2.0)),
            dca_levels=dca_levels,
            max_dca_buys_per_24h=max(0, _int("max_dca_buys_per_24h", 2)),
            pm_start_pct_no_dca=max(0.0, _float("pm_start_pct_no_dca", 5.0)),
            pm_start_pct_with_dca=max(0.0, _float("pm_start_pct_with_dca", 2.5)),
            trailing_gap_pct=max(0.0, _float("trailing_gap_pct", 0.5)),
            paper_initial_cash=max(0.0, _float("paper_initial_cash", 10_000.0)),
        )
        return dict(_gui_settings_cache)

    except Exception:
        return dict(_gui_settings_cache)


# ─────────────────────────────────────────────────────────────────────────────
# Live globals (hot-reloaded in manage_trades)
# ─────────────────────────────────────────────────────────────────────────────
crypto_symbols:        list  = ["BTC", "ETH", "XRP", "BNB", "DOGE"]
main_dir:              str   = os.getcwd()
TRADE_START_LEVEL:     int   = 3
START_ALLOC_PCT:       float = 0.005
DCA_MULTIPLIER:        float = 2.0
DCA_LEVELS:            list  = [-2.5, -5.0, -10.0, -20.0, -30.0, -40.0, -50.0]
MAX_DCA_BUYS_PER_24H:  int   = 2
TRAILING_GAP_PCT:      float = 0.5
PM_START_PCT_NO_DCA:   float = 5.0
PM_START_PCT_WITH_DCA: float = 2.5
PAPER_INITIAL_CASH:    float = 10_000.0

# Maximum number of DCA stages driven by the neural signal (stages 0..3 → neural levels 4..7)
MAX_NEURAL_DCA_STAGES: int = 4

_last_settings_mtime: Any = None


def _refresh_globals() -> None:
    global crypto_symbols, main_dir
    global TRADE_START_LEVEL, START_ALLOC_PCT, DCA_MULTIPLIER, DCA_LEVELS, MAX_DCA_BUYS_PER_24H
    global TRAILING_GAP_PCT, PM_START_PCT_NO_DCA, PM_START_PCT_WITH_DCA
    global PAPER_INITIAL_CASH, _last_settings_mtime

    s = _load_gui_settings()
    mtime = s.get("mtime")
    if mtime is None or _last_settings_mtime == mtime:
        return

    _last_settings_mtime = mtime

    coins = s.get("coins") or list(crypto_symbols)
    mndir = s.get("main_neural_dir") or main_dir
    if not os.path.isdir(mndir):
        mndir = os.getcwd()

    crypto_symbols       = list(coins)
    main_dir             = mndir
    TRADE_START_LEVEL    = max(1, min(int(s.get("trade_start_level", TRADE_START_LEVEL)), 7))
    START_ALLOC_PCT      = max(0.0, float(s.get("start_allocation_pct", START_ALLOC_PCT)))
    DCA_MULTIPLIER       = max(0.0, float(s.get("dca_multiplier", DCA_MULTIPLIER)))
    DCA_LEVELS           = list(s.get("dca_levels", DCA_LEVELS))
    MAX_DCA_BUYS_PER_24H = max(0, int(float(s.get("max_dca_buys_per_24h", MAX_DCA_BUYS_PER_24H))))
    TRAILING_GAP_PCT     = max(0.0, float(s.get("trailing_gap_pct", TRAILING_GAP_PCT)))
    PM_START_PCT_NO_DCA  = max(0.0, float(s.get("pm_start_pct_no_dca", PM_START_PCT_NO_DCA)))
    PM_START_PCT_WITH_DCA = max(0.0, float(s.get("pm_start_pct_with_dca", PM_START_PCT_WITH_DCA)))
    PAPER_INITIAL_CASH   = max(0.0, float(s.get("paper_initial_cash", PAPER_INITIAL_CASH)))


# ─────────────────────────────────────────────────────────────────────────────
# Price fetching (yfinance)
# ─────────────────────────────────────────────────────────────────────────────
_price_cache: Dict[str, Dict] = {}
_PRICE_CACHE_TTL = 15.0  # seconds — refresh at most once per 15 s


def _fetch_price(symbol: str) -> tuple[float, float]:
    """
    Return (ask_price, bid_price) for *symbol* (e.g. "BTC").
    Uses a simple 0.05% spread around the last close from yfinance.
    Falls back to (0.0, 0.0) on any error.
    """
    if not _YF_AVAILABLE:
        return 0.0, 0.0

    now = time.time()
    cached = _price_cache.get(symbol)
    if cached and (now - float(cached.get("ts", 0))) < _PRICE_CACHE_TTL:
        return cached["ask"], cached["bid"]

    try:
        ticker = yf.Ticker(f"{symbol}-USD")
        hist = ticker.history(period="1d", interval="1m")
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
            if price > 0:
                ask = price * 1.0005
                bid = price * 0.9995
                _price_cache[symbol] = {"ask": ask, "bid": bid, "ts": now}
                return ask, bid
    except Exception:
        pass

    # Fallback: return cached stale value if available
    if cached:
        return cached["ask"], cached["bid"]
    return 0.0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _atomic_write_json(path: str, data: dict) -> None:
    try:
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass


def _append_jsonl(path: str, obj: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj) + "\n")
    except Exception:
        pass


def _fmt_price(price: float) -> str:
    try:
        p = float(price)
    except Exception:
        return "N/A"
    if p == 0:
        return "0"
    ap = abs(p)
    if ap >= 1.0:
        decimals = 2
    else:
        decimals = max(2, min(12, int(-math.floor(math.log10(ap))) + 3))
    s = f"{p:.{decimals}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Paper Trading Engine
# ─────────────────────────────────────────────────────────────────────────────

class PaperCryptoTrading:
    """
    Drop-in paper-trading equivalent of CryptoAPITrading (pt_trader.py).

    All orders are simulated instantly at the current yfinance price.
    Account state persists across restarts via hub_data/paper_account.json.
    """

    # ── construction ─────────────────────────────────────────────────────────

    def __init__(self) -> None:
        _refresh_globals()

        # ── load or init simulated account state ─────────────────────────────
        self._account = self._load_account()

        # Ensure initial cash is set (only on first run)
        if self._account.get("cash") is None:
            self._account["cash"] = float(PAPER_INITIAL_CASH)
            self._account.setdefault("positions", {})
            self._save_account()

        # ── DCA tracking (same structure as live trader) ──────────────────────
        self.dca_levels_triggered: Dict[str, list] = dict(
            self._account.get("dca_levels_triggered", {})
        )
        self.dca_levels: list = list(DCA_LEVELS)

        # ── Trailing PM (per-coin state) ──────────────────────────────────────
        self.trailing_pm: Dict[str, dict] = dict(
            self._account.get("trailing_pm", {})
        )
        self.trailing_gap_pct      = float(TRAILING_GAP_PCT)
        self.pm_start_pct_no_dca   = float(PM_START_PCT_NO_DCA)
        self.pm_start_pct_with_dca = float(PM_START_PCT_WITH_DCA)

        self._last_trailing_settings_sig = (
            float(self.trailing_gap_pct),
            float(self.pm_start_pct_no_dca),
            float(self.pm_start_pct_with_dca),
        )

        # ── PnL ledger (mirrors live trader structure) ────────────────────────
        self._pnl_ledger = self._load_pnl_ledger()

        # ── DCA rate-limit (rolling 24h window) ──────────────────────────────
        self.max_dca_buys_per_24h = int(MAX_DCA_BUYS_PER_24H)
        self.dca_window_seconds   = 24 * 60 * 60
        self._dca_buy_ts:      Dict[str, list]  = {}
        self._dca_last_sell_ts: Dict[str, float] = {}
        self._seed_dca_window_from_history()

        # ── account value snapshot cache (prevents transient dips in GUI) ─────
        self._last_good_account_snapshot: Dict[str, Any] = {
            "total_account_value": None,
            "buying_power": None,
            "holdings_sell_value": None,
            "holdings_buy_value": None,
            "percent_in_trade": None,
        }

        print("\n[PaperTrader] Started. Virtual account loaded.")
        print(f"[PaperTrader] Cash: ${self._account['cash']:,.2f}")
        print(f"[PaperTrader] Positions: {list(self._account['positions'].keys())}")
        print(f"[PaperTrader] Status file: {PAPER_TRADER_STATUS_PATH}\n")

    # ── account persistence ───────────────────────────────────────────────────

    def _load_account(self) -> dict:
        try:
            if os.path.isfile(PAPER_ACCOUNT_STATE_PATH):
                with open(PAPER_ACCOUNT_STATE_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data.setdefault("cash", float(PAPER_INITIAL_CASH))
                    data.setdefault("positions", {})
                    data.setdefault("dca_levels_triggered", {})
                    data.setdefault("trailing_pm", {})
                    return data
        except Exception:
            pass
        return {
            "cash": float(PAPER_INITIAL_CASH),
            "positions": {},
            "dca_levels_triggered": {},
            "trailing_pm": {},
        }

    def _save_account(self) -> None:
        """Persist simulated account state (cash + positions + DCA + trailing PM)."""
        self._account["dca_levels_triggered"] = dict(self.dca_levels_triggered)
        self._account["trailing_pm"] = {
            k: {ik: iv for ik, iv in v.items() if ik != "settings_sig"}
            for k, v in self.trailing_pm.items()
        }
        _atomic_write_json(PAPER_ACCOUNT_STATE_PATH, self._account)

    # ── PnL ledger ────────────────────────────────────────────────────────────

    def _load_pnl_ledger(self) -> dict:
        try:
            if os.path.isfile(PAPER_PNL_LEDGER_PATH):
                with open(PAPER_PNL_LEDGER_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                if not isinstance(data, dict):
                    data = {}
                data.setdefault("total_realized_profit_usd", 0.0)
                data.setdefault("last_updated_ts", time.time())
                data.setdefault("open_positions", {})
                return data
        except Exception:
            pass
        return {
            "total_realized_profit_usd": 0.0,
            "last_updated_ts": time.time(),
            "open_positions": {},
        }

    def _save_pnl_ledger(self) -> None:
        self._pnl_ledger["last_updated_ts"] = time.time()
        _atomic_write_json(PAPER_PNL_LEDGER_PATH, self._pnl_ledger)

    # ── trade recording ───────────────────────────────────────────────────────

    def _record_trade(
        self,
        side: str,
        symbol: str,
        qty: float,
        price: Optional[float] = None,
        avg_cost_basis: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        tag: Optional[str] = None,
        order_id: Optional[str] = None,
    ) -> None:
        ts = time.time()
        side_l = str(side or "").lower().strip()
        base = str(symbol or "").upper().split("-")[0].strip()

        self._pnl_ledger.setdefault("total_realized_profit_usd", 0.0)
        self._pnl_ledger.setdefault("open_positions", {})

        realized: Optional[float] = None

        if base and price is not None and qty > 0:
            open_pos = self._pnl_ledger.get("open_positions", {})
            pos = open_pos.get(base, {"usd_cost": 0.0, "qty": 0.0})

            pos_usd_cost = float(pos.get("usd_cost", 0.0) or 0.0)
            pos_qty      = float(pos.get("qty", 0.0) or 0.0)

            if side_l == "buy":
                usd_used = float(price) * float(qty)
                pos["usd_cost"] = pos_usd_cost + usd_used
                pos["qty"]      = pos_qty + float(qty)
                open_pos[base]  = pos
                self._save_pnl_ledger()

            elif side_l == "sell":
                frac = min(1.0, float(qty) / max(pos_qty, 1e-12))
                cost_used = pos_usd_cost * frac
                usd_got   = float(price) * float(qty)
                realized  = usd_got - cost_used

                pos["usd_cost"] = pos_usd_cost - cost_used
                pos["qty"]      = pos_qty - float(qty)
                if float(pos.get("qty", 0.0)) <= 1e-12 or float(pos.get("usd_cost", 0.0)) <= 1e-6:
                    open_pos.pop(base, None)
                else:
                    open_pos[base] = pos

                self._pnl_ledger["total_realized_profit_usd"] = (
                    float(self._pnl_ledger.get("total_realized_profit_usd", 0.0)) + float(realized)
                )
                self._save_pnl_ledger()

        entry = {
            "ts": ts,
            "side": side,
            "tag": tag,
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "avg_cost_basis": avg_cost_basis,
            "pnl_pct": pnl_pct,
            "realized_profit_usd": realized,
            "order_id": order_id,
            "paper": True,
        }
        _append_jsonl(PAPER_TRADE_HISTORY_PATH, entry)

    # ── simulated order execution ─────────────────────────────────────────────

    def _sim_buy(
        self,
        symbol: str,
        amount_usd: float,
        price: float,
        avg_cost_basis: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        tag: Optional[str] = None,
    ) -> bool:
        """Execute a paper buy. Returns True on success."""
        if price <= 0 or amount_usd <= 0:
            return False
        cash = float(self._account.get("cash", 0.0))
        if amount_usd > cash:
            print(f"  [PaperTrader] Insufficient cash: need ${amount_usd:.2f}, have ${cash:.2f}")
            return False

        qty = round(amount_usd / price, 8)
        if qty <= 0:
            return False

        # Update positions
        positions = self._account.setdefault("positions", {})
        pos = positions.get(symbol, {"qty": 0.0, "avg_cost": 0.0})
        old_qty   = float(pos.get("qty", 0.0))
        old_cost  = float(pos.get("avg_cost", 0.0))
        new_qty   = old_qty + qty
        new_cost  = ((old_cost * old_qty) + (price * qty)) / max(new_qty, 1e-12)
        positions[symbol] = {"qty": new_qty, "avg_cost": new_cost}
        self._account["cash"] = cash - amount_usd
        self._save_account()

        order_id = str(uuid.uuid4())
        self._record_trade(
            side="buy",
            symbol=f"{symbol}-USD",
            qty=qty,
            price=price,
            avg_cost_basis=avg_cost_basis,
            pnl_pct=pnl_pct,
            tag=tag,
            order_id=order_id,
        )
        return True

    def _sim_sell(
        self,
        symbol: str,
        qty: float,
        price: float,
        avg_cost_basis: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        tag: Optional[str] = None,
    ) -> bool:
        """Execute a paper sell. Returns True on success."""
        if price <= 0 or qty <= 0:
            return False

        positions = self._account.get("positions", {})
        pos = positions.get(symbol)
        if not pos or float(pos.get("qty", 0.0)) < qty:
            print(f"  [PaperTrader] Insufficient position to sell {qty} {symbol}")
            return False

        proceeds = price * qty
        pos["qty"] = float(pos["qty"]) - qty
        if float(pos["qty"]) <= 1e-10:
            positions.pop(symbol, None)
        self._account["cash"] = float(self._account.get("cash", 0.0)) + proceeds
        self._save_account()

        order_id = str(uuid.uuid4())
        self._record_trade(
            side="sell",
            symbol=f"{symbol}-USD",
            qty=qty,
            price=price,
            avg_cost_basis=avg_cost_basis,
            pnl_pct=pnl_pct,
            tag=tag,
            order_id=order_id,
        )
        return True

    # ── neural signal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _coin_folder(symbol: str) -> str:
        sym = str(symbol).upper().strip()
        if sym == "BTC":
            return main_dir
        sub = os.path.join(main_dir, sym)
        return sub if os.path.isdir(sub) else main_dir

    @staticmethod
    def _read_signal_file(path: str) -> int:
        try:
            with open(path, "r") as f:
                return int(float(f.read().strip()))
        except Exception:
            return 0

    @classmethod
    def _read_long_dca_signal(cls, symbol: str) -> int:
        return cls._read_signal_file(
            os.path.join(cls._coin_folder(symbol), "long_dca_signal.txt")
        )

    @classmethod
    def _read_short_dca_signal(cls, symbol: str) -> int:
        return cls._read_signal_file(
            os.path.join(cls._coin_folder(symbol), "short_dca_signal.txt")
        )

    @classmethod
    def _read_long_price_levels(cls, symbol: str) -> list:
        path = os.path.join(cls._coin_folder(symbol), "low_bound_prices.html")
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = (f.read() or "").strip()
            if not raw:
                return []
            raw = raw.strip().strip("[]()")
            raw = raw.replace(",", " ").replace(";", " ").replace("|", " ")
            raw = raw.replace("\n", " ").replace("\t", " ")
            vals = []
            seen: set = set()
            for p in raw.split():
                try:
                    v = float(p)
                    k = round(v, 12)
                    if k not in seen:
                        seen.add(k)
                        vals.append(v)
                except Exception:
                    continue
            vals.sort(reverse=True)
            return vals
        except Exception:
            return []

    # ── DCA window helpers (mirrors live trader) ──────────────────────────────

    def _dca_window_count(self, base: str, now_ts: Optional[float] = None) -> int:
        base = str(base).upper().strip()
        now  = float(now_ts if now_ts is not None else time.time())
        cutoff    = now - float(self.dca_window_seconds)
        last_sell = float(self._dca_last_sell_ts.get(base, 0.0) or 0.0)
        ts_list   = [t for t in self._dca_buy_ts.get(base, []) if t > last_sell and t >= cutoff]
        self._dca_buy_ts[base] = ts_list
        return len(ts_list)

    def _note_dca_buy(self, base: str, ts: Optional[float] = None) -> None:
        t = float(ts if ts is not None else time.time())
        self._dca_buy_ts.setdefault(base, []).append(t)
        self._dca_window_count(base, now_ts=t)

    def _reset_dca_window_for_trade(self, base: str, sold: bool = False, ts: Optional[float] = None) -> None:
        if sold:
            self._dca_last_sell_ts[base] = float(ts if ts is not None else time.time())
        self._dca_buy_ts[base] = []

    def _seed_dca_window_from_history(self) -> None:
        now_ts = time.time()
        cutoff = now_ts - float(self.dca_window_seconds)
        self._dca_buy_ts = {}
        self._dca_last_sell_ts = {}
        if not os.path.isfile(PAPER_TRADE_HISTORY_PATH):
            return
        try:
            with open(PAPER_TRADE_HISTORY_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = (line or "").strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    ts_f  = float(obj.get("ts", 0) or 0)
                    side  = str(obj.get("side", "")).lower()
                    tag   = obj.get("tag")
                    sym   = str(obj.get("symbol", "")).upper().split("-")[0].strip()
                    if not sym:
                        continue
                    if side == "sell":
                        if ts_f > float(self._dca_last_sell_ts.get(sym, 0.0)):
                            self._dca_last_sell_ts[sym] = ts_f
                    elif side == "buy" and tag == "DCA":
                        self._dca_buy_ts.setdefault(sym, []).append(ts_f)
        except Exception:
            return
        for base, ts_list in list(self._dca_buy_ts.items()):
            last_sell = float(self._dca_last_sell_ts.get(base, 0.0))
            kept = [t for t in ts_list if t > last_sell and t >= cutoff]
            kept.sort()
            self._dca_buy_ts[base] = kept

    # ── core trade loop ───────────────────────────────────────────────────────

    def manage_trades(self) -> None:  # noqa: C901 (complex but mirrors live trader intentionally)
        trades_made = False

        # ── hot-reload settings ───────────────────────────────────────────────
        try:
            _refresh_globals()
            self.dca_levels          = list(DCA_LEVELS)
            self.max_dca_buys_per_24h = int(MAX_DCA_BUYS_PER_24H)

            old_sig  = getattr(self, "_last_trailing_settings_sig", None)
            new_gap  = float(TRAILING_GAP_PCT)
            new_pm0  = float(PM_START_PCT_NO_DCA)
            new_pm1  = float(PM_START_PCT_WITH_DCA)
            self.trailing_gap_pct      = new_gap
            self.pm_start_pct_no_dca   = new_pm0
            self.pm_start_pct_with_dca = new_pm1
            new_sig = (new_gap, new_pm0, new_pm1)
            if old_sig is not None and new_sig != old_sig:
                self.trailing_pm = {}
            self._last_trailing_settings_sig = new_sig
        except Exception:
            pass

        # ── fetch prices for all tracked coins ────────────────────────────────
        current_buy_prices:  Dict[str, float] = {}
        current_sell_prices: Dict[str, float] = {}
        valid_symbols: list = []

        for sym in crypto_symbols:
            ask, bid = _fetch_price(sym)
            if ask > 0 and bid > 0:
                full = f"{sym}-USD"
                current_buy_prices[full]  = ask
                current_sell_prices[full] = bid
                valid_symbols.append(full)

        # ── account snapshot ──────────────────────────────────────────────────
        cash = float(self._account.get("cash", 0.0))
        positions_state = self._account.get("positions", {})

        holdings_buy_value  = 0.0
        holdings_sell_value = 0.0
        snapshot_ok = True

        for sym, pos in positions_state.items():
            qty = float(pos.get("qty", 0.0))
            if qty <= 0:
                continue
            full = f"{sym}-USD"
            bp = float(current_buy_prices.get(full, 0.0))
            sp = float(current_sell_prices.get(full, 0.0))
            if bp <= 0 or sp <= 0:
                snapshot_ok = False
                continue
            holdings_buy_value  += qty * bp
            holdings_sell_value += qty * sp

        total_account_value = cash + holdings_sell_value
        in_use = (holdings_sell_value / total_account_value * 100) if total_account_value > 0 else 0.0

        if (not snapshot_ok) or total_account_value <= 0:
            last = self._last_good_account_snapshot
            if last.get("total_account_value") is not None:
                total_account_value = float(last["total_account_value"])
                cash                = float(last.get("buying_power", cash))
                holdings_sell_value = float(last.get("holdings_sell_value", holdings_sell_value))
                holdings_buy_value  = float(last.get("holdings_buy_value", holdings_buy_value))
                in_use              = float(last.get("percent_in_trade", in_use))
        else:
            self._last_good_account_snapshot = {
                "total_account_value": total_account_value,
                "buying_power": cash,
                "holdings_sell_value": holdings_sell_value,
                "holdings_buy_value": holdings_buy_value,
                "percent_in_trade": in_use,
            }

        os.system("cls" if os.name == "nt" else "clear")
        print("\n--- [PAPER TRADING] Account Summary ---")
        print(f"Total Account Value: ${total_account_value:,.2f}  (virtual)")
        print(f"Holdings Value:      ${holdings_sell_value:,.2f}")
        print(f"Cash:                ${cash:,.2f}")
        print(f"Percent In Trade:    {in_use:.2f}%")
        print(
            f"Trailing PM: start +{self.pm_start_pct_no_dca:.2f}% (no DCA) "
            f"/ +{self.pm_start_pct_with_dca:.2f}% (with DCA) "
            f"| gap {self.trailing_gap_pct:.2f}%"
        )
        print("\n--- [PAPER TRADING] Current Trades ---")

        gui_positions: Dict[str, dict] = {}

        # ── process each held coin ────────────────────────────────────────────
        for sym, pos in list(positions_state.items()):
            quantity = float(pos.get("qty", 0.0))
            if quantity <= 0:
                continue

            full_symbol = f"{sym}-USD"
            if full_symbol not in valid_symbols:
                continue

            current_buy_price  = current_buy_prices.get(full_symbol, 0.0)
            current_sell_price = current_sell_prices.get(full_symbol, 0.0)
            avg_cost_basis     = float(pos.get("avg_cost", 0.0))

            if avg_cost_basis > 0:
                gain_loss_pct_buy  = ((current_buy_price  - avg_cost_basis) / avg_cost_basis) * 100
                gain_loss_pct_sell = ((current_sell_price - avg_cost_basis) / avg_cost_basis) * 100
            else:
                gain_loss_pct_buy  = 0.0
                gain_loss_pct_sell = 0.0

            value = quantity * current_sell_price
            triggered_levels_count = len(self.dca_levels_triggered.get(sym, []))

            # ── determine next DCA display ────────────────────────────────────
            next_stage = triggered_levels_count
            if self.dca_levels:
                hard_next = self.dca_levels[next_stage] if next_stage < len(self.dca_levels) else self.dca_levels[-1]
            else:
                hard_next = -2.5  # sensible default if list is somehow empty
            start_level     = max(1, min(int(TRADE_START_LEVEL), 7))
            neural_dca_max  = max(0, 7 - start_level)

            if next_stage < neural_dca_max:
                neural_next      = start_level + 1 + next_stage
                next_dca_display = f"{hard_next:.2f}% / N{neural_next}"
            else:
                next_dca_display = f"{hard_next:.2f}%"

            dca_line_price  = avg_cost_basis * (1.0 + hard_next / 100.0) if avg_cost_basis > 0 else 0.0
            dca_line_source = "HARD"
            dca_line_pct    = gain_loss_pct_buy

            if avg_cost_basis > 0 and next_stage < neural_dca_max:
                neural_level_disp = start_level + 1 + next_stage
                neural_levels     = self._read_long_price_levels(sym)
                if len(neural_levels) >= neural_level_disp:
                    neural_line_price = float(neural_levels[neural_level_disp - 1])
                    if neural_line_price > dca_line_price:
                        dca_line_price  = neural_line_price
                        dca_line_source = f"NEURAL N{neural_level_disp}"

            color  = Fore.GREEN if dca_line_pct  >= 0 else Fore.RED
            color2 = Fore.GREEN if gain_loss_pct_sell >= 0 else Fore.RED

            # ── trailing PM state ─────────────────────────────────────────────
            trail_status   = "N/A"
            trail_line_disp = 0.0
            trail_peak_disp = 0.0
            above_disp      = False
            dist_to_trail   = 0.0

            if avg_cost_basis > 0:
                pm_start_pct = (
                    self.pm_start_pct_no_dca if triggered_levels_count == 0
                    else self.pm_start_pct_with_dca
                )
                base_pm_line = avg_cost_basis * (1.0 + pm_start_pct / 100.0)
                trail_gap    = self.trailing_gap_pct / 100.0

                settings_sig = (
                    float(self.trailing_gap_pct),
                    float(self.pm_start_pct_no_dca),
                    float(self.pm_start_pct_with_dca),
                )

                state = self.trailing_pm.get(sym)
                if state is None or state.get("settings_sig") != settings_sig:
                    state = {
                        "active": False,
                        "line": base_pm_line,
                        "peak": 0.0,
                        "was_above": False,
                        "settings_sig": settings_sig,
                    }
                    self.trailing_pm[sym] = state
                else:
                    state["settings_sig"] = settings_sig
                    if not state.get("active", False):
                        state["line"] = base_pm_line
                    elif state.get("line", 0.0) < base_pm_line:
                        state["line"] = base_pm_line

                above_now = current_sell_price >= state["line"]

                if not state["active"] and above_now:
                    state["active"] = True
                    state["peak"]   = current_sell_price

                if state["active"]:
                    if current_sell_price > state["peak"]:
                        state["peak"] = current_sell_price
                    new_line = max(base_pm_line, state["peak"] * (1.0 - trail_gap))
                    if new_line > state["line"]:
                        state["line"] = new_line

                    # Trigger sell on cross from above → below trailing line
                    if state["was_above"] and current_sell_price < state["line"]:
                        print(
                            f"  [PaperTrader] Trailing PM hit for {sym}. "
                            f"Sell price {current_sell_price:.4f} < trailing line {state['line']:.4f}"
                        )
                        ok = self._sim_sell(
                            sym, quantity, current_sell_price,
                            avg_cost_basis=avg_cost_basis,
                            pnl_pct=gain_loss_pct_sell,
                            tag="TRAIL_SELL",
                        )
                        if ok:
                            trades_made = True
                            self.trailing_pm.pop(sym, None)
                            self._reset_dca_window_for_trade(sym, sold=True)
                            print(f"  [PaperTrader] Sold {quantity:.8f} {sym} (paper).")
                            continue

                state["was_above"] = above_now

                trail_line_disp = float(state.get("line", base_pm_line))
                trail_peak_disp = float(state.get("peak", 0.0))
                above_disp      = current_sell_price >= trail_line_disp
                trail_status    = "ON" if (state.get("active") or above_disp) else "OFF"
                if trail_line_disp > 0:
                    dist_to_trail = ((current_sell_price - trail_line_disp) / trail_line_disp) * 100.0

            # ── DCA logic ─────────────────────────────────────────────────────
            current_stage = len(self.dca_levels_triggered.get(sym, []))
            if self.dca_levels:
                hard_level = self.dca_levels[current_stage] if current_stage < len(self.dca_levels) else self.dca_levels[-1]
            else:
                hard_level = -2.5  # sensible default if list is somehow empty
            hard_hit      = gain_loss_pct_buy <= hard_level

            neural_hit = False
            if current_stage < MAX_NEURAL_DCA_STAGES:
                neural_level_needed = current_stage + MAX_NEURAL_DCA_STAGES
                neural_level_now    = self._read_long_dca_signal(sym)
                neural_hit = (gain_loss_pct_buy < 0) and (neural_level_now >= neural_level_needed)

            if hard_hit or neural_hit:
                dca_amount   = value * float(DCA_MULTIPLIER or 0.0)
                recent_dca   = self._dca_window_count(sym)

                if recent_dca >= self.max_dca_buys_per_24h:
                    print(f"  [PaperTrader] Skipping DCA for {sym}: 24h limit reached.")
                elif dca_amount > cash:
                    print(f"  [PaperTrader] Skipping DCA for {sym}: not enough cash (${dca_amount:.2f} > ${cash:.2f}).")
                else:
                    reason = "HARD" if hard_hit else "NEURAL"
                    print(f"  [PaperTrader] DCA buying {sym} (stage {current_stage + 1}) via {reason}.")
                    ok = self._sim_buy(
                        sym, dca_amount, current_buy_price,
                        avg_cost_basis=avg_cost_basis,
                        pnl_pct=gain_loss_pct_buy,
                        tag="DCA",
                    )
                    if ok:
                        trades_made = True
                        self.dca_levels_triggered.setdefault(sym, []).append(current_stage)
                        self._note_dca_buy(sym)
                        self.trailing_pm.pop(sym, None)
                        cash = float(self._account.get("cash", 0.0))
                        print(f"  [PaperTrader] DCA buy placed for {sym} (paper).")

            # ── print status ──────────────────────────────────────────────────
            print(
                f"\nSymbol: {sym}"
                f"  |  DCA: {color}{dca_line_pct:+.2f}%{Style.RESET_ALL}"
                f" @ {_fmt_price(current_buy_price)}"
                f"  |  Gain/Loss SELL: {color2}{gain_loss_pct_sell:+.2f}%{Style.RESET_ALL}"
                f" @ {_fmt_price(current_sell_price)}"
                f"  |  DCA Levels: {triggered_levels_count}"
                f"  |  Value: ${value:.2f}"
            )
            if avg_cost_basis > 0:
                print(f"  Trailing PM  |  Line: {_fmt_price(trail_line_disp)}  |  Above: {above_disp}  |  Status: {trail_status}")

            # ── write file with current price (mirrors live trader behavior) ──
            try:
                with open(f"{sym}_current_price.txt", "w") as f:
                    f.write(str(current_buy_price))
            except Exception:
                pass

            gui_positions[sym] = {
                "quantity": quantity,
                "avg_cost_basis": avg_cost_basis,
                "current_buy_price": current_buy_price,
                "current_sell_price": current_sell_price,
                "gain_loss_pct_buy": gain_loss_pct_buy,
                "gain_loss_pct_sell": gain_loss_pct_sell,
                "value_usd": value,
                "dca_triggered_stages": int(triggered_levels_count),
                "next_dca_display": next_dca_display,
                "dca_line_price": float(dca_line_price),
                "dca_line_source": dca_line_source,
                "dca_line_pct": float(dca_line_pct),
                "trail_active": trail_status == "ON",
                "trail_line": float(trail_line_disp),
                "trail_peak": float(trail_peak_disp),
                "dist_to_trail_pct": float(dist_to_trail),
            }

        # ── add bid/ask entries for untracked coins (GUI overlay lines) ────────
        for sym in crypto_symbols:
            if sym in gui_positions:
                continue
            full = f"{sym}-USD"
            if full not in valid_symbols:
                continue
            bp = current_buy_prices.get(full, 0.0)
            sp = current_sell_prices.get(full, 0.0)
            try:
                with open(f"{sym}_current_price.txt", "w") as f:
                    f.write(str(bp))
            except Exception:
                pass
            gui_positions[sym] = {
                "quantity": 0.0,
                "avg_cost_basis": 0.0,
                "current_buy_price": bp,
                "current_sell_price": sp,
                "gain_loss_pct_buy": 0.0,
                "gain_loss_pct_sell": 0.0,
                "value_usd": 0.0,
                "dca_triggered_stages": int(len(self.dca_levels_triggered.get(sym, []))),
                "next_dca_display": "",
                "dca_line_price": 0.0,
                "dca_line_source": "N/A",
                "dca_line_pct": 0.0,
                "trail_active": False,
                "trail_line": 0.0,
                "trail_peak": 0.0,
                "dist_to_trail_pct": 0.0,
            }

        # ── new trade entry (no position for this coin yet) ───────────────────
        holding_syms = set(positions_state.keys())

        for sym in crypto_symbols:
            if sym in holding_syms:
                continue

            full = f"{sym}-USD"
            if full not in valid_symbols:
                continue

            long_sig  = self._read_long_dca_signal(sym)
            short_sig = self._read_short_dca_signal(sym)
            start_level = max(1, min(int(TRADE_START_LEVEL), 7))

            if not (long_sig >= start_level and short_sig == 0):
                continue

            current_buy_price = current_buy_prices.get(full, 0.0)
            if current_buy_price <= 0:
                continue

            alloc_pct    = float(START_ALLOC_PCT or 0.005)
            alloc_usd    = max(0.5, total_account_value * (alloc_pct / 100.0))

            print(
                f"[PaperTrader] Starting new trade for {sym} "
                f"(AI long={long_sig}, short={short_sig}). Allocating ${alloc_usd:.2f}."
            )
            ok = self._sim_buy(sym, alloc_usd, current_buy_price)
            if ok:
                trades_made = True
                self.dca_levels_triggered[sym] = []
                self._reset_dca_window_for_trade(sym, sold=False)
                self.trailing_pm.pop(sym, None)
                cash = float(self._account.get("cash", 0.0))
                # Refresh position data for this sym so the loop above runs on next tick
                positions_state = self._account.get("positions", {})

        if trades_made:
            self._save_account()

        # ── write hub status files ────────────────────────────────────────────
        try:
            status = {
                "timestamp": time.time(),
                "paper": True,
                "account": {
                    "total_account_value": total_account_value,
                    "buying_power": cash,
                    "holdings_sell_value": holdings_sell_value,
                    "holdings_buy_value": holdings_buy_value,
                    "percent_in_trade": in_use,
                    "pm_start_pct_no_dca": float(self.pm_start_pct_no_dca),
                    "pm_start_pct_with_dca": float(self.pm_start_pct_with_dca),
                    "trailing_gap_pct": float(self.trailing_gap_pct),
                },
                "positions": gui_positions,
            }
            _append_jsonl(
                PAPER_ACCOUNT_VALUE_HISTORY_PATH,
                {"ts": status["timestamp"], "total_account_value": total_account_value},
            )
            _atomic_write_json(PAPER_TRADER_STATUS_PATH, status)
        except Exception:
            pass

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        while True:
            try:
                self.manage_trades()
                time.sleep(5)  # yfinance has 1-min candles; no need to hammer it
            except Exception:
                print(traceback.format_exc())


if __name__ == "__main__":
    bot = PaperCryptoTrading()
    bot.run()
