"""Deterministic fake market that imitates what yfinance returns, including Yahoo's quirks:

* evening: options volume of the session, open interest still at the start-of-day level
* morning before the open: open interest updated for the previous session, expired series gone
* contracts that did not trade keep showing an old volume with an old lastTradeDate
"""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone

import numpy as np
import pandas as pd

from screener.common import ET
from screener.provider import ProviderError

UNIVERSE = [
    ("AAA", "Alpha", "Information Technology", "XLK"),
    ("BBB", "Beta", "Information Technology", "XLK"),
    ("CCC", "Gamma", "Information Technology", "XLK"),
    ("DDD", "Delta", "Financials", "XLF"),
    ("EEE", "Epsilon", "Financials", "XLF"),
    ("FFF", "Phi", "Financials", "XLF"),
]
BENCH = ["SPY", "XLK", "XLF"]
STRIKES = [90, 95, 100, 105, 110]


def business_days(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


class FakeMarket:
    def __init__(self, start: date = date(2025, 3, 3), n_days: int = 400, anomalies: dict | None = None):
        self.days = business_days(start, n_days)
        self.day_idx = {d: i for i, d in enumerate(self.days)}
        self.anomalies = anomalies or {}  # (symbol, date) -> dict(extra, net_share, price_jump, drift_after)
        self.symbols = [u[0] for u in UNIVERSE] + BENCH
        self.prices = {s: self._make_prices(s) for s in self.symbols}
        self._oi_cache: dict = {}
        self.stale_oi_for: set = set()  # sessions for which the morning feed still shows old OI
        self.stale_symbols: set = set()  # symbols whose morning OI is still the old one
        self.fail_chains = False  # every option chain request fails
        self.history_gaps: set = set()  # sessions missing from the daily price history (Yahoo lag)

    def _seed(self, *parts) -> int:
        return abs(hash(("fm",) + parts)) % (2**32)

    def _make_prices(self, sym: str) -> pd.DataFrame:
        rng = np.random.default_rng(self._seed(sym))
        rets = rng.normal(0.0003, 0.012, len(self.days))
        for (s, d), a in self.anomalies.items():
            if s != sym:
                continue
            i = self.day_idx[d]
            rets[i] = a.get("price_jump", 0.0)
            drift = a.get("drift_after", 0.0)
            for j in range(i + 1, min(i + 11, len(rets))):
                rets[j] = drift / 10 + rng.normal(0, 0.002)
        close = 100 * np.exp(np.cumsum(rets))
        vol = rng.integers(1_000_000, 3_000_000, len(self.days)).astype(float)
        return pd.DataFrame({"Date": [d.isoformat() for d in self.days], "Open": close, "High": close * 1.01,
                             "Low": close * 0.99, "Close": close, "Adj Close": close, "Volume": vol})

    # ---- options model --------------------------------------------------------------------------
    def expirations_on(self, d: date) -> list[str]:
        out, x = [], d
        while x <= d + timedelta(days=130):
            if x.weekday() == 4 and x >= d:
                out.append(x.isoformat())
            x += timedelta(days=1)
        return out

    def contract(self, sym: str, expiry: str, typ: str, strike: int) -> str:
        return f"{sym}{expiry.replace('-', '')[2:]}{'C' if typ == 'call' else 'P'}{strike:08d}"

    def session_volume(self, sym: str, expiry: str, typ: str, strike: int, d: date) -> tuple[int, int]:
        """(volume, net OI change) of one contract on session d."""
        if expiry < d.isoformat():
            return 0, 0
        rng = np.random.default_rng(self._seed(sym, expiry, typ, strike, d.isoformat()))
        base = 40 if strike == 100 else 15
        vol = int(rng.poisson(base * rng.lognormal(0, 0.25)))
        if strike == 90:  # deep strike trades only sometimes
            vol = vol if rng.random() < 0.3 else 0
        net = int(round(vol * 0.1))
        a = self.anomalies.get((sym, d))
        if a and typ == "call" and strike == 100 and expiry == self.expirations_on(d)[1]:
            extra = int(a["extra"])
            vol += extra
            net += int(round(extra * a["net_share"]))
        return vol, net

    def oi_after(self, sym: str, expiry: str, typ: str, strike: int, d: date) -> int:
        key = (sym, expiry, typ, strike, d)
        if key not in self._oi_cache:
            prev = self.days[self.day_idx[d] - 1]
            before = 10_000 if self.day_idx[d] == 0 else self.oi_after(sym, expiry, typ, strike, prev)
            _, net = self.session_volume(sym, expiry, typ, strike, d)
            self._oi_cache[key] = max(0, before + net)
        return self._oi_cache[key]


class FakeProvider:
    """Same interface as screener.provider.YahooProvider, driven by a simulated clock."""

    source = "FakeMarket (Test)"

    def __init__(self, market: FakeMarket, now_utc: datetime):
        self.m = market
        self.now = now_utc
        self.request_count = 0
        et = now_utc.astimezone(ET)
        today = et.date()
        self.after_close = et.time() >= time(16, 0)
        past = [d for d in self.m.days if d < today]
        self.today_is_session = today in self.m.day_idx
        # the session whose volume the feed shows, and the session after which OI is shown
        if self.today_is_session and self.after_close:
            self.vol_session = today
            self.oi_session = past[-1]
        elif self.today_is_session and et.time() >= time(9, 30):
            raise RuntimeError("intraday not simulated")
        else:
            self.vol_session = past[-1]
            self.oi_session = past[-1]
        self.today = today

    def history(self, sym: str, period: str = "1y") -> pd.DataFrame:
        self.request_count += 1
        df = self.m.prices[sym]
        last = self.vol_session.isoformat()
        df = df[~df["Date"].isin({d.isoformat() for d in self.m.history_gaps})]
        return df[df["Date"] <= last].tail(252).reset_index(drop=True)

    def expirations(self, sym: str) -> list[str]:
        self.request_count += 1
        if sym in BENCH:
            return []
        return [e for e in self.m.expirations_on(self.today) if e >= self.today.isoformat()] if self.after_close \
            else [e for e in self.m.expirations_on(self.today) if e > self.vol_session.isoformat()]

    def chain(self, sym: str, expiry: str):
        self.request_count += 1
        if expiry not in self.expirations(sym):
            raise ProviderError(f"Optionskette {sym} {expiry}: Expiration `{expiry}` cannot be found")
        if self.m.fail_chains:
            raise ProviderError(f"Optionskette {sym} {expiry}: HTTPError 503")
        oi_d = self.oi_session
        if not self.after_close and (self.oi_session in self.m.stale_oi_for or sym in self.m.stale_symbols):
            oi_d = self.m.days[self.m.day_idx[self.oi_session] - 1]
        frames = {}
        close_ts = datetime.combine(self.vol_session, time(15, 59), tzinfo=ET).astimezone(timezone.utc)
        old_ts = close_ts - timedelta(days=3)
        for typ in ("call", "put"):
            rows = []
            for k in STRIKES:
                vol, _ = self.m.session_volume(sym, expiry, typ, k, self.vol_session)
                traded = vol > 0
                rows.append({
                    "contractSymbol": self.m.contract(sym, expiry, typ, k),
                    "lastTradeDate": close_ts if traded else old_ts,
                    "strike": float(k), "lastPrice": 1.0, "bid": 0.9, "ask": 1.1,
                    "volume": float(vol) if traded else 7.0,  # Yahoo quirk: stale volume on untraded contracts
                    "openInterest": float(self.m.oi_after(sym, expiry, typ, k, oi_d)),
                    "impliedVolatility": 0.3, "inTheMoney": (k < 100) if typ == "call" else (k > 100),
                    "contractSize": "REGULAR", "currency": "USD",
                })
            frames[typ] = pd.DataFrame(rows)
        px = float(self.m.prices[sym].set_index("Date").loc[self.vol_session.isoformat(), "Close"])
        return frames["call"], frames["put"], {"regularMarketPrice": px}

    def calendar(self, sym: str) -> dict:
        self.request_count += 1
        if sym == "EEE":
            return {"Ex-Dividend Date": self.vol_session + timedelta(days=2),
                    "Earnings Date": [self.vol_session + timedelta(days=40)]}
        return {"Earnings Date": [self.vol_session + timedelta(days=60)]}
