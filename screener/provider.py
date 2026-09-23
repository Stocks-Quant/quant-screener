"""Thin wrapper around yfinance with retries, pacing and normalised outputs.

Anything that goes wrong raises ProviderError. Callers turn that into UNAVAILABLE; nothing is estimated.
The test-suite swaps this class for a fake with the same four methods.
"""
from __future__ import annotations

import logging
import time

import pandas as pd

log = logging.getLogger("screener")

PRICE_COLS = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]  # plus "Provisional"
CHAIN_COLS = ["contractSymbol", "lastTradeDate", "strike", "lastPrice", "bid", "ask", "volume",
              "openInterest", "impliedVolatility", "inTheMoney"]


class ProviderError(Exception):
    pass


def normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        raise ProviderError("leere Kurshistorie")
    out = df.copy()
    idx = out.index
    if getattr(idx, "tz", None) is not None:
        dates = [ts.date().isoformat() for ts in idx]  # index is in exchange time (US/Eastern)
    else:
        dates = [pd.Timestamp(ts).date().isoformat() for ts in idx]
    out = out.reset_index(drop=True)
    out.insert(0, "Date", dates)
    if "Adj Close" not in out.columns:
        out["Adj Close"] = out["Close"]
    out = out[PRICE_COLS].dropna(subset=["Close"])
    out = out.drop_duplicates("Date", keep="last").reset_index(drop=True)
    out["Provisional"] = False
    return out


def complete_with_quote(hist: pd.DataFrame, quote: dict) -> pd.DataFrame:
    """Yahoo's daily history can lag the session by hours while the quote already has the day's close.

    If the quote belongs to a later trading day than the last daily bar, append that day as a bar built from
    the quote (open unknown, adjusted close = close) and mark it Provisional. The next fetch replaces it.
    """
    t = quote.get("regularMarketTime")
    price = quote.get("regularMarketPrice")
    if t is None or price is None or len(hist) == 0:
        return hist
    ts = pd.Timestamp(t)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
    qd = ts.tz_convert("America/New_York").date().isoformat()
    if qd <= hist["Date"].iloc[-1]:
        return hist
    row = {"Date": qd, "Open": float("nan"), "High": quote.get("regularMarketDayHigh", float("nan")),
           "Low": quote.get("regularMarketDayLow", float("nan")), "Close": float(price), "Adj Close": float(price),
           "Volume": quote.get("regularMarketVolume", float("nan")), "Provisional": True}
    return pd.concat([hist, pd.DataFrame([row])], ignore_index=True)


class YahooProvider:
    def __init__(self, pause: float = 0.35, retries: int = 3):
        import yfinance as yf

        self.yf = yf
        self.pause = pause
        self.retries = retries
        self._tickers: dict = {}
        self.request_count = 0
        self.source = f"Yahoo Finance via yfinance {yf.__version__} (inoffizielle Schnittstelle, Daten verzögert)"

    def _ticker(self, sym: str):
        if sym not in self._tickers:
            self._tickers[sym] = self.yf.Ticker(sym)
        return self._tickers[sym]

    def _call(self, what: str, fn, sym: str | None = None, accept=None):
        """Run fn with retries. `accept(result)` False counts as a failed attempt (e.g. an empty answer).

        A ValueError saying an expiration 'cannot be found' is permanent only when Yahoo did list other
        expirations; with an empty list it is a failed download (yfinance swallows the HTTP error) and is retried.
        """
        last = None
        for attempt in range(self.retries):
            self.request_count += 1
            try:
                res = fn()
                time.sleep(self.pause)
                if accept is None or accept(res):
                    return res
                last = ProviderError("leere Antwort")
            except ValueError as e:
                msg = str(e)
                if "cannot be found" in msg and "Available expirations are: []" not in msg:
                    raise ProviderError(f"{what}: Verfallstermin nicht mehr gelistet") from e
                last = e
            except Exception as e:  # network, rate limit, parsing
                last = e
            rate_limited = "RateLimit" in type(last).__name__ or "Too Many" in str(last)
            wait = 60 * (attempt + 1) if rate_limited else 3 * (attempt + 1)
            log.warning("%s fehlgeschlagen (Versuch %d/%d): %s: %s; warte %ss",
                        what, attempt + 1, self.retries, type(last).__name__, last, wait)
            if sym is not None:
                self._tickers.pop(sym, None)  # fresh Ticker object, so cached empty state is dropped
            time.sleep(wait)
        raise ProviderError(f"{what}: {type(last).__name__}: {last}")

    def history(self, sym: str, period: str = "1y") -> pd.DataFrame:
        df = self._call(f"Kurshistorie {sym}", lambda: self._ticker(sym).history(
            period=period, interval="1d", auto_adjust=False, actions=False, raise_errors=True), sym=sym,
            accept=lambda d: d is not None and len(d) > 0)
        return complete_with_quote(normalize_history(df), self.quote(sym))

    def quote(self, sym: str) -> dict:
        """Quote fields from the metadata of the last history request (no extra request)."""
        try:
            md = self._ticker(sym).history_metadata
            keys = ("regularMarketTime", "regularMarketPrice", "regularMarketDayHigh", "regularMarketDayLow",
                    "regularMarketVolume")
            return {k: md.get(k) for k in keys if md.get(k) is not None}
        except Exception as e:  # metadata is optional; history without it is still valid
            log.warning("Kursquote %s nicht lesbar: %s", sym, e)
            return {}

    def expirations(self, sym: str) -> list[str]:
        # Every stock in the universe has listed options, so an empty tuple means a failed download.
        res = self._call(f"Verfallstermine {sym}", lambda: self._ticker(sym).options, sym=sym,
                         accept=lambda r: bool(r))
        return list(res)

    def chain(self, sym: str, expiry: str):
        oc = self._call(f"Optionskette {sym} {expiry}", lambda: self._ticker(sym).option_chain(expiry), sym=sym,
                        accept=lambda o: o.calls is not None and o.puts is not None)
        return oc.calls, oc.puts, (oc.underlying or {})

    def calendar(self, sym: str) -> dict:
        res = self._call(f"Kalender {sym}", lambda: self._ticker(sym).calendar, sym=sym)
        return dict(res or {})
