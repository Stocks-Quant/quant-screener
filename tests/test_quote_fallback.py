import sys
import types

import pandas as pd

from screener.provider import YahooProvider, complete_with_quote, normalize_history


def _hist(dates):
    idx = pd.DatetimeIndex(pd.to_datetime(dates).tz_localize("America/New_York"), name="Date")
    n = len(dates)
    return pd.DataFrame({"Open": [1.0] * n, "High": [1.0] * n, "Low": [1.0] * n, "Close": [1.0] * n,
                         "Adj Close": [1.0] * n, "Volume": [10] * n}, index=idx)


def test_missing_session_bar_is_filled_from_quote():
    h = normalize_history(_hist(["2026-09-18", "2026-09-21"]))
    q = {"regularMarketTime": pd.Timestamp("2026-09-22 16:00", tz="America/New_York"), "regularMarketPrice": 2.0,
         "regularMarketDayHigh": 2.1, "regularMarketDayLow": 1.9, "regularMarketVolume": 99}
    out = complete_with_quote(h, q)
    assert list(out["Date"]) == ["2026-09-18", "2026-09-21", "2026-09-22"]
    last = out.iloc[-1]
    assert last["Close"] == 2.0 and bool(last["Provisional"]) and not out["Provisional"].iloc[:-1].any()


def test_quote_for_existing_or_older_day_changes_nothing():
    h = normalize_history(_hist(["2026-09-21", "2026-09-22"]))
    q = {"regularMarketTime": pd.Timestamp("2026-09-22 16:00", tz="America/New_York"), "regularMarketPrice": 5.0}
    assert complete_with_quote(h, q).equals(h)
    assert complete_with_quote(h, {}).equals(h)


def test_provider_uses_history_metadata(monkeypatch):
    class T:
        def __init__(self, sym):
            self.history_metadata = {"regularMarketTime": pd.Timestamp("2026-09-22 16:00", tz="America/New_York"),
                                     "regularMarketPrice": 3.0}

        def history(self, **kw):
            return _hist(["2026-09-21"])

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=T, __version__="stub"))
    p = YahooProvider(pause=0, retries=1)
    h = p.history("SPY")
    assert list(h["Date"]) == ["2026-09-21", "2026-09-22"] and bool(h["Provisional"].iloc[-1])
