"""Grades every flag after the fact and compares it with the base rate of all other ticker-days.

A flag cannot be graded as 'right' or 'wrong' on direction, because the screener never claims one.
What it can be graded on: was the stock's move over the next N trading days unusually large relative to its
sector ETF and its own volatility? If flags are not followed by large moves more often than random
ticker-days, the screener is not telling you anything.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from .common import Paths, fnum, iso_utc, pct, list_sessions, parse_date, read_csv, read_csv_if_exists, write_csv, write_json
from .analyze import PriceCache

log = logging.getLogger("screener")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


_PAIRS: dict = {}


def _pair(prices: PriceCache, symbol: str, etf: str):
    key = (id(prices), symbol, etf)
    if key not in _PAIRS:
        s, e = prices.get(symbol), prices.get(etf)
        if s is None or e is None:
            _PAIRS[key] = None
        else:
            m = s[["Date", "Adj Close"]].merge(e[["Date", "Adj Close"]], on="Date", suffixes=("_s", "_e")).dropna()
            dates = m["Date"].tolist()
            _PAIRS[key] = (dates, {d: i for i, d in enumerate(dates)}, m["Adj Close_s"].to_numpy(), m["Adj Close_e"].to_numpy())
    return _PAIRS[key]


def outcome(prices: PriceCache, symbol: str, etf: str, session: str, h: int, sigma_window: int) -> dict | None:
    pair = _pair(prices, symbol, etf)
    if pair is None or session not in pair[1]:
        return None
    dates, idx, rs, re_ = pair
    i = idx[session]
    if i + h >= len(dates) or i < sigma_window:
        return None
    excess = (rs[i + h] / rs[i] - 1) - (re_[i + h] / re_[i] - 1)
    daily = np.diff(rs[i - sigma_window:i + 1]) / rs[i - sigma_window:i] - np.diff(re_[i - sigma_window:i + 1]) / re_[i - sigma_window:i]
    sd = float(np.std(daily, ddof=1))
    if not sd > 0:
        return None
    z = excess / (sd * math.sqrt(h))
    return {"end_date": dates[i + h], "return_stock": rs[i + h] / rs[i] - 1, "return_etf": re_[i + h] / re_[i] - 1,
            "excess_return": excess, "excess_z": z}


def evaluate(paths: Paths, settings: dict) -> dict:
    h, hit_sigma, win = int(settings["eval_horizon_days"]), float(settings["eval_hit_sigma"]), int(settings["eval_sigma_window"])
    prices = PriceCache(paths)
    # Outcomes are append-only: price files hold about one year, so an old outcome could not be recomputed.
    known = {}
    prev = read_csv_if_exists(paths.evaluation / "outcomes.csv")
    if prev is not None and len(prev):
        for _, r in prev.iterrows():
            known[(str(r["session"]), str(r["symbol"]))] = {
                k: r[k] for k in ("end_date", "return_stock", "return_etf", "excess_return", "excess_z")}
    rows = []
    for s in list_sessions(paths, "analysis.csv"):
        a = read_csv(paths.session(s) / "analysis.csv")
        a = a[(a["data_status"] == "OK")]
        for _, r in a.iterrows():
            o = known.get((s, str(r["symbol"])))
            if o is None:
                o = outcome(prices, r["symbol"], r["sector_etf"], s, h, win)
            if o is None:
                continue
            earn = r.get("next_earnings")
            earn_in = isinstance(earn, str) and s <= earn <= o["end_date"]  # after-close report on D counts
            rows.append({"session": s, "symbol": r["symbol"], "is_candidate": bool(r["is_candidate"]),
                         "is_elevated": bool(r["is_elevated"]), "earnings_in_window": earn_in, **o,
                         "hit": abs(o["excess_z"]) >= hit_sigma})
    out = pd.DataFrame(rows)
    paths.evaluation.mkdir(parents=True, exist_ok=True)
    write_csv(out, paths.evaluation / "outcomes.csv")

    def group(df: pd.DataFrame) -> dict:
        n, k = int(len(df)), int(df["hit"].sum()) if len(df) else 0
        lo, hi = wilson(k, n)
        return {"n": n, "hits": k, "hit_rate": k / n if n else math.nan, "ci95": [lo, hi],
                "mean_abs_excess_z": float(df["excess_z"].abs().mean()) if n else math.nan}

    summary = {"generated_at": iso_utc(), "horizon_trading_days": h,
               "hit_definition": f"|Überrendite ggü. Sektor-ETF nach {h} Handelstagen| >= {hit_sigma} Sigma "
                                 f"(Sigma aus {win} Tagen vor dem Signal)",
               "groups": {}}
    if len(out):
        for name, df in (("alle_tickertage", out), ("erhoehtes_volumen", out[out["is_elevated"]]),
                         ("kandidaten", out[out["is_candidate"]]),
                         ("kandidaten_ohne_quartalszahlen", out[out["is_candidate"] & ~out["earnings_in_window"]]),
                         ("alle_ohne_quartalszahlen", out[~out["earnings_in_window"]])):
            summary["groups"][name] = group(df)
        summary["first_session"], summary["last_session"] = out["session"].min(), out["session"].max()
    write_json(summary, paths.evaluation / "summary.json")
    (paths.evaluation / "summary.md").write_text(render_summary(summary))
    _PAIRS.clear()
    log.info("Auswertung: %d bewertbare Tickertage", len(out))
    return summary


def render_summary(s: dict) -> str:
    L = ["# Trefferquote des Screeners", "", f"Stand {s['generated_at']}. Treffer: {s['hit_definition']}.", ""]
    g = s.get("groups", {})
    if not g:
        L += ["Noch keine bewertbaren Tage. Die erste Bewertung ist möglich, sobald eine Analyse "
              f"{s['horizon_trading_days']} Handelstage alt ist.", ""]
        return "\n".join(L)
    L.append(f"Zeitraum der Signale: {s['first_session']} bis {s['last_session']}.")
    L += ["", "| Gruppe | Anzahl | Treffer | Quote | 95-%-Intervall | mittleres abs. z |", "|---|---:|---:|---:|---|---:|"]
    labels = {"alle_tickertage": "Basisrate: alle Tickertage", "erhoehtes_volumen": "Erhöhtes Optionsvolumen",
              "kandidaten": "Kandidaten", "kandidaten_ohne_quartalszahlen": "Kandidaten ohne Quartalszahlen im Fenster",
              "alle_ohne_quartalszahlen": "Basisrate ohne Quartalszahlen im Fenster"}
    for key, label in labels.items():
        x = g.get(key)
        if not x:
            continue
        rate = "n/a" if x["n"] == 0 else pct(x['hit_rate'], 1)
        ci = "n/a" if x["n"] == 0 else f"{pct(x['ci95'][0], 1)} bis {pct(x['ci95'][1], 1)}"
        mz = "n/a" if x["n"] == 0 else fnum(x['mean_abs_excess_z'], 2)
        L.append(f"| {label} | {x['n']} | {x['hits']} | {rate} | {ci} | {mz} |")
    L += ["", "Lesart: Nur wenn die Quote der Kandidaten klar über der Basisrate liegt und sich die Intervalle kaum "
          "überschneiden, sagt der Screener mehr als der Zufall. Bei weniger als etwa 30 Kandidaten ist jede Aussage "
          "vorläufig.", ""]
    return "\n".join(L)
