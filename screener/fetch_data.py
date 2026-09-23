"""Agent 1: the data agent. It gathers, it never analyses.

Two runs per trading day, because volume and open interest become final at different times:

* evening (after the US close): daily bar, stock volume, per-contract options volume of the session,
  open interest as shown during the session (= start-of-day OI), event calendar.
* morning (next trading day, before the open): open interest after the session, which the OCC
  publishes overnight. Only this allows telling opened from closed positions.

Every figure carries source and timestamp. Missing data is written as UNAVAILABLE, never filled.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .common import (ET, UNAVAILABLE, Paths, et_date, iso_utc, load_benchmarks, load_universe,
                     list_sessions, parse_date, read_csv_if_exists, read_json, write_csv, write_json, write_prices)
from .events import structural_events
from .provider import CHAIN_COLS, ProviderError

log = logging.getLogger("screener")

SUMMARY_COLS = [
    "session", "symbol", "yahoo_symbol", "name", "sector", "sector_etf",
    "price_status", "price_reason", "open", "high", "low", "close", "prev_close", "return_1d",
    "stock_volume", "avg_stock_volume_30d", "rel_stock_volume_30d", "price_source", "price_fetched_at",
    "options_status", "options_reason", "call_volume", "put_volume", "total_option_volume",
    "volume_expiring_today", "call_oi_start", "put_oi_start", "n_expiries", "n_expiries_failed",
    "n_contracts_stale_volume", "options_last_trade_et", "max_expiry_days", "options_source",
    "options_fetched_at",
    "calendar_status", "earnings_dates", "ex_dividend_date", "dividend_date", "calendar_source",
    "calendar_fetched_at",
    "structural_events",
]
EXP_COLS = ["session", "symbol", "expiry", "type", "volume", "oi_start", "n_contracts", "n_oi_missing",
            "status", "fetched_at"]
TOP_COLS = ["session", "symbol", "rank", "contract", "type", "expiry", "strike", "volume", "oi_start",
            "last_trade_et", "bid", "ask", "last_price", "implied_vol", "in_the_money", "underlying_price",
            "fetched_at"]
MORNING_EXP_COLS = ["session", "symbol", "expiry", "type", "oi_after", "n_contracts", "n_oi_missing",
                    "status", "fetched_at"]
MORNING_TOP_COLS = ["session", "symbol", "contract", "expiry", "oi_after", "note", "fetched_at"]


def contracts_frame(calls: pd.DataFrame, puts: pd.DataFrame, expiry: str) -> pd.DataFrame:
    frames = []
    for typ, df in (("call", calls), ("put", puts)):
        if df is None or len(df) == 0:
            continue
        d = df.reindex(columns=CHAIN_COLS).copy()
        d["type"] = typ
        d["expiry"] = expiry
        frames.append(d)
    if not frames:
        return pd.DataFrame(columns=CHAIN_COLS + ["type", "expiry", "last_trade_et", "last_trade_date_et"])
    out = pd.concat(frames, ignore_index=True)
    for col in ("strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["lastTradeDate"] = pd.to_datetime(out["lastTradeDate"], utc=True, errors="coerce")
    out["last_trade_et"] = out["lastTradeDate"].dt.tz_convert(ET)
    out["last_trade_date_et"] = out["last_trade_et"].dt.strftime("%Y-%m-%d")
    return out


def _oi_sum(s: pd.Series) -> float:
    return float(s.sum()) if s.notna().any() else math.nan


def _nan_dict(keys) -> dict:
    return {k: math.nan for k in keys}


# ---------------------------------------------------------------------------------------------- evening

def _price_block(hist: pd.DataFrame, session: str, source: str, fetched_at: str) -> dict:
    keys = ["open", "high", "low", "close", "prev_close", "return_1d", "stock_volume",
            "avg_stock_volume_30d", "rel_stock_volume_30d"]
    res = _nan_dict(keys)
    res.update(price_source=source, price_fetched_at=fetched_at)
    dates = hist["Date"].tolist()
    if session not in dates:
        res.update(price_status="UNAVAILABLE", price_reason=f"kein Tagesbalken für {session}")
        return res
    i = dates.index(session)
    bar = hist.iloc[i]
    res.update(open=bar["Open"], high=bar["High"], low=bar["Low"], close=bar["Close"],
               stock_volume=bar["Volume"], price_status="OK", price_reason="")
    if i >= 1:
        res["prev_close"] = hist["Close"].iloc[i - 1]
        res["return_1d"] = hist["Adj Close"].iloc[i] / hist["Adj Close"].iloc[i - 1] - 1
    if i >= 30:
        avg = hist["Volume"].iloc[i - 30:i].mean()
        res["avg_stock_volume_30d"] = avg
        res["rel_stock_volume_30d"] = bar["Volume"] / avg if avg and avg > 0 else math.nan
    else:
        res["price_reason"] = "weniger als 30 Handelstage Historie für Volumendurchschnitt"
    return res


def _options_evening(provider, symbol: str, ysym: str, session: str, settings: dict, fetched_at: str):
    d0 = parse_date(session)
    max_d = d0 + timedelta(days=int(settings["max_expiry_days"]))
    totals = _nan_dict(["call_volume", "put_volume", "total_option_volume", "volume_expiring_today",
                        "call_oi_start", "put_oi_start"])
    totals.update(n_expiries=0, n_expiries_failed=0, n_contracts_stale_volume=0, options_last_trade_et=math.nan,
                  max_expiry_days=settings["max_expiry_days"], options_source=provider.source,
                  options_fetched_at=fetched_at)
    exp_rows, top_rows = [], []

    try:
        all_exps = provider.expirations(ysym)
    except ProviderError as e:
        totals.update(options_status="UNAVAILABLE", options_reason=str(e))
        return totals, exp_rows, top_rows
    sel = [e for e in all_exps if d0 <= parse_date(e) <= max_d]
    if not sel:
        totals.update(options_status="UNAVAILABLE", options_reason="keine gelisteten Verfallstermine im Fenster")
        return totals, exp_rows, top_rows

    frames, failed = [], []
    underlying_price = math.nan
    for e in sel:
        try:
            calls, puts, und = provider.chain(ysym, e)
        except ProviderError as ex:
            failed.append(e)
            for typ in ("call", "put"):
                exp_rows.append({"session": session, "symbol": symbol, "expiry": e, "type": typ,
                                 "volume": math.nan, "oi_start": math.nan, "n_contracts": math.nan,
                                 "n_oi_missing": math.nan, "status": f"UNAVAILABLE: {ex}", "fetched_at": fetched_at})
            continue
        if und.get("regularMarketPrice") is not None:
            underlying_price = float(und["regularMarketPrice"])
        c = contracts_frame(calls, puts, e)
        # Yahoo keeps showing a contract's last volume on days it did not trade. Only count trades of this session.
        traded_today = c["last_trade_date_et"] == session
        c["session_volume"] = np.where(traded_today, c["volume"].fillna(0), 0.0)
        totals["n_contracts_stale_volume"] += int(((~traded_today) & (c["volume"].fillna(0) > 0)).sum())
        frames.append(c)
        for typ in ("call", "put"):
            sub = c[c["type"] == typ]
            exp_rows.append({"session": session, "symbol": symbol, "expiry": e, "type": typ,
                             "volume": float(sub["session_volume"].sum()), "oi_start": _oi_sum(sub["openInterest"]),
                             "n_contracts": int(len(sub)), "n_oi_missing": int(sub["openInterest"].isna().sum()),
                             "status": "OK", "fetched_at": fetched_at})

    totals["n_expiries"] = len(sel)
    totals["n_expiries_failed"] = len(failed)
    if not frames:
        totals.update(options_status="UNAVAILABLE", options_reason="keine Optionskette abrufbar")
        return totals, exp_rows, top_rows

    allc = pd.concat(frames, ignore_index=True)
    last_trade = allc["last_trade_et"].max()
    totals["options_last_trade_et"] = last_trade.isoformat() if pd.notna(last_trade) else math.nan
    last_session = last_trade.strftime("%Y-%m-%d") if pd.notna(last_trade) else None

    if failed:
        totals.update(options_status="PARTIAL",
                      options_reason=f"{len(failed)} von {len(sel)} Verfallsterminen nicht abrufbar: {', '.join(failed)}")
    elif last_session != session:
        totals.update(options_status="STALE",
                      options_reason=f"letzter Optionshandel am {last_session}, nicht am {session}")
    else:
        totals.update(options_status="OK", options_reason="")

    if totals["options_status"] == "OK":
        calls_ = allc[allc["type"] == "call"]
        puts_ = allc[allc["type"] == "put"]
        totals["call_volume"] = float(calls_["session_volume"].sum())
        totals["put_volume"] = float(puts_["session_volume"].sum())
        totals["total_option_volume"] = totals["call_volume"] + totals["put_volume"]
        totals["volume_expiring_today"] = float(allc.loc[allc["expiry"] == session, "session_volume"].sum())
        totals["call_oi_start"] = _oi_sum(calls_.loc[calls_["expiry"] > session, "openInterest"])
        totals["put_oi_start"] = _oi_sum(puts_.loc[puts_["expiry"] > session, "openInterest"])

    # Contracts expiring on the session day cannot show an OI change afterwards, so they are not top candidates.
    top = allc[(allc["session_volume"] > 0) & (allc["expiry"] > session)].sort_values(
        "session_volume", ascending=False).head(
        int(settings["top_contracts"]))
    for rank, (_, r) in enumerate(top.iterrows(), start=1):
        top_rows.append({
            "session": session, "symbol": symbol, "rank": rank, "contract": r["contractSymbol"], "type": r["type"],
            "expiry": r["expiry"], "strike": r["strike"], "volume": r["session_volume"], "oi_start": r["openInterest"],
            "last_trade_et": r["last_trade_et"].isoformat() if pd.notna(r["last_trade_et"]) else math.nan,
            "bid": r["bid"], "ask": r["ask"], "last_price": r["lastPrice"], "implied_vol": r["impliedVolatility"],
            "in_the_money": r["inTheMoney"], "underlying_price": underlying_price, "fetched_at": fetched_at,
        })
    return totals, exp_rows, top_rows


def _calendar_block(provider, ysym: str, fetched_at: str) -> dict:
    res = {"calendar_status": "OK", "earnings_dates": math.nan, "ex_dividend_date": math.nan,
           "dividend_date": math.nan, "calendar_source": provider.source, "calendar_fetched_at": fetched_at}
    try:
        cal = provider.calendar(ysym)
    except ProviderError as e:
        res["calendar_status"] = f"UNAVAILABLE: {e}"
        return res
    earn = cal.get("Earnings Date") or []
    if earn:
        res["earnings_dates"] = ";".join(sorted(str(d)[:10] for d in earn))
    if cal.get("Ex-Dividend Date"):
        res["ex_dividend_date"] = str(cal["Ex-Dividend Date"])[:10]
    if cal.get("Dividend Date"):
        res["dividend_date"] = str(cal["Dividend Date"])[:10]
    return res


def run_evening(paths: Paths, provider, settings: dict, now_utc: datetime, allow_past: bool = False,
                symbols: list[str] | None = None) -> dict:
    started = iso_utc(now_utc)
    et_now = now_utc.astimezone(ET)
    today = et_now.date().isoformat()
    spy = provider.history("SPY", "1y")
    session = spy["Date"].iloc[-1]
    late = False
    if session != today and not allow_past:
        # A delayed scheduled run after midnight New York may still record the previous session: volume stays until
        # the next open. The window ends early, before the source publishes the overnight open interest.
        prev_day = (et_now.date() - timedelta(days=1)).isoformat()
        cutoff = int(settings.get("late_evening_cutoff_hour_et", 3))
        if et_now.hour < cutoff and session >= prev_day and not (paths.session(session) / "evening_meta.json").exists():
            late = True
            log.info("Verspäteter Abendlauf nach Mitternacht New York: erfasse %s.", session)
        else:
            log.info("Kein Handelstag heute (%s); letzter Tagesbalken %s. Nichts zu tun.", today, session)
            return {"status": "SKIPPED", "reason": f"kein Tagesbalken für {today}", "session": session}
    prev_meta = read_json(paths.session(session) / "evening_meta.json")
    if prev_meta and not allow_past and not symbols and set(prev_meta.get("options_status_counts", {})) == {"OK"} \
            and prev_meta.get("price_unavailable", 1) == 0:
        # Several evening schedules exist as a backup against dropped GitHub runs; the first complete one wins.
        log.info("Abenddaten für %s bereits vollständig erfasst.", session)
        return {"status": "SKIPPED", "reason": "Abenddaten bereits vollständig", "session": session}

    universe = load_universe(paths)
    if symbols:
        universe = universe[universe["symbol"].isin(symbols)]
    benchmarks = load_benchmarks(paths)
    errors = []

    fetched = iso_utc()
    write_prices(paths, "SPY", spy)
    for _, b in benchmarks.iterrows():
        if b["symbol"] == "SPY":
            continue
        try:
            write_prices(paths, b["symbol"], provider.history(b["yahoo_symbol"], "1y"))
        except ProviderError as e:
            errors.append(f"{b['symbol']}: {e}")

    s_events = structural_events(session, int(settings["event_lookahead_days"]))
    s_events_str = "; ".join(f"{e['date']} {e['label']}" for e in s_events)
    rows, exp_rows, top_rows = [], [], []
    for n, (_, u) in enumerate(universe.iterrows(), start=1):
        sym, ysym = u["symbol"], u["yahoo_symbol"]
        log.info("[%d/%d] %s", n, len(universe), sym)
        row = {"session": session, "symbol": sym, "yahoo_symbol": ysym, "name": u["name"], "sector": u["sector"],
               "sector_etf": u["sector_etf"], "structural_events": s_events_str}
        fetched = iso_utc()
        try:
            hist = provider.history(ysym, "1y")
            write_prices(paths, sym, hist)
            row.update(_price_block(hist, session, provider.source, fetched))
        except ProviderError as e:
            row.update(_nan_dict(["open", "high", "low", "close", "prev_close", "return_1d", "stock_volume",
                                  "avg_stock_volume_30d", "rel_stock_volume_30d"]))
            row.update(price_status="UNAVAILABLE", price_reason=str(e), price_source=provider.source,
                       price_fetched_at=fetched)
            errors.append(f"{sym} Kurs: {e}")
        totals, e_rows, t_rows = _options_evening(provider, sym, ysym, session, settings, iso_utc())
        row.update(totals)
        if totals["options_status"] != "OK":
            errors.append(f"{sym} Optionen: {totals['options_status']} {totals['options_reason']}")
        exp_rows += e_rows
        top_rows += t_rows
        row.update(_calendar_block(provider, ysym, iso_utc()))
        rows.append(row)

    sdir = paths.session(session)
    write_csv(pd.DataFrame(rows).reindex(columns=SUMMARY_COLS), sdir / "evening_summary.csv")
    write_csv(pd.DataFrame(exp_rows).reindex(columns=EXP_COLS), sdir / "evening_expiries.csv.gz")
    write_csv(pd.DataFrame(top_rows).reindex(columns=TOP_COLS), sdir / "evening_top_contracts.csv.gz")
    status_counts = pd.Series([r["options_status"] for r in rows]).value_counts().to_dict()
    meta = {
        "run": "evening", "status": "OK", "session": session, "started_at": started, "finished_at": iso_utc(),
        "late_run": late,
        "source": provider.source, "n_tickers": len(rows), "options_status_counts": status_counts,
        "price_unavailable": int(sum(r["price_status"] != "OK" for r in rows)),
        "request_count": getattr(provider, "request_count", None), "errors": errors,
        "settings": {k: settings[k] for k in ("max_expiry_days", "top_contracts")},
        "note": ("Optionsvolumen zählt nur Kontrakte, deren letzter Handel am Handelstag war. "
                 "Open Interest in dieser Datei ist der Stand zu Beginn des Handelstags."),
    }
    write_json(meta, sdir / "evening_meta.json")
    log.info("Abend-Lauf %s fertig: %s", session, status_counts)
    return meta


# ---------------------------------------------------------------------------------------------- morning

def previous_session(provider, now_utc: datetime, paths: Paths | None = None) -> str:
    """Last trading day before today (New York).

    Yahoo's daily history sometimes lacks the latest bar for hours; during market hours the quote cannot fill the
    gap because it already belongs to today. Sessions the evening run recorded are trading days too, so both
    sources are combined and the later date wins.
    """
    today = et_date(now_utc).isoformat()
    spy = provider.history("SPY", "3mo")
    prior = {d for d in spy["Date"] if d < today}
    if paths is not None:
        prior |= {s for s in list_sessions(paths, "evening_meta.json") if s < today}
    if not prior:
        raise ProviderError("kein vorheriger Handelstag in der SPY-Historie")
    return max(prior)


def run_morning(paths: Paths, provider, settings: dict, now_utc: datetime, session: str | None = None,
                force: bool = False, symbols: list[str] | None = None) -> dict:
    started = iso_utc(now_utc)
    expected = previous_session(provider, now_utc, paths)
    if session and session != expected:
        # Later OI would be recorded as "after the session" and produce a wrong change. Re-analysis: use `analyze`.
        raise ValueError(f"Morgenlauf nur für den letzten Handelstag ({expected}) möglich, nicht für {session}")
    session = expected
    sdir = paths.session(session)
    prev_meta = read_json(sdir / "morning_meta.json")
    if prev_meta and prev_meta.get("oi_status") in ("OK", "UNGEPRÜFT") and not force:
        log.info("Open Interest für %s bereits aktuell erfasst.", session)
        return {"status": "SKIPPED", "session": session, "reason": "OI bereits erfasst"}

    # On a retry keep symbols that were complete and fresh; fetch only failed or stale ones again.
    keep_exp = keep_top = None
    keep: set = set()
    if prev_meta and not force:
        prev_exp = read_csv_if_exists(sdir / "morning_expiries.csv.gz")
        prev_top = read_csv_if_exists(sdir / "morning_top_contracts.csv.gz")
        if prev_exp is not None and len(prev_exp):
            bad = set(prev_meta.get("oi_stale_symbols", [])) | set(prev_meta.get("oi_failed_symbols", []))
            bad |= set(prev_exp.loc[prev_exp["status"] != "OK", "symbol"])
            keep = set(prev_exp["symbol"]) - bad
            keep_exp = prev_exp[prev_exp["symbol"].isin(keep)]
            keep_top = prev_top[prev_top["symbol"].isin(keep)] if prev_top is not None else None

    universe = load_universe(paths)
    if symbols:
        universe = universe[universe["symbol"].isin(symbols)]
    ev_exp = read_csv_if_exists(sdir / "evening_expiries.csv.gz")
    ev_top = read_csv_if_exists(sdir / "evening_top_contracts.csv.gz")
    d0 = parse_date(session)
    max_d = d0 + timedelta(days=int(settings["max_expiry_days"]))
    exp_rows, top_rows, errors = [], [], []
    failed_symbols: set = set()

    for n, (_, u) in enumerate(universe.iterrows(), start=1):
        sym, ysym = u["symbol"], u["yahoo_symbol"]
        if sym in keep:
            continue
        log.info("[%d/%d] %s OI", n, len(universe), sym)
        fetched = iso_utc()
        exps = []
        if ev_exp is not None:
            exps = sorted({str(e) for e in ev_exp.loc[ev_exp["symbol"] == sym, "expiry"] if str(e) > session})
        if not exps:
            try:
                exps = [e for e in provider.expirations(ysym) if d0 < parse_date(e) <= max_d]
            except ProviderError as e:
                errors.append(f"{sym}: {e}")
                failed_symbols.add(sym)
        oi_map, failed = {}, set()
        for e in exps:
            try:
                calls, puts, _ = provider.chain(ysym, e)
            except ProviderError as ex:
                failed.add(e)
                failed_symbols.add(sym)
                errors.append(f"{sym} {e}: {ex}")
                for typ in ("call", "put"):
                    exp_rows.append({"session": session, "symbol": sym, "expiry": e, "type": typ,
                                     "oi_after": math.nan, "n_contracts": math.nan, "n_oi_missing": math.nan,
                                     "status": f"UNAVAILABLE: {ex}", "fetched_at": fetched})
                continue
            c = contracts_frame(calls, puts, e)
            oi_map.update(dict(zip(c["contractSymbol"], c["openInterest"])))
            for typ in ("call", "put"):
                sub = c[c["type"] == typ]
                exp_rows.append({"session": session, "symbol": sym, "expiry": e, "type": typ,
                                 "oi_after": _oi_sum(sub["openInterest"]), "n_contracts": int(len(sub)),
                                 "n_oi_missing": int(sub["openInterest"].isna().sum()), "status": "OK",
                                 "fetched_at": fetched})
        if ev_top is not None:
            for _, t in ev_top[ev_top["symbol"] == sym].iterrows():
                exp = str(t["expiry"])
                if exp <= session:
                    val, note = math.nan, "am Handelstag verfallen"
                elif exp in failed:
                    val, note = math.nan, "Optionskette nicht abrufbar"
                elif t["contract"] in oi_map:
                    val, note = oi_map[t["contract"]], ""
                    if pd.isna(val):
                        note = "OI von der Quelle nicht geliefert"
                else:
                    val, note = math.nan, "Kontrakt nicht mehr gelistet"
                top_rows.append({"session": session, "symbol": sym, "contract": t["contract"], "expiry": exp,
                                 "oi_after": val, "note": note, "fetched_at": fetched})

    mexp = pd.DataFrame(exp_rows).reindex(columns=MORNING_EXP_COLS)
    mtop = pd.DataFrame(top_rows).reindex(columns=MORNING_TOP_COLS)
    if keep_exp is not None and len(keep_exp):
        mexp = pd.concat([keep_exp.reindex(columns=MORNING_EXP_COLS), mexp], ignore_index=True) if len(mexp) else \
            keep_exp.reindex(columns=MORNING_EXP_COLS)
    if keep_top is not None and len(keep_top):
        mtop = pd.concat([keep_top.reindex(columns=MORNING_TOP_COLS), mtop], ignore_index=True) if len(mtop) else \
            keep_top.reindex(columns=MORNING_TOP_COLS)

    # Per symbol: if every expiry still shows exactly the evening OI although contracts traded, the source
    # has not updated that symbol yet. Its OI change is then UNAVAILABLE, never zero.
    oi_status, unchanged_share, n_compared, stale_symbols = "UNGEPRÜFT", math.nan, 0, []
    if ev_exp is not None:
        m = ev_exp.merge(mexp, on=["symbol", "expiry", "type"], how="inner", suffixes=("_ev", "_mo"))
        m = m[m["oi_start"].notna() & m["oi_after"].notna()].copy()
        m["same"] = m["oi_start"] == m["oi_after"]
        vol = ev_exp.groupby("symbol")["volume"].sum()
        per_sym = m.groupby("symbol")["same"].all()
        per_sym = per_sym[[vol.get(s, 0) > 0 for s in per_sym.index]]
        n_compared = int(len(per_sym))
        stale_symbols = sorted(per_sym[per_sym].index.tolist())
        if n_compared == 0:
            oi_status = "FEHLGESCHLAGEN"
        else:
            unchanged_share = float(per_sym.mean())
            if unchanged_share >= float(settings["oi_unchanged_share_stale"]):
                oi_status = "STALE"
            elif stale_symbols or failed_symbols:
                oi_status = "TEILWEISE"
            else:
                oi_status = "OK"
    elif not len(mexp) or mexp["status"].ne("OK").all():
        oi_status = "FEHLGESCHLAGEN"

    write_csv(mexp, sdir / "morning_expiries.csv.gz")
    write_csv(mtop, sdir / "morning_top_contracts.csv.gz")
    meta = {
        "run": "morning", "status": "OK", "session": session, "started_at": started, "finished_at": iso_utc(),
        "attempt": int((prev_meta or {}).get("attempt", 0)) + 1, "source": provider.source,
        "oi_status": oi_status, "oi_unchanged_share": unchanged_share, "n_tickers_compared": n_compared,
        "oi_stale_symbols": stale_symbols, "oi_failed_symbols": sorted(failed_symbols),
        "symbols_kept_from_previous_attempt": len(keep),
        "request_count": getattr(provider, "request_count", None), "errors": errors,
        "note": ("oi_after ist das Open Interest nach Abschluss des Handelstags (Veröffentlichung über Nacht). "
                 "STALE: die Quelle zeigt überwiegend noch den alten Stand. TEILWEISE: einzelne Ticker alt oder "
                 "nicht abrufbar, deren Veränderung ist UNAVAILABLE. Weitere Versuche holen nur diese Ticker neu."),
    }
    write_json(meta, sdir / "morning_meta.json")
    log.info("Morgen-Lauf %s fertig: OI-Status %s", session, oi_status)
    return meta
