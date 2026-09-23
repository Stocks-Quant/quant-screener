"""Agent 2: the analysis agent. It describes, it never recommends and never calls activity bullish or bearish.

Input: the files written by the data agent. No network access, so every number here is reproducible.
Output per session: analysis.csv (all tickers), candidates.json (input for the flagging agent), report.md.
"""
from __future__ import annotations

import logging
import math
from datetime import timedelta

import numpy as np
import pandas as pd

from .common import (UNAVAILABLE, Paths, fnum, pct, iso_utc, is_missing, list_sessions, parse_date, read_csv,
                     read_csv_if_exists, read_json, read_prices, write_csv, write_json)
from .events import structural_events

log = logging.getLogger("screener")

CANNOT_DETERMINE = [
    "ob das Volumen Käufe oder Verkäufe waren (jeder Kontrakt hat einen Käufer und einen Verkäufer)",
    "ob der Handel für die Gegenseite eröffnend oder schließend war",
    "ob es sich um eine Richtungswette oder um eine Absicherung handelt",
]

OI_OPENED = "überwiegend neu eröffnet"
OI_CLOSED = "überwiegend geschlossen"
OI_MIXED = "gemischt, nicht eindeutig"

ANALYSIS_COLS = [
    "session", "symbol", "name", "sector", "sector_etf", "data_status", "data_issues",
    "close", "return_1d", "price_z", "sector_return_1d", "excess_return_1d", "stock_rel_volume_30d",
    "option_volume", "call_volume", "put_volume", "baseline_n", "baseline_ok", "baseline_median",
    "option_volume_ratio", "option_volume_z", "option_volume_pctile",
    "cp_ratio", "cp_ratio_30d", "cp_ratio_vs_30d",
    "delta_oi_calls", "delta_oi_puts", "delta_oi_total", "net_open_share_all", "oi_volume_coverage",
    "top_volume", "top_delta_oi", "top_open_share", "oi_classification", "oi_base_check", "oi_consistency",
    "short_dated_share", "itm_call_share", "expiring_today_share",
    "days_to_earnings", "next_earnings", "days_to_ex_dividend", "ex_dividend_date", "structural_event",
    "sector_elevated_share", "market_elevated_share",
    "is_elevated", "is_candidate", "exclusion_reason", "boring_hints",
]


# ------------------------------------------------------------------------------------------------ helpers

def robust_z(x: float, base: np.ndarray) -> float:
    """z-score of log(1+x) against the ticker's own history, using median and MAD (robust to past spikes)."""
    if is_missing(x) or len(base) < 5:
        return math.nan
    lb = np.log1p(base.astype(float))
    med = float(np.median(lb))
    mad = float(np.median(np.abs(lb - med))) * 1.4826
    if mad < 1e-9:
        mad = float(np.std(lb, ddof=1)) if len(lb) > 1 else 0.0
    if mad < 1e-9:
        return math.nan
    return (math.log1p(float(x)) - med) / mad


class PriceCache:
    def __init__(self, paths: Paths):
        self.paths = paths
        self._c: dict = {}

    def get(self, symbol: str) -> pd.DataFrame | None:
        if symbol not in self._c:
            self._c[symbol] = read_prices(self.paths, symbol)
        return self._c[symbol]

    def day_metrics(self, symbol: str, session: str, lookback: int = 20) -> dict:
        out = {"ret": math.nan, "z": math.nan}
        df = self.get(symbol)
        if df is None:
            return out
        dates = df["Date"].tolist()
        if session not in dates:
            return out
        i = dates.index(session)
        ret = df["Adj Close"].pct_change()
        out["ret"] = float(ret.iloc[i]) if i >= 1 else math.nan
        prior = ret.iloc[max(1, i - lookback):i].dropna()
        if len(prior) >= 15:
            sd = float(prior.std(ddof=1))
            if sd > 0 and not math.isnan(out["ret"]):
                out["z"] = out["ret"] / sd
        return out


def _load_baseline(paths: Paths, session: str, window: int) -> pd.DataFrame:
    prior = [s for s in list_sessions(paths, "evening_summary.csv") if s < session][-(window * 2):]
    frames = []
    for s in prior:
        df = read_csv(paths.session(s) / "evening_summary.csv")
        frames.append(df[["session", "symbol", "options_status", "call_volume", "put_volume", "total_option_volume"]])
    if not frames:
        return pd.DataFrame(columns=["session", "symbol", "call_volume", "put_volume", "total_option_volume"])
    b = pd.concat(frames, ignore_index=True)
    b = b[(b["options_status"] == "OK") & b["total_option_volume"].notna()]
    b = b.sort_values("session").groupby("symbol", group_keys=False).tail(window)
    return b


def _events(row: pd.Series, session: str, lookahead: int) -> dict:
    d0 = parse_date(session)
    res = {"days_to_earnings": math.nan, "next_earnings": math.nan, "days_to_ex_dividend": math.nan,
           "ex_dividend_date": math.nan}
    if not is_missing(row.get("earnings_dates")):
        cands = []
        for d in str(row["earnings_dates"]).split(";"):
            try:
                days = (parse_date(d) - d0).days
            except ValueError:
                continue
            if -2 <= days <= lookahead:
                cands.append((days, d))
        if cands:
            days, d = sorted(cands)[0]
            res.update(days_to_earnings=days, next_earnings=d)
    if not is_missing(row.get("ex_dividend_date")):
        try:
            days = (parse_date(row["ex_dividend_date"]) - d0).days
            if 0 <= days <= lookahead:
                res.update(days_to_ex_dividend=days, ex_dividend_date=str(row["ex_dividend_date"]))
        except ValueError:
            pass
    return res


# ------------------------------------------------------------------------------------------------ main

def analyze_session(paths: Paths, settings: dict, session: str) -> dict:
    sdir = paths.session(session)
    ev = read_csv_if_exists(sdir / "evening_summary.csv")
    if ev is None:
        msg = f"Keine Abenddaten für {session}: Optionsvolumen UNAVAILABLE, keine Analyse möglich."
        log.warning(msg)
        (sdir / "report.md").parent.mkdir(parents=True, exist_ok=True)
        (sdir / "report.md").write_text(f"# Analyse {session}\n\n{msg}\n")
        write_json({"session": session, "generated_at": iso_utc(), "status": "NO_DATA", "message": msg,
                    "candidates": []}, sdir / "candidates.json")
        return {"session": session, "status": "NO_DATA"}

    ev_exp = read_csv_if_exists(sdir / "evening_expiries.csv.gz")
    ev_top = read_csv_if_exists(sdir / "evening_top_contracts.csv.gz")
    mo_exp = read_csv_if_exists(sdir / "morning_expiries.csv.gz")
    mo_top = read_csv_if_exists(sdir / "morning_top_contracts.csv.gz")
    mo_meta = read_json(sdir / "morning_meta.json") or {}
    oi_status = mo_meta.get("oi_status", "FEHLT")
    oi_usable = mo_exp is not None and oi_status in ("OK", "TEILWEISE", "UNGEPRÜFT")
    oi_bad_symbols = set(mo_meta.get("oi_stale_symbols", [])) | set(mo_meta.get("oi_failed_symbols", []))

    prev_mo = [s for s in list_sessions(paths, "morning_expiries.csv.gz") if s < session]
    prev_mo_exp = read_csv(paths.session(prev_mo[-1]) / "morning_expiries.csv.gz") if prev_mo else None

    window, min_hist = int(settings["baseline_window"]), int(settings["min_history"])
    baseline = _load_baseline(paths, session, window)
    prices = PriceCache(paths)
    s_events = structural_events(session, int(settings["event_lookahead_days"]))
    short_cut = (parse_date(session) + timedelta(days=int(settings["short_dated_days"]))).isoformat()
    thr_open = float(settings["opening_share_threshold"])

    recs, tops = [], {}
    for _, s in ev.iterrows():
        sym = s["symbol"]
        issues = []
        rec = {k: s.get(k) for k in ("session", "symbol", "name", "sector", "sector_etf", "close")}
        rec["stock_rel_volume_30d"] = s.get("rel_stock_volume_30d")

        pm = prices.day_metrics(sym, session)
        em = prices.day_metrics(s["sector_etf"], session)
        rec.update(return_1d=pm["ret"], price_z=pm["z"], sector_return_1d=em["ret"],
                   excess_return_1d=pm["ret"] - em["ret"])
        if s["price_status"] != "OK":
            issues.append(f"Kurs: {s['price_reason']}")
        elif math.isnan(pm["z"]):
            issues.append("Kurs-z nicht berechenbar (zu wenig Historie)")

        # options volume against the ticker's own history
        opt_ok = s["options_status"] == "OK"
        if not opt_ok:
            issues.append(f"Optionsvolumen {s['options_status']}: {s['options_reason']}")
        vol = float(s["total_option_volume"]) if opt_ok else math.nan
        b = baseline[baseline["symbol"] == sym]
        n = int(len(b))
        base_vals = b["total_option_volume"].to_numpy(dtype=float)
        rec["expiring_today_share"] = (float(s["volume_expiring_today"]) / vol
                                       if opt_ok and vol > 0 and not is_missing(s["volume_expiring_today"]) else math.nan)
        rec.update(option_volume=vol, call_volume=s["call_volume"] if opt_ok else math.nan,
                   put_volume=s["put_volume"] if opt_ok else math.nan, baseline_n=n, baseline_ok=n >= min_hist)
        if n < min_hist:
            issues.append(f"Baseline erst {n} von {min_hist} Handelstagen")
        med = float(np.median(base_vals)) if n else math.nan
        rec["baseline_median"] = med
        rec["option_volume_ratio"] = vol / med if opt_ok and n and med > 0 else math.nan
        rec["option_volume_z"] = robust_z(vol, base_vals) if opt_ok else math.nan
        rec["option_volume_pctile"] = float((base_vals < vol).mean() * 100) if opt_ok and n else math.nan
        put_v = rec["put_volume"]
        rec["cp_ratio"] = rec["call_volume"] / put_v if opt_ok and put_v and put_v > 0 else math.nan
        bp = b["put_volume"].sum()
        rec["cp_ratio_30d"] = b["call_volume"].sum() / bp if n and bp > 0 else math.nan
        rec["cp_ratio_vs_30d"] = (rec["cp_ratio"] / rec["cp_ratio_30d"]
                                  if not is_missing(rec["cp_ratio"]) and not is_missing(rec["cp_ratio_30d"])
                                  and rec["cp_ratio_30d"] > 0 else math.nan)

        # open interest change, aggregate over expiries alive after the session
        rec.update(delta_oi_calls=math.nan, delta_oi_puts=math.nan, delta_oi_total=math.nan,
                   net_open_share_all=math.nan, oi_volume_coverage=math.nan, oi_base_check="nicht prüfbar")
        e_sym = ev_exp[(ev_exp["symbol"] == sym) & (ev_exp["expiry"].astype(str) > session)] if ev_exp is not None else None
        sym_oi_ok = oi_usable and sym not in oi_bad_symbols
        if not oi_usable:
            issues.append(f"OI-Veränderung UNAVAILABLE (Morgen-Lauf: {oi_status})")
        elif not sym_oi_ok:
            issues.append("OI-Veränderung UNAVAILABLE (Quelle für diesen Ticker nicht aktualisiert oder nicht abrufbar)")
        elif e_sym is not None and len(e_sym):
            m = e_sym.merge(mo_exp[mo_exp["symbol"] == sym], on=["expiry", "type"], how="inner")
            m = m[m["oi_start"].notna() & m["oi_after"].notna()]
            if len(m):
                m = m.assign(d=m["oi_after"] - m["oi_start"])
                rec["delta_oi_calls"] = float(m.loc[m["type"] == "call", "d"].sum())
                rec["delta_oi_puts"] = float(m.loc[m["type"] == "put", "d"].sum())
                rec["delta_oi_total"] = rec["delta_oi_calls"] + rec["delta_oi_puts"]
                v_cmp, v_all = float(m["volume"].sum()), float(e_sym["volume"].sum())
                rec["net_open_share_all"] = rec["delta_oi_total"] / v_cmp if v_cmp > 0 else math.nan
                rec["oi_volume_coverage"] = v_cmp / v_all if v_all > 0 else math.nan
            if prev_mo_exp is not None:
                p = prev_mo_exp[prev_mo_exp["symbol"] == sym].merge(e_sym, on=["expiry", "type"], how="inner")
                p = p[p["oi_after"].notna() & p["oi_start"].notna()]
                if len(p):
                    a, bb = float(p["oi_after"].sum()), float(p["oi_start"].sum())
                    rel = abs(bb - a) / max(a, 1.0)
                    rec["oi_base_check"] = ("OK" if rel <= float(settings["oi_base_tolerance"])
                                            else f"ABWEICHUNG {pct(rel, 1)}")

        # the most active strikes: did open interest rise (new positions) or fall (closing)?
        t = ev_top[ev_top["symbol"] == sym].copy() if ev_top is not None else pd.DataFrame()
        rec.update(top_volume=math.nan, top_delta_oi=math.nan, top_open_share=math.nan,
                   oi_classification=UNAVAILABLE, oi_consistency="", short_dated_share=math.nan,
                   itm_call_share=math.nan)
        if len(t):
            if sym_oi_ok and mo_top is not None:
                t = t.merge(mo_top[mo_top["symbol"] == sym][["contract", "oi_after", "note"]], on="contract", how="left")
            else:
                t["oi_after"], t["note"] = math.nan, "OI nach Handelsschluss UNAVAILABLE"
            t["delta_oi"] = t["oi_after"] - t["oi_start"]
            valid = t["delta_oi"].notna() & (t["expiry"].astype(str) > session)
            tv = float(t.loc[valid, "volume"].sum())
            coverage = tv / float(t["volume"].sum()) if float(t["volume"].sum()) > 0 else 0.0
            if valid.any() and tv > 0 and coverage < float(settings["oi_min_coverage"]):
                issues.append(f"OI-Veränderung nur für {pct(coverage)} des Top-Volumens verfügbar")
            elif valid.any() and tv > 0:
                rec["top_volume"] = tv
                rec["top_delta_oi"] = float(t.loc[valid, "delta_oi"].sum())
                share = rec["top_delta_oi"] / tv
                rec["top_open_share"] = share
                rec["oi_classification"] = OI_OPENED if share >= thr_open else OI_CLOSED if share <= -thr_open else OI_MIXED
                bad = int((t.loc[valid, "delta_oi"].abs() > t.loc[valid, "volume"] * 1.05 + 10).sum())
                if bad:
                    rec["oi_consistency"] = f"bei {bad} Kontrakt(en) ist die OI-Veränderung größer als das Volumen: Datenbasis prüfen"
            total_top = float(t["volume"].sum())
            if total_top > 0:
                rec["short_dated_share"] = float(t.loc[t["expiry"].astype(str) <= short_cut, "volume"].sum()) / total_top
                itm = t["in_the_money"].astype(str).str.lower().isin(["true", "1"])
                rec["itm_call_share"] = float(t.loc[(t["type"] == "call") & itm, "volume"].sum()) / total_top
            tops[sym] = t

        rec.update(_events(s, session, int(settings["event_lookahead_days"])))
        near = [e for e in s_events if e["days"] <= 3]
        rec["structural_event"] = "; ".join(f"{e['date']} {e['label']}" for e in near) if near else math.nan
        rec["data_issues"] = " | ".join(issues)
        complete = opt_ok and rec["baseline_ok"] and rec["oi_classification"] != UNAVAILABLE and not math.isnan(pm["z"])
        rec["data_status"] = "OK" if complete else "UNVOLLSTÄNDIG"
        recs.append(rec)

    a = pd.DataFrame(recs)
    zt, rt = float(settings["z_threshold"]), float(settings["ratio_threshold"])
    a["is_elevated"] = (a["baseline_ok"].astype(bool) & (a["option_volume_z"] >= zt) & (a["option_volume_ratio"] >= rt)).fillna(False)
    comparable = a["baseline_ok"].astype(bool) & a["option_volume_z"].notna()
    n_comp = int(comparable.sum())
    a["market_elevated_share"] = float(a.loc[comparable, "is_elevated"].mean()) if n_comp else math.nan
    sec_share = []
    for _, r in a.iterrows():
        peers = a[(a["sector"] == r["sector"]) & (a["symbol"] != r["symbol"]) & comparable]
        sec_share.append(float(peers["is_elevated"].mean()) if len(peers) else math.nan)
    a["sector_elevated_share"] = sec_share

    hints_col, cand_col, excl_col = [], [], []
    quiet = float(settings["price_quiet_z"])
    for _, r in a.iterrows():
        hints = []
        de = r["days_to_earnings"]
        if not is_missing(de):
            if de >= 0 and de <= 14:
                hints.append(f"Quartalszahlen am {r['next_earnings']} (in {int(de)} Tagen)")
            elif de < 0:
                hints.append(f"Quartalszahlen vor {int(-de)} Tag(en) ({r['next_earnings']})")
        dx = r["days_to_ex_dividend"]
        if not is_missing(dx) and dx <= 5:
            if dx <= 3 and not is_missing(r["itm_call_share"]) and r["itm_call_share"] >= 0.5:
                hints.append(f"Ex-Dividende am {r['ex_dividend_date']} und {pct(r['itm_call_share'])} des Top-Volumens in "
                             "Calls im Geld: typisches Muster von Dividendenarbitrage")
            else:
                hints.append(f"Ex-Dividende am {r['ex_dividend_date']}")
        if not is_missing(r["structural_event"]):
            hints.append(str(r["structural_event"]))
        if not is_missing(r["sector_elevated_share"]) and r["sector_elevated_share"] >= float(settings["sector_wide_share"]):
            hints.append(f"branchenweit: {pct(r['sector_elevated_share'])} der Werte im Sektor {r['sector']} ebenfalls erhöht")
        if not is_missing(r["market_elevated_share"]) and r["market_elevated_share"] >= float(settings["market_wide_share"]):
            hints.append(f"marktweit: {pct(r['market_elevated_share'])} des Universums mit erhöhtem Optionsvolumen")
        if not is_missing(r["expiring_today_share"]) and r["expiring_today_share"] >= 0.4:
            hints.append(f"{pct(r['expiring_today_share'])} des Optionsvolumens in Kontrakten, die am Handelstag verfielen")
        if not is_missing(r["short_dated_share"]) and r["short_dated_share"] >= 0.6:
            hints.append(f"{pct(r['short_dated_share'])} des Top-Volumens verfällt innerhalb von {settings['short_dated_days']} Tagen")
        hints_col.append(" | ".join(hints))

        cand, excl = False, ""
        if r["is_elevated"]:
            if r["oi_classification"] == UNAVAILABLE:
                excl = "OI-Veränderung nicht verfügbar: ohne sie ist die Aktivität kein Beleg für irgendetwas"
            elif r["oi_classification"] != OI_OPENED:
                excl = f"OI-Veränderung: {r['oi_classification']}"
            elif is_missing(r["price_z"]):
                excl = "Kursbewegung nicht berechenbar"
            elif abs(r["price_z"]) >= quiet:
                excl = f"Kurs hat sich bereits deutlich bewegt (z = {fnum(r['price_z'], 1, signed=True)})"
            else:
                cand = True
        cand_col.append(cand)
        excl_col.append(excl)
    a["boring_hints"], a["is_candidate"], a["exclusion_reason"] = hints_col, cand_col, excl_col

    a = a.reindex(columns=ANALYSIS_COLS)
    write_csv(a, sdir / "analysis.csv")

    cands = a[a["is_candidate"]].sort_values("option_volume_z", ascending=False).head(int(settings["max_candidates"]))
    ev_idx = ev.set_index("symbol")
    payload = {
        "session": session,
        "generated_at": iso_utc(),
        "status": "OK",
        "source": (read_json(sdir / "evening_meta.json") or {}).get("source", UNAVAILABLE),
        "oi_status": oi_status,
        "n_tickers": int(len(a)),
        "n_complete": int((a["data_status"] == "OK").sum()),
        "n_baseline_ok": int(a["baseline_ok"].sum()),
        "min_history": min_hist,
        "market_elevated_share": a["market_elevated_share"].iloc[0] if len(a) else math.nan,
        "rules": {
            "unusual": f"robustes z des Optionsvolumens >= {zt} und mindestens {rt}x Median der eigenen letzten {window} Tage",
            "opened": f"OI an den aktivsten Kontrakten gestiegen um >= {pct(thr_open)} ihres Volumens",
            "price_quiet": f"|Kurs-z| < {quiet} (Tagesrendite relativ zur eigenen 20-Tage-Volatilität)",
            "ranking": "nach Ungewöhnlichkeit relativ zur eigenen Historie, nicht nach absoluter Größe",
        },
        "cannot_determine": CANNOT_DETERMINE,
        "candidates": [],
        "elevated_not_candidates": [],
        "incomplete": [],
    }
    for rank, (_, r) in enumerate(cands.iterrows(), start=1):
        sym = r["symbol"]
        src = ev_idx.loc[sym]
        t = tops.get(sym, pd.DataFrame())
        top_list = [] if not len(t) else t[["contract", "type", "expiry", "strike", "volume", "oi_start", "oi_after",
                                            "delta_oi", "in_the_money", "last_price", "implied_vol", "note"]].to_dict("records")
        payload["candidates"].append({
            "rank": rank, "symbol": sym, "name": r["name"], "sector": r["sector"],
            "option_volume": r["option_volume"], "baseline_median": r["baseline_median"],
            "option_volume_ratio": r["option_volume_ratio"], "option_volume_z": r["option_volume_z"],
            "option_volume_pctile": r["option_volume_pctile"], "baseline_n": r["baseline_n"],
            "call_volume": r["call_volume"], "put_volume": r["put_volume"],
            "cp_ratio": r["cp_ratio"], "cp_ratio_30d": r["cp_ratio_30d"],
            "delta_oi_calls": r["delta_oi_calls"], "delta_oi_puts": r["delta_oi_puts"],
            "top_open_share": r["top_open_share"], "oi_classification": r["oi_classification"],
            "oi_base_check": r["oi_base_check"], "oi_consistency": r["oi_consistency"],
            "price": {"close": r["close"], "return_1d": r["return_1d"], "price_z": r["price_z"],
                      "sector_etf": r["sector_etf"], "sector_return_1d": r["sector_return_1d"],
                      "stock_rel_volume_30d": r["stock_rel_volume_30d"]},
            "events": {"next_earnings": r["next_earnings"], "days_to_earnings": r["days_to_earnings"],
                       "ex_dividend_date": r["ex_dividend_date"], "structural_event": r["structural_event"]},
            "boring_hints_from_code": [h for h in str(r["boring_hints"]).split(" | ") if h],
            "top_contracts": top_list,
            "timestamps": {"price_fetched_at": src.get("price_fetched_at"),
                           "options_fetched_at": src.get("options_fetched_at"),
                           "options_last_trade_et": src.get("options_last_trade_et"),
                           "oi_after_fetched_at": mo_meta.get("finished_at", UNAVAILABLE)},
        })
    for _, r in a[a["is_elevated"] & ~a["is_candidate"]].sort_values("option_volume_z", ascending=False).iterrows():
        payload["elevated_not_candidates"].append({"symbol": r["symbol"], "option_volume_z": r["option_volume_z"],
                                                   "option_volume_ratio": r["option_volume_ratio"],
                                                   "reason": r["exclusion_reason"]})
    for _, r in a[a["data_status"] != "OK"].iterrows():
        payload["incomplete"].append({"symbol": r["symbol"], "issues": r["data_issues"]})
    write_json(payload, sdir / "candidates.json")
    (sdir / "report.md").write_text(render_report(a, payload, settings))
    _update_flags_log(paths, session, a, cands)
    write_json({"latest_session": session, "generated_at": payload["generated_at"], "oi_status": oi_status,
                "n_candidates": len(payload["candidates"]), "n_complete": payload["n_complete"],
                "files": {k: f"data/sessions/{session}/{k}" for k in ("candidates.json", "report.md", "analysis.csv")}},
               paths.data / "latest.json")
    log.info("Analyse %s: %d Kandidaten, %d vollständig", session, len(payload["candidates"]), payload["n_complete"])
    return {"session": session, "status": "OK", "n_candidates": len(payload["candidates"])}


def _update_flags_log(paths: Paths, session: str, a: pd.DataFrame, cands: pd.DataFrame) -> None:
    f = paths.data / "flags_log.csv"
    cols = ["session", "symbol", "sector", "sector_etf", "rank", "option_volume_z", "option_volume_ratio",
            "top_open_share", "price_z", "next_earnings", "boring_hints", "logged_at"]
    new = cands.reset_index(drop=True).copy()
    new["rank"] = range(1, len(new) + 1)
    new["logged_at"] = iso_utc()
    new = new.reindex(columns=cols)
    old = read_csv_if_exists(f)
    if old is not None:
        old = old[old["session"].astype(str) != session]
        if len(new) and len(old):
            new = pd.concat([old, new], ignore_index=True)
        elif len(old):
            new = old
    write_csv(new.sort_values(["session", "rank"]).reset_index(drop=True), f)


# ------------------------------------------------------------------------------------------------ report

def render_report(a: pd.DataFrame, p: dict, settings: dict) -> str:
    L = [f"# Optionsaktivität: Analyse für den Handelstag {p['session']}", ""]
    L.append(f"Erstellt {p['generated_at']} aus Dateien des Datenagenten. Quelle: {p['source']}.")
    L.append(f"Open Interest nach Handelsschluss: **{p['oi_status']}**. "
             f"Vollständige Datensätze: {p['n_complete']} von {p['n_tickers']}. "
             f"Mit ausreichender Baseline ({p['min_history']} Tage): {p['n_baseline_ok']}.")
    mes = p["market_elevated_share"]
    if not is_missing(mes):
        L.append(f"Anteil des Universums mit erhöhtem Optionsvolumen: {pct(mes)}.")
    L += ["", "Dieser Bericht beschreibt Aktivität. Er bewertet sie nicht als bullisch oder bärisch und empfiehlt keinen Trade.", ""]

    L += ["## Kandidaten: ungewöhnliches Volumen, neue Positionen, Kurs unauffällig", ""]
    if not p["candidates"]:
        L += ["Heute nichts, das alle drei Bedingungen erfüllt.", ""]
    for c in p["candidates"]:
        L.append(f"### {c['rank']}. {c['symbol']} ({c['name']}, {c['sector']})")
        L.append(f"* Optionsvolumen {fnum(c['option_volume'], 0)} Kontrakte, Median der letzten {c['baseline_n']} Tage "
                 f"{fnum(c['baseline_median'], 0)} ({fnum(c['option_volume_ratio'], 1)}x, z = {fnum(c['option_volume_z'], 1)}, "
                 f"höher als {fnum(c['option_volume_pctile'], 0)} % der Vergleichstage)")
        L.append(f"* Call/Put-Verhältnis {fnum(c['cp_ratio'])} gegenüber {fnum(c['cp_ratio_30d'])} im Schnitt der Baseline")
        L.append(f"* Open Interest an den aktivsten Kontrakten: {c['oi_classification']} "
                 f"(Veränderung {fnum(c['top_open_share'], 0, pct=True)} des Volumens); gesamt Calls "
                 f"{fnum(c['delta_oi_calls'], 0, signed=True)}, Puts {fnum(c['delta_oi_puts'], 0, signed=True)}. "
                 f"OI-Basisprüfung: {c['oi_base_check']}")
        pr = c["price"]
        L.append(f"* Kurs {fnum(pr['close'])} ({fnum(pr['return_1d'], 1, pct=True, signed=True)}, z = {fnum(pr['price_z'], 1, signed=True)}); "
                 f"Sektor-ETF {pr['sector_etf']} {fnum(pr['sector_return_1d'], 1, pct=True, signed=True)}; "
                 f"Aktienvolumen {fnum(pr['stock_rel_volume_30d'], 1)}x des 30-Tage-Schnitts")
        hints = c["boring_hints_from_code"]
        L.append(f"* Naheliegende Erklärungen aus den Daten: {'; '.join(hints) if hints else 'keine im Code erkennbar, bitte Nachrichten prüfen'}")
        L.append(f"* Nicht bestimmbar aus diesen Daten: {'; '.join(CANNOT_DETERMINE)}")
        if c["top_contracts"]:
            L += ["", "| Kontrakt | Typ | Verfall | Strike | Volumen | OI vorher | OI nachher | Veränderung |",
                  "|---|---|---|---:|---:|---:|---:|---:|"]
            for t in c["top_contracts"][:5]:
                L.append(f"| {t['contract']} | {t['type']} | {t['expiry']} | {fnum(t['strike'])} | {fnum(t['volume'], 0)} | "
                         f"{fnum(t['oi_start'], 0)} | {fnum(t['oi_after'], 0)} | {fnum(t['delta_oi'], 0, signed=True)} |")
        L.append("")

    L += ["## Erhöhtes Volumen, aber kein Kandidat", ""]
    if p["elevated_not_candidates"]:
        L += ["| Ticker | z | Faktor | Grund |", "|---|---:|---:|---|"]
        for e in p["elevated_not_candidates"]:
            L.append(f"| {e['symbol']} | {fnum(e['option_volume_z'], 1)} | {fnum(e['option_volume_ratio'], 1)}x | {e['reason']} |")
    else:
        L.append("Keine.")
    L.append("")

    L += ["## Höchstes Optionsvolumen relativ zur eigenen Historie (Top 15)", "",
          "| Ticker | z | Faktor | C/P heute | C/P Schnitt | OI-Veränderung Top-Kontrakte | Kurs-z | Hinweise |",
          "|---|---:|---:|---:|---:|---|---:|---|"]
    top = a[a["option_volume_z"].notna()].sort_values("option_volume_z", ascending=False).head(15)
    for _, r in top.iterrows():
        L.append(f"| {r['symbol']} | {fnum(r['option_volume_z'], 1)} | {fnum(r['option_volume_ratio'], 1)}x | "
                 f"{fnum(r['cp_ratio'])} | {fnum(r['cp_ratio_30d'])} | {r['oi_classification']} | "
                 f"{fnum(r['price_z'], 1, signed=True)} | {r['boring_hints'] if r['boring_hints'] else ''} |")
    if not len(top):
        L.append("| (noch keine Baseline) | | | | | | | |")
    L.append("")

    L += ["## Datenlücken", ""]
    if p["incomplete"]:
        L.append(f"{len(p['incomplete'])} Ticker unvollständig. Häufigste Gründe:")
        reasons = pd.Series([i.split(":")[0] for x in p["incomplete"] for i in str(x["issues"]).split(" | ") if i])
        for reason, n in reasons.value_counts().head(8).items():
            L.append(f"* {reason}: {n}")
    else:
        L.append("Keine.")
    L += ["", "---", "Screener, kein Signal. Ungewöhnliche Aktivität ist ein Grund, sich ein Unternehmen anzusehen, "
          "kein Grund, eine Position einzugehen. Daten verzögert und inoffiziell: vor jeder Entscheidung an der Quelle prüfen.", ""]
    return "\n".join(L)
