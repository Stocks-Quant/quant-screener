import json
from datetime import date, datetime, time, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from screener.analyze import OI_CLOSED, OI_OPENED, analyze_session, robust_z
from screener.common import ET, UNAVAILABLE, Paths, load_settings, read_csv, read_json
from screener.evaluate_flags import evaluate, wilson
from screener.events import structural_events, third_friday
from screener.fetch_data import run_evening, run_morning
from tests.fake_market import BENCH, UNIVERSE, FakeMarket, FakeProvider

REPO = Path(__file__).resolve().parent.parent


def make_paths(tmp_path: Path, **overrides) -> Paths:
    cfg = tmp_path / "config"
    cfg.mkdir()
    pd.DataFrame([{"symbol": s, "yahoo_symbol": s, "name": n, "sector": sec, "sector_etf": e}
                  for s, n, sec, e in UNIVERSE]).to_csv(cfg / "universe.csv", index=False)
    pd.DataFrame([{"symbol": b, "yahoo_symbol": b, "name": b, "role": "Benchmark"} for b in BENCH]).to_csv(
        cfg / "benchmarks.csv", index=False)
    settings = json.loads((REPO / "config" / "settings.json").read_text())
    settings.update(min_history=10, request_pause_seconds=0, **overrides)
    (cfg / "settings.json").write_text(json.dumps(settings))
    return Paths(tmp_path)


def at(d: date, hh: int, mm: int) -> datetime:
    return datetime.combine(d, time(hh, mm), tzinfo=ET).astimezone(timezone.utc)


def simulate(paths: Paths, market: FakeMarket, sessions: list[date]) -> None:
    settings = load_settings(paths)
    for d in sessions:
        run_evening(paths, FakeProvider(market, at(d, 17, 30)), settings, at(d, 17, 30))
        nxt = market.days[market.day_idx[d] + 1]
        meta = run_morning(paths, FakeProvider(market, at(nxt, 7, 30)), settings, at(nxt, 7, 30))
        analyze_session(paths, settings, meta["session"])
    evaluate(paths, settings)


@pytest.fixture(scope="module")
def sim(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sim")
    base = FakeMarket()
    d_anom = base.days[320]
    market = FakeMarket(anomalies={
        ("AAA", d_anom): {"extra": 6000, "net_share": 0.8, "price_jump": 0.0, "drift_after": 0.18},
        ("BBB", d_anom): {"extra": 6000, "net_share": -0.8, "price_jump": 0.0},
        ("CCC", d_anom): {"extra": 6000, "net_share": 0.8, "price_jump": 0.06},
    })
    paths = make_paths(tmp)
    simulate(paths, market, market.days[300:342])
    return paths, market, d_anom.isoformat()


def test_volume_only_counts_contracts_traded_in_session(sim):
    paths, market, _ = sim
    d = market.days[305]
    s = read_csv(paths.session(d.isoformat()) / "evening_summary.csv").set_index("symbol")
    expected = 0
    max_exp = date.fromordinal(d.toordinal() + load_settings(paths)["max_expiry_days"]).isoformat()
    for e in [x for x in market.expirations_on(d) if x <= max_exp]:
        for typ in ("call", "put"):
            for k in (90, 95, 100, 105, 110):
                expected += market.session_volume("AAA", e, typ, k, d)[0]
    assert s.loc["AAA", "options_status"] == "OK"
    assert s.loc["AAA", "total_option_volume"] == expected
    assert s.loc["AAA", "n_contracts_stale_volume"] > 0


def test_oi_delta_matches_market_model(sim):
    paths, market, anom = sim
    a = read_csv(paths.session(anom) / "analysis.csv").set_index("symbol")
    assert a.loc["AAA", "oi_classification"] == OI_OPENED
    assert a.loc["BBB", "oi_classification"] == OI_CLOSED
    assert a.loc["AAA", "oi_base_check"] == "OK"
    assert a.loc["AAA", "oi_consistency"] != a.loc["AAA", "oi_consistency"] or a.loc["AAA", "oi_consistency"] == ""


def test_candidates_and_exclusions(sim):
    paths, _, anom = sim
    c = read_json(paths.session(anom) / "candidates.json")
    syms = [x["symbol"] for x in c["candidates"]]
    assert syms == ["AAA"], syms
    excl = {x["symbol"]: x["reason"] for x in c["elevated_not_candidates"]}
    assert "geschlossen" in excl["BBB"]
    assert "Kurs" in excl["CCC"]
    cand = c["candidates"][0]
    assert cand["option_volume_z"] >= 2 and cand["option_volume_ratio"] >= 2
    assert cand["top_contracts"][0]["delta_oi"] > 0
    assert len(c["cannot_determine"]) == 3
    report = (paths.session(anom) / "report.md").read_text()
    assert "AAA" in report and "bullisch" not in report.replace("nicht als bullisch", "")


def test_quiet_days_have_no_candidates(sim):
    paths, market, anom = sim
    n = 0
    for d in market.days[312:340]:
        if d.isoformat() == anom:
            continue
        c = read_json(paths.session(d.isoformat()) / "candidates.json")
        n += len(c["candidates"])
    assert n == 0


def test_warmup_blocks_flags(sim):
    paths, market, _ = sim
    a = read_csv(paths.session(market.days[303].isoformat()) / "analysis.csv")
    assert not a["baseline_ok"].any()
    assert not a["is_candidate"].any()
    assert a["data_issues"].str.contains("Baseline erst").all()


def test_boring_hint_ex_dividend(sim):
    paths, _, anom = sim
    a = read_csv(paths.session(anom) / "analysis.csv").set_index("symbol")
    assert "Ex-Dividende" in a.loc["EEE", "boring_hints"]


def test_unavailable_is_written_literally(sim):
    paths, market, _ = sim
    raw = (paths.session(market.days[301].isoformat()) / "analysis.csv").read_text()
    assert UNAVAILABLE in raw


def test_flags_log_and_evaluation(sim):
    paths, _, anom = sim
    log = read_csv(paths.data / "flags_log.csv")
    assert list(log["symbol"]) == ["AAA"] and log["session"].iloc[0] == anom
    out = read_csv(paths.evaluation / "outcomes.csv")
    row = out[(out["symbol"] == "AAA") & (out["session"] == anom)]
    assert len(row) == 1 and bool(row["hit"].iloc[0]) and bool(row["is_candidate"].iloc[0])
    summ = read_json(paths.evaluation / "summary.json")
    assert summ["groups"]["kandidaten"]["n"] == 1
    assert summ["groups"]["alle_tickertage"]["n"] > 100
    assert (paths.evaluation / "summary.md").exists()
    latest = read_json(paths.data / "latest.json")
    assert latest["latest_session"] >= anom


def test_stale_oi_is_detected_and_retry_fixes_it(tmp_path):
    market = FakeMarket()
    paths = make_paths(tmp_path)
    settings = load_settings(paths)
    d = market.days[310]
    nxt = market.days[311]
    run_evening(paths, FakeProvider(market, at(d, 17, 30)), settings, at(d, 17, 30))
    market.stale_oi_for.add(d)
    meta = run_morning(paths, FakeProvider(market, at(nxt, 7, 30)), settings, at(nxt, 7, 30))
    assert meta["oi_status"] == "STALE"
    analyze_session(paths, settings, d.isoformat())
    a = read_csv(paths.session(d.isoformat()) / "analysis.csv")
    assert a["delta_oi_total"].isna().all()
    assert a["data_issues"].str.contains("OI-Veränderung UNAVAILABLE").all()
    market.stale_oi_for.clear()
    meta2 = run_morning(paths, FakeProvider(market, at(nxt, 9, 0)), settings, at(nxt, 9, 0))
    assert meta2["oi_status"] == "OK" and meta2["attempt"] == 2
    meta3 = run_morning(paths, FakeProvider(market, at(nxt, 9, 5)), settings, at(nxt, 9, 5))
    assert meta3["status"] == "SKIPPED"


def test_evening_skips_non_trading_day(tmp_path):
    market = FakeMarket()
    paths = make_paths(tmp_path)
    sat = market.days[310]
    while sat.weekday() != 5:
        sat = date.fromordinal(sat.toordinal() + 1)
    meta = run_evening(paths, FakeProvider(market, at(sat, 17, 30)), load_settings(paths), at(sat, 17, 30))
    assert meta["status"] == "SKIPPED"
    assert not paths.sessions.exists()


def test_helpers():
    assert third_friday(2026, 9) == date(2026, 9, 18)
    assert third_friday(2026, 10) == date(2026, 10, 16)
    ev = structural_events("2026-09-14", 30)
    assert ev[0]["date"] == "2026-09-18" and ev[0]["type"] == "quad_witching"
    base = np.array([100, 110, 90, 105, 95, 100, 102, 98, 101, 99], dtype=float)
    assert robust_z(100, base) == pytest.approx(0, abs=0.3)
    assert robust_z(400, base) > 10
    assert np.isnan(robust_z(100, base[:3]))
    lo, hi = wilson(5, 50)
    assert 0.03 < lo < 0.1 < hi < 0.25


def test_yahoo_provider_with_stubbed_yfinance(monkeypatch):
    """Exercise YahooProvider against objects shaped like yfinance 1.x output (no network)."""
    import sys
    import types
    from collections import namedtuple

    from screener.provider import YahooProvider

    idx = pd.DatetimeIndex(pd.to_datetime(["2026-09-21", "2026-09-22"]).tz_localize("America/New_York"), name="Date")
    hist = pd.DataFrame({"Open": [1.0, 2.0], "High": [1.0, 2.0], "Low": [1.0, 2.0], "Close": [1.0, 2.0],
                         "Adj Close": [1.0, 2.0], "Volume": [10, 20]}, index=idx)
    chain_df = pd.DataFrame({"contractSymbol": ["X260925C00100000"], "lastTradeDate": pd.to_datetime(
        ["2026-09-22 19:59:00"]).tz_localize("UTC"), "strike": [100.0], "lastPrice": [1.0], "bid": [0.9], "ask": [1.1],
        "change": [0.0], "percentChange": [0.0], "volume": [5.0], "openInterest": [50.0],
        "impliedVolatility": [0.3], "inTheMoney": [False], "contractSize": ["REGULAR"], "currency": ["USD"]})
    Opt = namedtuple("Options", ["calls", "puts", "underlying"])

    class T:
        def __init__(self, sym):
            self.options = ("2026-09-25",)
            self.calendar = {"Earnings Date": [date(2026, 10, 20)], "Ex-Dividend Date": date(2026, 11, 10)}

        def history(self, **kw):
            assert kw["auto_adjust"] is False and kw["raise_errors"] is True
            return hist

        def option_chain(self, e):
            return Opt(chain_df, chain_df.assign(contractSymbol=["X260925P00100000"]), {"regularMarketPrice": 101.0})

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=T, __version__="stub"))
    p = YahooProvider(pause=0, retries=1)
    h = p.history("X")
    assert list(h["Date"]) == ["2026-09-21", "2026-09-22"] and "Adj Close" in h.columns
    assert p.expirations("X") == ["2026-09-25"]
    calls, puts, und = p.chain("X", "2026-09-25")
    from screener.fetch_data import contracts_frame
    c = contracts_frame(calls, puts, "2026-09-25")
    assert list(c["last_trade_date_et"]) == ["2026-09-22", "2026-09-22"]
    cal = p.calendar("X")
    assert str(cal["Earnings Date"][0]) == "2026-10-20"


def test_morning_failures_partial_staleness_and_session_guard(tmp_path):
    market = FakeMarket()
    paths = make_paths(tmp_path)
    settings = load_settings(paths)
    d, nxt = market.days[310], market.days[311]
    run_evening(paths, FakeProvider(market, at(d, 17, 30)), settings, at(d, 17, 30))
    top = read_csv(paths.session(d.isoformat()) / "evening_top_contracts.csv.gz")
    assert (top["expiry"].astype(str) > d.isoformat()).all()

    # every chain fails: status FEHLGESCHLAGEN and the retry is not skipped
    market.fail_chains = True
    m1 = run_morning(paths, FakeProvider(market, at(nxt, 7, 30)), settings, at(nxt, 7, 30))
    assert m1["oi_status"] == "FEHLGESCHLAGEN"
    market.fail_chains = False

    # two symbols still stale: TEILWEISE, their change is UNAVAILABLE, the others are real
    market.stale_symbols = {"AAA", "BBB"}
    m2 = run_morning(paths, FakeProvider(market, at(nxt, 8, 0)), settings, at(nxt, 8, 0))
    assert m2["status"] != "SKIPPED" and m2["oi_status"] == "TEILWEISE"
    assert m2["oi_stale_symbols"] == ["AAA", "BBB"]
    analyze_session(paths, settings, d.isoformat())
    a = read_csv(paths.session(d.isoformat()) / "analysis.csv").set_index("symbol")
    assert np.isnan(a.loc["AAA", "delta_oi_total"]) and pd.isna(a.loc["AAA", "oi_classification"])
    assert not np.isnan(a.loc["DDD", "delta_oi_total"])

    # third attempt only refetches the stale ones and ends OK
    market.stale_symbols = set()
    prov = FakeProvider(market, at(nxt, 8, 30))
    m3 = run_morning(paths, prov, settings, at(nxt, 8, 30))
    assert m3["oi_status"] == "OK" and m3["symbols_kept_from_previous_attempt"] == 4
    mo = read_csv(paths.session(d.isoformat()) / "morning_expiries.csv.gz")
    assert set(mo["symbol"]) == {u[0] for u in UNIVERSE} and not mo.duplicated(["symbol", "expiry", "type"]).any()
    mt = read_csv(paths.session(d.isoformat()) / "morning_top_contracts.csv.gz")
    assert (mt["note"].fillna("X") == "").all()  # empty note stays empty, not UNAVAILABLE

    with pytest.raises(ValueError):
        run_morning(paths, FakeProvider(market, at(nxt, 8, 45)), settings, at(nxt, 8, 45),
                    session=market.days[305].isoformat())


def test_status_values_survive_csv_roundtrip(tmp_path):
    from screener.common import write_csv
    f = tmp_path / "x.csv"
    write_csv(pd.DataFrame({"options_status": ["UNAVAILABLE"], "value": [float("nan")], "note": [""]}), f)
    back = read_csv(f)
    assert back.loc[0, "options_status"] == "UNAVAILABLE"
    assert np.isnan(back.loc[0, "value"]) and back.loc[0, "note"] == ""
