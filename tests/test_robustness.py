"""Failures seen on the first live day (23.09.2026) and the guards against them."""
from screener.common import load_settings, read_csv, read_csv_if_exists, read_json
from screener.evaluate_flags import evaluate
from screener.fetch_data import previous_session, run_evening, run_morning
from screener.run import main
from tests.fake_market import FakeMarket, FakeProvider
from tests.test_pipeline import at, make_paths


def test_empty_outcomes_file_does_not_break_the_next_run(tmp_path):
    paths = make_paths(tmp_path)
    settings = load_settings(paths)
    evaluate(paths, settings)  # no analysed session yet
    f = paths.evaluation / "outcomes.csv"
    assert f.stat().st_size > 0 and read_csv(f).empty  # header only
    evaluate(paths, settings)
    f.write_text("")  # a 0-byte file from an older version
    assert read_csv_if_exists(f) is None
    evaluate(paths, settings)


def test_morning_finds_the_session_when_the_price_history_lags(tmp_path):
    market = FakeMarket()
    paths = make_paths(tmp_path)
    settings = load_settings(paths)
    d, nxt = market.days[310], market.days[311]
    run_evening(paths, FakeProvider(market, at(d, 17, 30)), settings, at(d, 17, 30))
    market.history_gaps.add(d)  # Yahoo shows no daily bar for d yet
    prov = FakeProvider(market, at(nxt, 8, 0))
    assert previous_session(prov, at(nxt, 8, 0)) == market.days[309].isoformat()  # the old behaviour
    assert previous_session(prov, at(nxt, 8, 0), paths) == d.isoformat()
    meta = run_morning(paths, prov, settings, at(nxt, 8, 0))
    assert meta["session"] == d.isoformat() and meta["oi_status"] == "OK"
    assert not paths.session(market.days[309].isoformat()).exists()


def test_evening_backup_runs_skip_and_late_runs_record_the_previous_session(tmp_path):
    market = FakeMarket()
    paths = make_paths(tmp_path)
    settings = load_settings(paths)
    d, nxt = market.days[310], market.days[311]
    m1 = run_evening(paths, FakeProvider(market, at(d, 17, 30)), settings, at(d, 17, 30))
    assert m1["status"] == "OK" and m1["late_run"] is False
    m2 = run_evening(paths, FakeProvider(market, at(d, 19, 30)), settings, at(d, 19, 30))
    assert m2["status"] == "SKIPPED" and "bereits" in m2["reason"]

    d2, nxt2 = market.days[312], market.days[313]
    late = run_evening(paths, FakeProvider(market, at(nxt2, 1, 30)), settings, at(nxt2, 1, 30))
    assert late["status"] == "OK" and late["session"] == d2.isoformat() and late["late_run"] is True
    d3, nxt3 = market.days[314], market.days[315]
    too_late = run_evening(paths, FakeProvider(market, at(nxt3, 4, 0)), settings, at(nxt3, 4, 0))
    assert too_late["status"] == "SKIPPED" and not paths.session(d3.isoformat()).exists()


def test_skipped_morning_retry_still_analyses_when_analysis_is_missing(tmp_path):
    market = FakeMarket()
    paths = make_paths(tmp_path)
    settings = load_settings(paths)
    d, nxt = market.days[310], market.days[311]
    run_evening(paths, FakeProvider(market, at(d, 17, 30)), settings, at(d, 17, 30))
    run_morning(paths, FakeProvider(market, at(nxt, 7, 30)), settings, at(nxt, 7, 30))  # stopped before analysis
    assert not (paths.session(d.isoformat()) / "analysis.csv").exists()
    now = at(nxt, 8, 30)
    assert main(["morning", "--now", now.isoformat()], provider=FakeProvider(market, now), paths=paths) == 0
    assert (paths.session(d.isoformat()) / "analysis.csv").exists()
    assert read_json(paths.data / "latest.json")["latest_session"] == d.isoformat()
