"""Command line entry point.

    python -m screener.run evening            # after the US close
    python -m screener.run morning            # next morning: OI after the session, then analysis and grading
    python -m screener.run analyze --session 2026-09-22
    python -m screener.run evaluate
    python -m screener.run smoke              # quick live test with 3 tickers, writes nothing

Options: --symbols AAPL,MSFT to limit the universe, --allow-past for manual evening runs on non-trading days.
"""
from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .common import Paths, load_settings, setup_logging, utc_now
from .analyze import analyze_session
from .evaluate_flags import evaluate
from .fetch_data import run_evening, run_morning

log = logging.getLogger("screener")


def make_provider(settings: dict):
    from .provider import YahooProvider

    return YahooProvider(pause=float(settings["request_pause_seconds"]), retries=int(settings["request_retries"]))


def main(argv: list[str] | None = None, provider=None, paths: Paths | None = None) -> int:
    ap = argparse.ArgumentParser(prog="screener")
    ap.add_argument("mode", choices=["evening", "morning", "analyze", "evaluate", "smoke"])
    ap.add_argument("--session", help="Handelstag YYYY-MM-DD (morning/analyze)")
    ap.add_argument("--symbols", help="kommagetrennte Teilmenge des Universums")
    ap.add_argument("--allow-past", action="store_true", help="Abendlauf auch ohne heutigen Tagesbalken")
    ap.add_argument("--force", action="store_true", help="Morgenlauf auch wenn OI schon erfasst ist")
    ap.add_argument("--now", help="Zeitpunkt überschreiben (ISO, für Tests)")
    args = ap.parse_args(argv)
    setup_logging()

    paths = paths or Paths()
    settings = load_settings(paths)
    now = datetime.fromisoformat(args.now).astimezone(timezone.utc) if args.now else utc_now()
    symbols = [s.strip() for s in args.symbols.split(",")] if args.symbols else None

    if args.mode == "smoke":
        return smoke(settings, provider)

    if args.mode == "evening":
        run_evening(paths, provider or make_provider(settings), settings, now, args.allow_past, symbols)
        return 0

    if args.mode == "morning":
        meta = run_morning(paths, provider or make_provider(settings), settings, now, args.session, args.force, symbols)
        session = meta["session"]
        analyzed = (paths.session(session) / "analysis.csv").exists()
        if meta.get("status") != "SKIPPED" or not analyzed:
            # A skipped retry still analyses when an earlier attempt fetched the data but stopped before analysing.
            analyze_session(paths, settings, session)
            evaluate(paths, settings)
        return 0

    if args.mode == "analyze":
        if not args.session:
            ap.error("--session fehlt")
        analyze_session(paths, settings, args.session)
        evaluate(paths, settings)
        return 0

    if args.mode == "evaluate":
        evaluate(paths, settings)
        return 0
    return 1


def smoke(settings: dict, provider=None) -> int:
    """Live check of the data source with three tickers in a temporary folder. Prints what came back."""
    import shutil

    provider = provider or make_provider(settings)
    spy = provider.history("SPY", "1mo")
    print("SPY, letzte Tagesbalken (Provisional = aus Kursquote ergänzt):")
    print(spy.tail(3).to_string(index=False))
    if hasattr(provider, "quote"):
        print("SPY Kursquote:", provider.quote("SPY"))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        shutil.copytree(Paths().config, root / "config")
        p = Paths(root)
        meta = run_evening(p, provider, settings, utc_now(), allow_past=True, symbols=["AAPL", "MSFT", "JPM"])
        print("Abend-Test:", meta.get("status"), meta.get("session"), meta.get("options_status_counts"))
        for e in meta.get("errors", []):
            print("  Fehler:", e)
        summ = p.session(meta["session"]) / "evening_summary.csv"
        if summ.exists():
            import pandas as pd

            df = pd.read_csv(summ)
            print(df[["symbol", "close", "return_1d", "options_status", "total_option_volume", "call_oi_start",
                      "options_last_trade_et", "earnings_dates"]].to_string(index=False))
        ok = meta.get("options_status_counts", {}).get("OK", 0)
        print("ERGEBNIS:", "Datenquelle funktioniert" if ok else "Optionsdaten nicht vollständig, siehe Fehler oben")
        return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
