"""Structural calendar events that explain a lot of 'unusual' options volume.

Monthly equity options expire on the third Friday. In March, June, September and December that day is
quad witching and the S&P quarterly rebalance becomes effective after the close. These are computed,
not fetched; exchange holidays that move expiration to Thursday are NOT handled and are noted as such.
"""
from __future__ import annotations

from datetime import date, timedelta

from .common import parse_date

NOTE = "berechnet (dritter Freitag, ohne Feiertagskorrektur)"


def third_friday(year: int, month: int) -> date:
    d = date(year, month, 1)
    offset = (4 - d.weekday()) % 7  # Friday = 4
    return d + timedelta(days=offset + 14)


def structural_events(session: str, lookahead_days: int = 30) -> list[dict]:
    d0 = parse_date(session)
    end = d0 + timedelta(days=lookahead_days)
    out = []
    y, m = d0.year, d0.month
    for _ in range(3):
        tf = third_friday(y, m)
        if d0 <= tf <= end:
            quarterly = m in (3, 6, 9, 12)
            out.append({
                "date": tf.isoformat(),
                "days": (tf - d0).days,
                "type": "quad_witching" if quarterly else "monthly_opex",
                "label": ("Quad Witching und S&P-Quartalsrebalancing (wirksam nach Handelsschluss)"
                          if quarterly else "Monatlicher Optionsverfall"),
                "source": NOTE,
            })
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out
