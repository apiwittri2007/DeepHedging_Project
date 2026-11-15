"""Transaction cost models."""

from __future__ import annotations


def proportional_cost(trade_notional_abs, rate):
    """Proportional cost: ``rate`` (decimal, e.g. 0.001 = 10 bps) times |notional|."""
    return rate * trade_notional_abs
