"""Periodic snapshots used by Baseline and Once; no dynamic CIR tests."""

import pytest

from shared_ssu_state import PeriodicSSUState


def test_lookup_does_not_read_hardware_or_refresh_stale_copy():
    reads = []
    state = PeriodicSSUState(2)
    def read(s):
        reads.append(s)
        return (len(reads), s), (1.0,)
    with pytest.raises(RuntimeError):
        state.get(0, 0)
    state.collect(0, read)
    old = state.get(0, 0)
    for time in (0.1, 1, 4.999):
        assert state.get(0, time) == old
    assert reads == [0, 1]
    with pytest.raises(ValueError):
        state.collect(4.9, read)
    state.collect(5, read)
    assert state.get(0, 5) != old
    assert reads == [0, 1, 0, 1]
    assert state.statistics()["max_snapshot_age_ms"] == 4.999


def test_sampling_minimum_period_is_a_hard_constraint():
    with pytest.raises(ValueError):
        PeriodicSSUState(1, interval_ms=4.9)
