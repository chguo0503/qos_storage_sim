"""The SSU must not be read between collector ticks, even after a write."""

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


def test_whole_ssu_cir_writes_are_separated_and_do_not_refresh_snapshot():
    state = PeriodicSSUState(1)
    state.collect(0, lambda s: ((1, 2), (3, 4)))
    writes = []
    def write(s, cirs, t):
        writes.append((s, cirs, t))
    assert not state.write_cir(0, (8, 9), 99.999, write)
    assert state.write_cir(0, (8, 9), 100, write)
    assert not state.write_cir(0, (10, 9), 150, write)
    assert state.write_cir(0, (10, 9), 200, write)
    assert state.cirs == ((3, 4),)
    assert len(writes) == 2
    assert state.fresh_reads_by_ssu == [1]


def test_control_minimum_periods_are_hard_constraints():
    with pytest.raises(ValueError):
        PeriodicSSUState(1, interval_ms=4.9)
    with pytest.raises(ValueError):
        PeriodicSSUState(1, cir_min_interval_ms=99)
