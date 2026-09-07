from types import SimpleNamespace

import pytest

from coflow_disk_policy import PathCandidate, choose_path, local_priority
from coflow_joint_policy import coflow_priority
from coflow_client_policy import ClientRead, priority as client_priority
from shared_path_common import IO_GIB


def test_priority_cannot_escape_minimum_tag_equivalence_set():
    choices = (PathCandidate(0, 1.0, (20,)), PathCandidate(32, 1.0, (10,)),
               PathCandidate(64, 1.1, (-100,)))
    assert choose_path(choices, 0).path_id == 32


def test_native_round_robin_breaks_equal_priorities():
    choices = tuple(PathCandidate(p, 1, (0,)) for p in (0, 32, 64))
    assert choose_path(choices, 1).path_id == 32
    assert choose_path(choices, 65).path_id == 0


def test_epsilon_matches_native_tag_tolerance():
    choices = (PathCandidate(0, 1, (20,)), PathCandidate(32, 1 + .5e-12, (10,)),
               PathCandidate(64, 1 + 2e-12, (0,)))
    assert choose_path(choices, 0).path_id == 32


def test_local_priority_uses_only_submitted_metadata():
    flow = SimpleNamespace(deadline_time=10, layer_work_gb=4 * IO_GIB, enqueue_time=2)
    assert local_priority(flow, 5) == (10, 4, 2)
    assert local_priority(flow, 20) == (10, 4, 2)


def test_global_coflow_estimate_accounts_for_disk_and_one_link():
    key = coflow_priority(20, (1000, 1000), 5)
    remaining = 1000 * IO_GIB * 2000 / 50
    assert key == pytest.approx((15 - remaining, remaining, 20))
    disk = coflow_priority(20, (2000, 0), 5)
    assert disk[1] == pytest.approx(1000 * IO_GIB * 2000 / 40)
    assert coflow_priority(20, (0, 0), 5) == (15, 0, 20)


def test_urgent_short_does_not_put_large_already_late_coflow_first():
    large, small = (1000,), (1,)
    assert coflow_priority(0, large, 0) < coflow_priority(0, small, 0)
    assert coflow_priority(0, small, 0, rule="urgent_short") < \
           coflow_priority(0, large, 0, rule="urgent_short")


def test_urgent_short_priority_prefix_matches_global_client_objective():
    for deadline, counts, now in ((0, (1000,), 0), (0, (1,), 0),
                                   (100, (100, 100), 5), (80, (1000, 10), 7)):
        read = ClientRead(1, 1, 0, deadline, counts, counts)
        assert coflow_priority(deadline, counts, now, rule="urgent_short") == \
               client_priority(read, now)[:3]
    # Nonurgent work remains ordered by deadline, not shortest remaining work.
    assert coflow_priority(80, (1000,), 0, rule="urgent_short") < \
           coflow_priority(100, (1,), 0, rule="urgent_short")


def test_unknown_joint_rule_does_not_silently_use_another_policy():
    with pytest.raises(ValueError, match="rule"):
        coflow_priority(0, (1,), 0, rule="unknown")
