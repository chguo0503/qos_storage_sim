"""Independent client planner tests; no simulator or SSU state access needed."""

from collections import Counter

from coflow_client_policy import ClientRead, choose_npu, choose_npu_pipeline, plan_global_batch, priority
from shared_path_common import IO_GIB
from shared_path_strategy1 import KnownNPU


def test_assignment_uses_only_arrived_compute_and_stable_ties():
    assert choose_npu((8, 2, 4, 9), 0) == 1
    assert choose_npu((2, 2, 4, 9), 1) == 1
    assert choose_npu((2, 2, 4, 9), 1, (1, 2, 1, 1)) == 0


def test_pipeline_assignment_avoids_lane_with_hidden_io_backlog():
    rows = (KnownNPU(0, 1, (10000, 10000), 1), KnownNPU(1, 2, (10, 10), 1))
    chosen, scores = choose_npu_pipeline(rows, (10, 10), 1, 0)
    assert choose_npu((1, 2), 0) == 0
    assert chosen == 1 and scores[1] == 10
    assert scores[0] > scores[1]


def test_pipeline_assignment_ties_and_sharing_count():
    rows = (KnownNPU(0, 0, (0,), 0), KnownNPU(1, 0, (0,), 0))
    chosen, scores = choose_npu_pipeline(rows, (100,), 0, 1, n_layers=1)
    assert chosen == 1
    assert scores == (1000 * IO_GIB * 100 / 40,) * 2


def test_urgent_short_coflow_precedes_long_and_nonurgent():
    long = ClientRead(1, 1, 0, 0, (100, 100), (100, 100))
    short = ClientRead(2, 1, 1, 0, (2, 2), (2, 2))
    later = ClientRead(3, 1, 2, 20, (1, 1), (1, 1))
    assert priority(short, 0) < priority(long, 0) < priority(later, 0)
    grants = plan_global_batch((later, long, short), (0, 0), (0, 0, 0), 0,
                               max_commands=5)
    assert [g[0] for g in grants] == [2] * 4 + [1]


def test_future_deadlines_use_edf_and_work_conserving_unused_space():
    first = ClientRead(1, 1, 0, 20, (1, 1), (1, 1))
    later = ClientRead(2, 1, 1, 30, (1, 1), (1, 1))
    grants = plan_global_batch((later, first), (0, 0), (0, 0), 0)
    assert [g[0] for g in grants] == [1, 1, 2, 2]


def test_actual_sent_counters_bound_both_resources_without_mutation():
    reads = tuple(ClientRead(n, 1, n, 0, (100, 100), (100, 100)) for n in range(4))
    disks, links = [10, 15], [4, 8, 0, 0]
    grants = plan_global_batch(reads, disks, links, 0, max_commands=1000)
    by_disk = Counter(g[3] for g in grants)
    by_npu = Counter(g[2] for g in grants)
    disk_cap = int(40 * .25 / (1000 * IO_GIB))
    link_cap = int(50 * .25 / (1000 * IO_GIB))
    assert sum(by_disk.values()) == sum(disk_cap - q for q in disks)
    assert all(q + by_disk[s] <= disk_cap for s, q in enumerate(disks))
    assert all(q + by_npu[n] <= link_cap for n, q in enumerate(links))
    assert disks == [10, 15] and links == [4, 8, 0, 0]
    assert all(r.unsent_by_ssu == (100, 100) for r in reads)


def test_blocked_disk_does_not_block_other_disk_or_exceed_unsent_work():
    read = ClientRead(1, 1, 0, 0, (20, 2), (20, 3))
    grants = plan_global_batch((read,), (1000, 0), (0,), 0)
    assert grants == ((1, 1, 0, 1),) * 2
    assert plan_global_batch((read,), (1000, 1000), (0,), 0) == ()
    assert plan_global_batch((read,), (0, 0), (1000,), 0) == ()


def test_ack_credit_and_new_urgent_work_change_next_plan():
    cap = int(40 * .25 / (1000 * IO_GIB))
    older = ClientRead(1, 1, 0, 30, (5,), (10,))
    urgent = ClientRead(2, 0, 1, 1, (5,), (5,))
    assert plan_global_batch((older,), (cap,), (10, 0), 2) == ()
    assert plan_global_batch((older, urgent), (cap - 1,), (9, 0), 2) == ((2, 0, 1, 0),)


def test_batch_and_single_io_minimum_limits():
    read = ClientRead(1, 1, 0, 0, (100,), (100,))
    assert len(plan_global_batch((read,), (0,), (0,), 0, max_commands=3)) == 3
    assert len(plan_global_batch((read,), (0,), (0,), 0, queue_window_ms=.000001)) == 1
