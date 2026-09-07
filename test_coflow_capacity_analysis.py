import pytest

from coflow_capacity_analysis import ActivatedCoflow, capacity_overloads, solo_read_lower_bound_ms
from shared_path_common import IO_GIB


def coflow(rid, ssd, link, deadline, *, npu=None, release=0):
    return ActivatedCoflow(rid, 1, rid if npu is None else npu, release, deadline,
                           tuple(ssd), link)


def test_ready_prefix_certifies_shared_ssd_but_not_separate_link_overload():
    reads = (coflow(1, (150,), 150, 1), coflow(2, (150,), 150, 1))
    result = capacity_overloads(reads, 0)
    assert len(result) == 1
    witness = result[0]
    assert (witness.resource, witness.resource_id, witness.start_ms, witness.end_ms) == ("ssu", 0, 0, 1)
    assert witness.required_gib == 300 * IO_GIB
    assert witness.capacity_gib == .04
    assert witness.required_service_ms == pytest.approx(1.25885009765625)
    assert witness.coflow_ids == ((1, 1), (2, 1))
    assert witness.kind == "ready_prefix"


def test_single_npu_link_can_overload_when_both_disks_pass():
    result = capacity_overloads((coflow(7, (300, 300), 600, 1.5),), 0)
    assert len(result) == 1
    assert result[0].resource == "npu_link" and result[0].resource_id == 7
    assert result[0].required_service_ms == pytest.approx(2.01416015625)


def test_later_deadline_work_is_not_charged_to_current_deadline():
    reads = (coflow(1, (200,), 200, 1), coflow(2, (200,), 200, 3))
    assert capacity_overloads(reads, 0) == ()


def test_data_already_off_ssd_is_only_link_work():
    result = capacity_overloads((coflow(1, (0,), 1000, 1),), 0)
    assert len(result) == 1 and result[0].resource == "npu_link"


def test_receive_links_are_per_npu_not_one_global_fifty_gib_link():
    reads = (coflow(1, (0,), 200, 1), coflow(2, (0,), 200, 1))
    assert capacity_overloads(reads, 0) == ()
    shared = (coflow(1, (0,), 200, 1, npu=9), coflow(2, (0,), 200, 1, npu=9))
    result = capacity_overloads(shared, 0)
    assert len(result) == 1 and result[0].resource_id == 9


def test_release_interval_finds_overload_that_a_long_now_prefix_misses():
    reads = (coflow(1, (300,), 300, 11, release=10),)
    assert capacity_overloads(reads, 0, release_intervals=False) == ()
    result = capacity_overloads(reads, 0)
    assert result
    assert all(row.start_ms == 10 and row.end_ms == 11 for row in result)
    assert all(row.kind == "release_interval" for row in result)


def test_old_releases_are_clamped_to_now_for_remaining_work():
    result = capacity_overloads((coflow(1, (300,), 300, 11, release=0),), 10)
    assert result
    assert all(row.start_ms == 10 and row.end_ms == 11 for row in result)


def test_partial_active_io_can_be_expressed_without_whole_command_overcount():
    half_service = 1000 * .5 * IO_GIB / 40
    assert capacity_overloads((coflow(1, (.25,), .25, half_service),), 0) == ()


def test_exact_capacity_equality_is_not_an_overload():
    deadline = 1000 * 300 * IO_GIB / 40
    assert capacity_overloads((coflow(1, (300,), 300, deadline),), 0) == ()


def test_already_expired_deadline_is_not_labeled_a_new_future_capacity_proof():
    result = capacity_overloads((coflow(1, (1,), 1, 2),), 5)
    assert result
    assert all(row.start_ms == row.end_ms == 5 for row in result)
    assert all(row.kind == "already_due_unfinished" for row in result)


def test_expired_work_is_not_circular_proof_of_an_additional_future_miss():
    reads = (coflow(1, (1000,), 1000, 1), coflow(2, (1,), 1, 6))
    result = capacity_overloads(reads, 5)
    assert result
    assert all(row.end_ms == 5 and row.kind == "already_due_unfinished" for row in result)


def test_release_after_deadline_is_a_fixed_input_infeasibility():
    result = capacity_overloads((coflow(1, (1,), 1, 10, release=11),), 0)
    assert any(row.start_ms == row.end_ms == 10 and row.capacity_gib == 0 for row in result)


def test_no_overload_does_not_guarantee_packetized_two_stage_feasibility():
    # One command must traverse SSD THEN the link, not both simultaneously.
    deadline = .005
    assert capacity_overloads((coflow(1, (1,), 1, deadline),), 0) == ()
    actual_isolated_minimum = 1000 * IO_GIB * (1 / 40 + 1 / 50)
    assert solo_read_lower_bound_ms((1,)) < deadline < actual_isolated_minimum


def test_empty_or_completed_work_has_no_certificate():
    assert capacity_overloads((), 0) == ()
    assert capacity_overloads((coflow(1, (0, 0), 0, 0),), 0) == ()


def test_solo_read_lower_bound_uses_hottest_disk_and_one_total_link():
    assert solo_read_lower_bound_ms((1000, 1000)) == pytest.approx(1000 * IO_GIB * 2000 / 50)
    assert solo_read_lower_bound_ms((2000, 0)) == pytest.approx(1000 * IO_GIB * 2000 / 40)
