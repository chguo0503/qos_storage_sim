import sim
from steady_state_workload import PROFILE_CYCLE, prepare_steady_state_workload


def test_steady_stream_is_balanced_and_prefix_stable():
    table = sim.load_bw_table_cache(num_npu=8)
    short = prepare_steady_state_workload(
        table,
        num_npu=8,
        num_ssu=4,
        n_layers=3,
        requests_per_npu=8,
    )
    extended = prepare_steady_state_workload(
        table,
        num_npu=8,
        num_ssu=4,
        n_layers=3,
        requests_per_npu=12,
    )

    short_requests = {request.request_id: request for request in short.requests}
    extended_requests = {request.request_id: request for request in extended.requests}
    assert set(short_requests) <= set(extended_requests)
    for request_id, request in short_requests.items():
        assert request == extended_requests[request_id]
        assert short.placement_by_request[request_id] == extended.placement_by_request[
            request_id
        ]

    for sequence in range(8):
        rows = [request for request in short.requests if request.stream_id == sequence]
        assert {category: sum(row.category == category for row in rows) for category in PROFILE_CYCLE} == {
            category: 2 for category in PROFILE_CYCLE
        }
        assert all(row.request_id == sequence * 8 + row.npu_id for row in rows)
    for npu_id in range(8):
        categories = {
            request.category
            for request in short.requests
            if request.npu_id == npu_id and request.stream_id < 4
        }
        assert categories == set(PROFILE_CYCLE)
