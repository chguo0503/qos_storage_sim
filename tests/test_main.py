import pickle
from types import SimpleNamespace

import main
from sim import IOSchedulingConfig


def test_partial_fair_cache_only_runs_missing_mode(tmp_path, monkeypatch):
    cache_file = tmp_path / "sweep.pkl"
    fair_key = (1, 0.5, "baseline_fair")
    qwrr_key = (1, 0.5, "baseline_qwrr")
    cached = {
        "ssu_list": [1],
        "ls_ratio_list": [0.5],
        "mode_names": ["baseline_fair"],
        "results": {fair_key: [10.0]},
        "seeds": [42],
        "mode_policies": {"baseline_fair": "fair"},
        "qos_policy_version": main.QOS_POLICY_VERSION,
        "fair_policy_version": main.FAIR_POLICY_VERSION,
        "qos_queue_count": main.QOS_QUEUE_COUNT,
        "qos_layout": main.QOS_LAYOUT,
        "batch_dispatch_interval_us": main.BATCH_DISPATCH_INTERVAL_US,
        "qos_queue_max_active_flows": main.QOS_QUEUE_MAX_ACTIVE_FLOWS,
    }
    with cache_file.open("wb") as file_obj:
        pickle.dump(cached, file_obj)

    monkeypatch.setattr(main, "PICKLE_FILE", str(cache_file))
    monkeypatch.setattr(main, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(main, "NPU_COUNT", 1)
    monkeypatch.setattr(main, "N_LAYERS", 1)
    monkeypatch.setattr(main, "N_REQ", 1)
    monkeypatch.setattr(main, "SSU_LIST", [1])
    monkeypatch.setattr(main, "LS_RATIO_LIST", [0.5])
    monkeypatch.setattr(main, "SEEDS", [42])
    monkeypatch.setattr(
        main,
        "MODES",
        [
            ("baseline_fair", "fair", IOSchedulingConfig()),
            ("baseline_qwrr", "queue_wrr", IOSchedulingConfig()),
        ],
    )
    monkeypatch.setattr(main, "load_bw_table_cache", lambda num_npu: object())

    calls = []

    def fake_simulate_continuous(**kwargs):
        calls.append(kwargs["policy"])
        return [SimpleNamespace(compute_end_time=1.0, total_compute_ms=0.2)], {}

    monkeypatch.setattr(main, "simulate_continuous", fake_simulate_continuous)

    data = main.run_sweep()

    assert calls == ["queue_wrr"]
    assert data["mode_names"] == ["baseline_fair", "baseline_qwrr"]
    assert data["results"][fair_key] == [10.0]
    assert data["results"][qwrr_key] == [20.0]

    with cache_file.open("rb") as file_obj:
        persisted = pickle.load(file_obj)
    assert persisted["mode_names"] == ["baseline_fair", "baseline_qwrr"]
    assert persisted["qos_policy_version"] == main.QOS_POLICY_VERSION
    assert persisted["fair_policy_version"] == main.FAIR_POLICY_VERSION
