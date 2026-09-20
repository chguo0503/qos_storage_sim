# Recovered archive notice

The original xy_underload_data_code.zip lacks its central directory and ends after source/fifo_20k_sensitivity.py.

Recovered 469 original entries using ZIP local headers; every entry passed CRC32 and uncompressed-length verification. All 72 native_summary.json.gz files match their original recorded SHA-256 and decode successfully. The 44 original present source files match recorded hashes. An additional 56 source files were recovered from other preserved archives only when SHA-256 exactly matched the experiment's source_sha256 records. No differing versions were substituted.

The primary run_experiment.py entry and all its import-time dependencies are now restored, including fixed_total_compat.py. Import validation succeeded without running any simulation. source/data and all original configurations and stored results are present. A fresh simulation was not executed during this organization task.

13 recorded source files remain absent; these are additional experiment/report helpers listed below. The root-level verify_and_export.py, summarize_theory.py and theory_summary.json referenced by the original README were not present in the recovered entries. Therefore the original full archive is not completely recovered, and the original README's full-matrix report rebuilding recipe remains incomplete.

The intact ttft_slo_cdf_bundle.zip and xy_ttft_cdf_data_code.zip independently reproduce CDF plots from preserved per-request CSV. Original experiment results are unchanged. See sibling xy_underload_data_code_recovery_audit.json and hash_matched_source_recovery.json for verification and provenance.

## Missing recorded sources

- plot_abcd_8npu.py
- plot_abcd_layer_demand_supply.py
- plot_abcd_total_demand.py
- proxy_4npu_agent.py
- report_4npu_exact_data.py
- report_ratio8_4npu.py
- report_ratio_sweep_4npu.py
- report_ratio_sweep_r40_4npu.py
- run_4npu_exact_data.py
- run_4npu_ratio8.py
- run_abcd_8npu.py
- run_abcd_bd_heavy_8npu.py
- run_abcd_fixed_8npu.py
