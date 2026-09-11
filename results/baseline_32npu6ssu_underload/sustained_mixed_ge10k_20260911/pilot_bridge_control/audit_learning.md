# Audit correction

The first input-only audit used exact binary equality between the sum of physical 176 KiB blocks and the data table's volume. S32/1024 and S48/1024 differ by one ULP (−6.938893903907228e−18 and −1.3877787807814457e−17 GiB). Exact block volume, count, per-block placement, and original C remain independently checked. The corrected assertion permits at most `math.ulp(original_data_volume)` and records every profile's difference in `analysis.json`. No simulator or frozen input was changed and no simulation was restarted.

This applies the self-improvement skill's error-recording guidance locally within the assigned directory; avoid replacing physical-byte checks with broad relative tolerances.
