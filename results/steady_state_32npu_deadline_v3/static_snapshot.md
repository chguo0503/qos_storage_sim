# Real sequence-0 V3 allocation snapshot

All 32 requests are waiting; every allocator sees the same full 16-layer manifest and fixed SSU10 placement.

| Allocator | Selected | Selected categories | Rejected categories | SSD min | mean | max | NPU max |
|---|---:|---|---|---:|---:|---:|---:|
| deadline_barrier_v3 | 8 | {'LS': 2, 'SS': 6} | {'LL': 8, 'LS': 6, 'SL': 8, 'SS': 2} | 40.000 | 40.000 | 40.000 | 50.000 |
| admission_v1 | 26 | {'LL': 8, 'LS': 8, 'SL': 8, 'SS': 2} | {'SS': 6} | 35.204 | 37.757 | 40.000 | 45.954 |
| cardinality_first_explicit_spill | 26 | {'LL': 8, 'LS': 8, 'SL': 8, 'SS': 2} | {'SS': 6} | 40.000 | 40.000 | 40.000 | 45.098 |
