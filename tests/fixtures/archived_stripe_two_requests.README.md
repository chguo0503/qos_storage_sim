# Historical stripe manifest fixture

`archived_stripe_two_requests.json.gz` is a 1381-byte, self-contained subset
of a committed historical input. Tests do not require Git or any `results/`
directory to read it.

- Source commit: `1c2bb30acdf8c721476b0294cb31079f0fab6fba`.
- Source path: `results/baseline_ab128_32_ratio12_20260912/slo_routing_ssu3_20260915/runs/baseline_seed7_remote_v2/manifest.json.gz`.
- Source file SHA256: `ec6337cc9939bcfa56279166502db06df11cc7471707fa33a28076cd8f36355b`.
- Selected request IDs: `0` and `1000003`, on NPUs 0 and 1. Their original
  identity keys are `67` and `1000069`; their complete layer placements contain
  224 and 1022 blocks respectively.
- Fixture SHA256: `2845b8214be2f7a27a91924889f8bdb675b5e813ae8dc043d4c87b04e20109ba`.

Both request records, load dictionaries, and complete placement values were
copied from `git show` of the source commit without regenerating any placement.
Only the deduplicated placement table indices and top-level input fingerprint
were recomputed for the two-request subset. The selected metadata fields retain
their original values, including `layout`, `placement_rule`, original NPU/SSU
counts, layers, order, and seed. A separate `fixture_provenance` field records
the extraction; original population counts and population fingerprints were
not mislabeled as properties of this subset. Compression uses gzip `mtime=0`.

This tests backward-compatible reading, not permission to generate new stripe
inputs. The test checks the exact historical stripe pattern on two different
NPUs, original identities and metadata, the subset fingerprint, and unchanged
file bytes after loading. New workload generation remains Ring Hash only.
