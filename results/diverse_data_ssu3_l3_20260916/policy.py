"""Historical experiment import facade; active implementation is in simulator.

Frozen numerical artifacts retain their original source hashes. Strict replay
must use the source revision recorded with that experiment.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from simulator.policies.slo_pool import (
    RequestBudget, RouterConfig, PoolDecision, VARIANTS, choose_path_pool, slo_path_ids,
)
from simulator.policies.once import once_path_ids
from simulator.adapters.slo_pool import (
    POLICIES, simulator_strategy, make_stats, upper_budget, install_policy,
)
