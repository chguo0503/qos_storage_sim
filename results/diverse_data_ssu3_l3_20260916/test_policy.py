"""Bounded checks for the unchanged heuristic on diverse raw-data profiles."""

from pathlib import Path
from types import SimpleNamespace as NS
import ast
import importlib.util
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]
import policy
from simulator.core import sim
from simulator.policies.policy_logic import category_path_ids, hardware_view
from simulator.policies.common import pressure_from_counts
from simulator.policies.profiles import FINAL_STATIC


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qos = hardware_view(FINAL_STATIC.hardware_config())
        cls.snapshot = pressure_from_counts(tuple((p*7+3)%11 for p in range(256)), cls.qos)
        cls.data = ast.literal_eval((ROOT/"data").read_text())
        old = ROOT/"results/baseline_ab128_32_ratio12_20260912/slo_routing_ssu3_20260915/slo_router.py"
        spec = importlib.util.spec_from_file_location("frozen_slo_router_for_test", old)
        cls.old = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.old
        spec.loader.exec_module(cls.old)

    def fixture(self, c, volume):
        request = NS(admitted=True, batch_size=1, per_layer_compute_ms=c,
            batch_id=0, admission_time_ms=0.,
            manifest=NS(placement=(((0,volume/3),(1,volume/3),(2,volume/3)),)))
        return NS(requests={1:request}, n_layers=8, num_ssu=3)

    def test_all_raw_profiles_static_rule_and_original_helper_equivalence(self):
        packed = full = 0
        for (length,miss), (_b,compute_us,_ttft,volume) in self.data.items():
            c = compute_us/1000
            ctx = self.fixture(c,volume)
            budgets = [policy.upper_budget(ctx,NS(request_id=1,layer=layer),now,"static")
                       for layer,now in ((0,0.),(4,9000.),(7,999999.))]
            self.assertEqual(budgets[0],budgets[1])
            self.assertEqual(budgets[0],budgets[2])
            allowed = category_path_ids(sim.classify_request(length,miss),self.qos)
            decision = policy.choose_path_pool(allowed,self.qos,budget=budgets[0])
            expected = c/2 > max(6.,1000*max(volume/3/40,volume/50))
            self.assertEqual(decision.mode=="slack_shared_prefix",expected)
            packed += expected
            full += not expected
            old_budget = self.old.RequestBudget(**vars(budgets[0]))
            for name in ("mild","aggressive"):
                new = policy.slo_path_ids(13,self.snapshot,allowed,self.qos,
                    budget=budgets[0],config=policy.VARIANTS[name],start_offset=17)
                old = self.old.slo_path_ids(13,self.snapshot,allowed,self.qos,
                    budget=old_budget,config=self.old.VARIANTS[name],start_offset=17)
                self.assertEqual(new,old)
                self.assertTrue(set(new)<=set(allowed))
        self.assertGreater(packed,0)
        self.assertGreater(full,0)

    def test_late_short_compute_can_be_packed_dynamically(self):
        c = self.data[128,256][1]/1000
        ctx = self.fixture(c,self.data[128,256][3])
        ctx.microbatches=[NS(compute_active_layer=0,
            layer_metrics=[NS(compute_start_ms=k*c) for k in range(8)])]
        allowed = category_path_ids("LS",self.qos)
        early = policy.upper_budget(ctx,NS(request_id=1,layer=1),0.,"aggressive")
        self.assertEqual(policy.choose_path_pool(allowed,self.qos,budget=early).mode,"urgent_full_pool")
        ctx.microbatches[0].compute_active_layer=6
        late = policy.upper_budget(ctx,NS(request_id=1,layer=7),6*c,"aggressive")
        self.assertEqual(policy.choose_path_pool(allowed,self.qos,budget=late).mode,"slack_shared_prefix")
        fixed = policy.upper_budget(ctx,NS(request_id=1,layer=7),6*c,"static")
        self.assertEqual(policy.choose_path_pool(allowed,self.qos,budget=fixed).mode,"urgent_full_pool")

    def test_unknown_admission_is_exact_once_for_every_variant(self):
        ctx=self.fixture(10.,.1)
        ctx.requests[1].admitted=False
        for name in ("mild","aggressive","static"):
            budget=policy.upper_budget(ctx,NS(request_id=1,layer=0),999.,name)
            self.assertIsNone(budget)
            allowed=category_path_ids("LL",self.qos)
            self.assertEqual(policy.slo_path_ids(31,self.snapshot,allowed,self.qos,budget=budget,
                             config=policy.VARIANTS[name]),
                             policy.once_path_ids(31,self.snapshot,allowed,self.qos))

    def test_install_restores_entry_point_without_running_simulation(self):
        from simulator.adapters import shared_path as shared
        before=shared.shared_path_adapter
        for name in policy.POLICIES:
            stats=policy.make_stats()
            with policy.install_policy(name,stats) as actual:
                self.assertIs(actual,stats)
                self.assertIsNot(shared.shared_path_adapter,before)
            self.assertIs(shared.shared_path_adapter,before)
        with self.assertRaises(ValueError):
            policy.simulator_strategy("unknown")


if __name__ == "__main__":
    unittest.main()
