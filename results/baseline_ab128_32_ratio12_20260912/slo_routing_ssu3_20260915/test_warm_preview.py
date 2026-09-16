"""Regression for finalized cohorts, live list shape, and clipped computation."""
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_trial import warm_statistics


class WarmPreviewTests(unittest.TestCase):
    def fixture(self):
        reqs={i:NS(request_id=i,load={'per_layer_us':10000.,'role':role})
              for i,role in ((1,'A'),(2,'B'),(3,'A'))}
        states={i:NS(manifest=q,admitted=True,completed=True,
                     admission_time_ms=a,completion_time_ms=z)
                for (i,q),(a,z) in zip(reqs.items(),((2100.,2200.),(3950.,4100.),(4000.,4050.)))}
        context=NS(requests=states,microbatches=[NS(layer_metrics=[
            NS(compute_start_ms=1990.,compute_duration_ms=20.),
            NS(compute_start_ms=3980.,compute_duration_ms=40.),
            NS(compute_start_ms=float('nan'),compute_duration_ms=0.)])])
        return context,reqs

    def test_live_list_shape_boundary_and_late_completion(self):
        context,reqs=self.fixture()
        out=warm_statistics(context,reqs)
        self.assertEqual(out['request_ids'],[1,2])
        self.assertEqual(out['slo']['overall'],{'count':2,'passed':1,'percent':50.})
        self.assertAlmostEqual(out['U_percent'],100*30/64000.)

    def test_unfinished_window_request_cannot_disappear(self):
        context,reqs=self.fixture()
        context.requests[2].completed=False
        self.assertIsNone(warm_statistics(context,reqs))


if __name__=='__main__':
    unittest.main()
