"""Synthetic-only controls for independently reconstructed REFIT checks."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_stage_g_refit import (ARMS, CHANNELS, audit, audit_statistics, categorical_basis,
    check_chunks, check_chunk_trace, check_full_dp, chunk_objective, phase_scale, same, score,
    saved_or_checked_window)
from scripts.run_stage_g_refit import HOMES, predict, summarize_records


class StageGAuditTests(unittest.TestCase):
    def model(self):
        return dict(levels=[[0.,10.],[0.,3.],[0.],[-2.,0.]],
            penalties=[1.,.5,0.,.2], event_threshold=1.,thresholds=[5.,1.,1.],
            training_mean=[5.,1.,0.,-1.])

    def test_recursive_seed_metrics_and_empty_values(self):
        value={"seeds":[{"seed":None,"metric":3.5}],"none":[],"labels":["a"]}
        same(value,copy.deepcopy(value),"example")
        broken=copy.deepcopy(value)
        broken["seeds"][0]["metric"]=4.
        with self.assertRaises(ValueError):
            same(broken,value,"example")

    def test_direct_energy_prior_and_little_endian_order(self):
        basis=categorical_basis((2,2),2)
        np.testing.assert_array_equal(basis[1],[[1,0],[0,0]])
        model=dict(levels=[[0.,10.],[0.,2.]],penalties=[3.,5.])
        chunk=dict(aggregate=[4.,9.],weights=[2,1])
        energies=chunk_objective(chunk,model,np.array([1,0]),basis)
        for i,states in enumerate(basis):
            power=np.array([sum(model["levels"][c][s] for c,s in enumerate(row)) for row in states])
            expected=2*(4-power[0])**2+(9-power[1])**2
            expected+=sum(model["penalties"][c]*(states[0,c]!=[1,0][c]) for c in range(2))
            expected+=sum(model["penalties"][c]*(states[0,c]!=states[1,c]) for c in range(2))
            self.assertEqual(energies[i],expected)
        self.assertEqual(phase_scale(dict(aggregate=[4.],weights=[1]),
            dict(levels=[[0.,10.]],penalties=[3.]),np.array([1])),17.)

    def test_producer_fixture_has_independently_valid_trace_draws_and_dp(self):
        mains=np.array([0.,10.,10.,3.,0.,0.,10.])
        timestamps=np.array([0,30,60,90,120,210,240])
        model=self.model()
        angles=dict(gammas=[1.7],betas=[.4])
        pending,archive=predict(mains,timestamps,model,angles,1,2)
        runs=check_chunks(archive["chunks"],mains,timestamps,model["event_threshold"])
        for (method,seed,prediction,_),trace in zip(pending,archive["traces"]):
            if method in ("qaoa_ideal","uniform","exact_chunk"):
                _,draws=check_chunk_trace(trace,archive["chunks"],prediction,model,angles,
                    method,seed,1,2,replay_qaoa=True)
                self.assertEqual(draws,0 if method=="exact_chunk" else 256*len(archive["chunks"]))
            elif method=="exact_full":
                self.assertLess(check_full_dp(trace,runs,mains,prediction,model),1e-9)
        trace=copy.deepcopy(archive["traces"][0])
        trace["detail"]["records"][0]["previous_states"]=[0,0,0,0]
        with self.assertRaisesRegex(ValueError,"carry"):
            check_chunk_trace(trace,archive["chunks"],pending[0][2],model,angles,"qaoa_ideal",1907,1,2)

    def test_all_pooled_metrics_and_bootstraps_match_independently(self):
        records=[]
        for home in (1,2):
            for day,blocks in ((0,home),(1,home*3)):
                timestamps=np.arange(blocks)*30
                reference=np.full((blocks,3),float(home))
                for method,seed in ARMS:
                    power=home+float(day)+(1 if method=="qaoa_ideal" else 0)+(seed or 0)*1e-5
                    predicted=np.full((blocks,3),power)
                    records.append(dict(house=home,window_id=f"test-{day:03d}",model=method,seed=seed,
                        blocks=blocks,solver_wall_time_s=.01,raw_aggregate_mae_w=3*abs(home-power),
                        appliances=score(reference,predicted,timestamps,[1.]*3)))
        protocol=dict(homes=[dict(house=h) for h in HOMES],
            day_bootstrap=dict(replicates=2000,seed_plus_house=8101),
            home_bootstrap=dict(replicates=10000,seed=9101))
        summary=summarize_records(records)
        self.assertEqual(audit_statistics(records,protocol,summary),12)
        summary["paired_home_comparisons"][0]["descriptive_95_interval_w"][0]+=1.
        with self.assertRaises(ValueError):
            audit_statistics(records,protocol,summary)

    def test_completion_gate_precedes_data_access(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(ValueError,"complete evaluation summary"):
                audit(Path(path))

    def test_precheck_reuse_requires_identical_binding(self):
        with tempfile.TemporaryDirectory() as path:
            args=(Path(path),{"house":1},{},{},{"id":"test-000"},0,{},False,True)
            with patch('audit_stage_g_refit.window_binding',return_value={"immutable":"original"}), \
                 patch('audit_stage_g_refit.audit_saved_window',return_value={"count":1}) as check:
                self.assertFalse(saved_or_checked_window(*args)[2])
                self.assertTrue(saved_or_checked_window(*args)[2])
                self.assertEqual(check.call_count,1)
            with patch('audit_stage_g_refit.window_binding',return_value={"immutable":"changed"}):
                with self.assertRaisesRegex(ValueError,"Stale/tampered"):
                    saved_or_checked_window(*args)


if __name__=="__main__":
    unittest.main()
