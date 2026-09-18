"""Independent hand-calculated support, Issues, gap and zero-count controls."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from quantum_nilm.refit import read_block_window


class RefitIndependentHandAudit(unittest.TestCase):
    def test_joint_irregular_support_flags_gap_and_zero_prevalence(self):
        header = "Time,Unix,Aggregate,"+",".join(f"Appliance{i}" for i in range(1, 10))+",Issues\n"
        times = [0, 6, 14, 30, 38, 46, 54, 62, 70, 100, 108, 116, 124]
        powers = {0:30, 6:60, 14:90, 38:0, 108:0}
        appliances = {46:(5,0), 62:(0,7), 100:(10,10), 116:(2,0)}
        rows = []
        for t in times:
            a, b = appliances.get(t, (0,0))
            rows.append(",".join(map(str, ["2014-01-01 00:00:00", t, powers.get(t,100), a,b,*([0]*7),int(t == 30)]))+"\n")
        content = (header+"".join(rows)).encode()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"hand_case.csv"
            path.write_bytes(content)
            result = read_block_window(path, 0, 120, channels=("Appliance1", "Appliance2"))
        # 0--30: 6*30+8*60+16*90 = 2,100 W s, with both appliances zero.
        np.testing.assert_array_equal(result["timestamps"], [0])
        np.testing.assert_allclose(result["values"], [[70.,0.,0.]])
        self.assertEqual(result["coverage"]["valid_seconds"], [30.,22.,10.,20.])
        self.assertEqual(result["coverage"]["supported_seconds"], [30.,30.,10.,20.])
        self.assertEqual(result["coverage"]["large_gap_seconds"], [0.,0.,20.,10.])
        q = result["quality"]
        for field, expected in dict(valid_blocks=1,rejected_blocks=3,source_rows_scanned=13,
                                    source_rows_within_window=12,support_successor_rows=1,
                                    valid_source_rows=12,rows_issues=1,
                                    supported_invalid_seconds=8.,unsupported_large_gap_seconds=30.,
                                    valid_covered_seconds=82.,uncovered_boundary_seconds=0.,
                                    all_selected_appliances_zero_source_rows=8,
                                    all_selected_appliances_zero_retained_blocks=1).items():
            self.assertEqual(q[field], expected, field)
        self.assertEqual(q["zero_source_rows_by_channel"], {"Aggregate":2,"Appliance1":9,"Appliance2":10})
        self.assertEqual(q["zero_retained_blocks_by_channel"], {"Aggregate":0,"Appliance1":1,"Appliance2":1})
        self.assertEqual(result["source_window_sha256"], hashlib.sha256(content).hexdigest())

    def test_large_file_byte_seek_matches_linear_window_support(self):
        header = "Time,Unix,Aggregate,"+",".join(f"Appliance{i}" for i in range(1,10))+",Issues\n"
        rows=[]
        times=np.arange(3000)*8
        for i,t in enumerate(times):
            rows.append(",".join(map(str,["2014-01-01 00:00:00",t,i+100,i%7,*([0]*8),0]))+"\n")
        content=(header+"".join(rows)).encode()
        self.assertGreater(len(content),65536)
        start,end=19003,19093
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"seek_case.csv"
            path.write_bytes(content)
            result=read_block_window(path,start,end,channels=("Appliance1",))
        expected=[]
        for boundary in range(start,end,30):
            energy=np.zeros(2)
            for i,(a,b) in enumerate(zip(times[:-1],times[1:])):
                overlap=max(0,min(int(b),boundary+30)-max(int(a),boundary))
                energy+=overlap*np.array([i+100,i%7])
            expected.append(energy/30)
        np.testing.assert_array_equal(result["timestamps"],[19003,19033,19063])
        np.testing.assert_allclose(result["values"],expected,rtol=0,atol=1e-12)
        lo=int(np.searchsorted(times,start-16))
        hi=int(np.searchsorted(times,end))
        support=(header+"".join(rows[lo:hi+1])).encode()
        self.assertEqual(result["source_window_sha256"],hashlib.sha256(support).hexdigest())
        self.assertEqual(result["quality"]["first_scanned_unix"],int(times[lo]))
        self.assertEqual(result["quality"]["last_scanned_unix"],int(times[hi]))


if __name__ == "__main__":
    unittest.main()
