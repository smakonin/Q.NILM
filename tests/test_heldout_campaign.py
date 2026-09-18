"""Small end-to-end campaign fixture; never touches the user's real data."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class HeldoutCampaignTests(unittest.TestCase):
    def test_frozen_train_validation_test_artifacts_and_overwrite_guard(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "synthetic.csv"
            output = Path(temporary) / "campaign"
            with source.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["unix_ts", "marker", "main", "dryr", "frdg", "vacu"])
                for second in range(1800):
                    dryer = 5 + 100 * ((second // 30) % 4)
                    fridge = 4 + 50 * ((second // 60) % 3)
                    vacuum = 2 + 200 * ((second // 90) % 2)
                    background = 30 + 10 * ((second // 120) % 3)
                    writer.writerow([second, "s" if second == 10 else "", dryer + fridge + vacuum + background, dryer, fridge, vacuum])
            env = dict(os.environ, PYTHONPATH=str(root / "src"))
            command = [sys.executable, str(root / "scripts/run_heldout_campaign.py"),
                       "--source", str(source), "--output-dir", str(output),
                       "--counts", "1", "1", "1", "--window-seconds", "300"]
            result = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            protocol = json.loads((output / "protocol.json").read_text())
            frozen = json.loads((output / "frozen_models.json").read_text())
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["protocol_sha256"], hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest())
            self.assertEqual(frozen["protocol_sha256"], summary["protocol_sha256"])
            self.assertEqual(len(summary["pooled_results"]), 14)
            self.assertEqual(len(protocol["manifest"]["windows"]), 3)
            for line in (output / "checksums.sha256").read_text().splitlines():
                expected, name = line.split(maxsplit=1)
                self.assertEqual(expected, hashlib.sha256((output / name).read_bytes()).hexdigest())
            before = (output / "summary.json").read_bytes()
            rerun = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertEqual(before, (output / "summary.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
