import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("boundary", Path(__file__).resolve().parents[1] / "scripts/check_distribution_boundary.py")
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)


class DistributionBoundaryTests(unittest.TestCase):
    def test_blocks_publication_and_credentials(self):
        for path in ("paper/main.tex", "paper/main.pdf", "output/submission/bundle.zip", "docs/dwave_experimental_proposal.tex", "docs/submission_guide.md", "results/stage_g/refit/run_001/table.tex", ".qiskit/qiskit-ibm.json", ".env"):
            with self.subTest(path=path):
                self.assertTrue(boundary.forbidden(path))

    def test_preserves_code_and_evidence(self):
        for path in ("README.md", "scripts/run_ibm_pilot.py", "results/quantum_heldout/hardware/plan.json", "results/example.qasm", "results/example.qpy", "docs/ibm_run_registry.md"):
            with self.subTest(path=path):
                self.assertFalse(boundary.forbidden(path))
