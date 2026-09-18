import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import tempfile
import unittest
from unittest.mock import patch
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/repeat_ibm_component.py"
spec = importlib.util.spec_from_file_location("repeat_component", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ComponentRepeatTests(unittest.TestCase):
    def test_date_guard_rejects_same_utc_day(self):
        self.assertFalse(module.independent_date("2026-09-14T16:01:16Z", datetime(2026, 9, 14, 23, tzinfo=timezone.utc)))
        self.assertTrue(module.independent_date("2026-09-14T16:01:16Z", datetime(2026, 9, 18, tzinfo=timezone.utc)))

    def test_submission_requires_authorization_before_access(self):
        with patch.object(module, "account", side_effect=AssertionError("Account access before authorization")):
            with self.assertRaisesRegex(RuntimeError, "authorization"):
                module.run(Path("unused"), False)

    def test_prepare_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "unused"):
                module.prepare(Path(folder))
