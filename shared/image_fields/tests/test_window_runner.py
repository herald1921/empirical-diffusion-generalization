import importlib.util
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_windows", ROOT / "eval_windows.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class NearestCopyTests(unittest.TestCase):
    def test_near_copy_is_not_lost_to_distance_cancellation(self):
        # With a large common mean, the matmul distance identity can round both
        # candidates to zero. The exact match is deliberately placed second.
        q = torch.full((1, 6400), 0.75)
        reference = q.repeat(32, 1)
        reference[0] += 1e-4
        reference[2:] += 0.1
        rmse, images = runner.nearest(q, reference, torch.device("cpu"), 32)
        self.assertEqual(float(rmse[0]), 0.0)
        np.testing.assert_array_equal(images[0], q[0].numpy())

    def test_chunked_search_preserves_global_nearest_sample(self):
        reference = torch.tensor([[0., 0.], [1., 1.], [3., 4.]])
        q = torch.tensor([[2.9, 4.1], [.1, .2]])
        rmse, images = runner.nearest(q, reference, torch.device("cpu"), 1)
        np.testing.assert_array_equal(images, reference[[2, 0]].numpy())
        np.testing.assert_allclose(rmse, ((q-reference[[2, 0]])**2).mean(1).sqrt().numpy())

    def test_run_resume_endpoint_windows_and_configuration_guard(self):
        # Exercise actual integration, metrics, serialization and resume; figure
        # generation is tested separately by the rendered end-to-end preview.
        with tempfile.TemporaryDirectory(prefix="nn_windows_test_") as temp:
            out = Path(temp) / "run"
            args = ["eval_windows.py", "--demo", "--out-dir", str(out),
                    "--num-seeds", "2", "--diagnostic-seeds", "1", "--gallery-seeds", "1",
                    "--batch-size", "1", "--solver-steps", "10", "--windows", "0:0.1,0.9:1",
                    "--probe-times", "0,0.1,0.9,0.98"]
            with patch("celeba_nn_stage.window_figures.plot_run", return_value=[]) as plot:
                with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                    runner.main()
                self.assertTrue((out / "COMPLETE").exists())
                batches = list((out / "batches").rglob("*.npz"))
                hashes = {p: runner.file_hash(p) for p in batches}
                with patch.object(sys, "argv", args + ["--resume"]), contextlib.redirect_stdout(io.StringIO()):
                    runner.main()
                self.assertEqual(hashes, {p: runner.file_hash(p) for p in batches})
                self.assertEqual(plot.call_count, 2)
                with patch.object(sys, "argv", args + ["--resume", "--noise-seed", "777"]), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                        runner.main()
                with patch.object(sys, "argv", ["eval_windows.py", "--out-dir", str(out), "--plot-only"]):
                    with patch.object(runner, "demo_fields", side_effect=AssertionError("loaded a model")):
                        runner.main()
                self.assertEqual(plot.call_count, 3)


if __name__ == "__main__":
    unittest.main()
