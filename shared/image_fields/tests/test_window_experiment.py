import math
import sys
from pathlib import Path

import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from celeba_nn_stage.window_experiment import (
    Window,
    build_grid,
    compare_clean,
    integrate_grid,
    run_window,
)


class WindowExperimentTests(unittest.TestCase):
    def test_grid_unions_every_boundary_and_capture_and_deduplicates_collisions(self):
        windows = [Window(0.1 + 0.2, 0.57), Window(0.57, 0.9000000000000001)]
        grid = build_grid(10, windows, [0.3, 0.30000000000000004, 0.44, 1.0])
        assert grid == (0.0, 0.1, 0.2, 0.3, 0.4, 0.44, 0.5, 0.57, 0.6, 0.7, 0.8, 0.9, 1.0)
        assert windows[0].start == 0.3
        assert windows[0].key == Window(0.3, 0.57).key
        assert all(c.isalnum() or c == "_" for c in windows[0].key)

    def test_invalid_times_are_rejected(self):
        for bad in [float("nan"), float("inf"), -0.01, 1.01]:
            with self.assertRaises(ValueError):
                build_grid(4, (), [bad])
            with self.assertRaises(ValueError):
                Window(0, bad)
            with self.assertRaises(ValueError):
                compare_clean(None, None, torch.zeros(1, 2), bad)

    def test_invalid_step_counts_are_rejected(self):
        for steps in [0, -1, 2.5, True]:
            with self.assertRaises(ValueError):
                build_grid(steps)

    def test_degenerate_windows_are_rejected_after_rounding(self):
        for start, end in [(0.3, 0.3), (0.8, 0.7), (0.3, 0.3 + 1e-14)]:
            with self.assertRaises(ValueError):
                Window(start, end)

    def test_midpoint_exact_time_dependent_polynomial_and_endpoint_avoidance(self):
        queries = []

        def field(x, t):
            queries.append(t)
            assert t < 1.0
            return torch.full_like(x, 2 * t + 3)

        x0 = torch.tensor([[2.0, -3.0]], dtype=torch.float64)
        grid = build_grid(7, [Window(0.23, 0.81)], [0.0, 0.42, 1.0])
        final, caps = integrate_grid(field, x0, grid, (0, 0.42, 1))
        torch.testing.assert_close(final, x0 + 4, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(caps[0.42], x0 + 0.42**2 + 3 * 0.42)
        expected_queries = [t for a, b in zip(grid, grid[1:]) for t in (a, (a + b) / 2)]
        assert len(queries) == len(expected_queries)
        assert all(math.isclose(got, expected, abs_tol=1e-15) for got, expected in zip(queries, expected_queries))
        assert caps[0.0].device.type == "cpu"
        with torch.inference_mode():
            final.add_(10)
        torch.testing.assert_close(caps[1.0], x0 + 4)
        torch.testing.assert_close(x0, torch.tensor([[2.0, -3.0]], dtype=torch.float64))

    def test_state_dependent_midpoint_has_second_order_accuracy(self):
        x0 = torch.ones(1, 2, dtype=torch.float64)
        coarse, _ = integrate_grid(lambda x, t: x, x0, build_grid(20))
        fine, _ = integrate_grid(lambda x, t: x, x0, build_grid(40))
        coarse_error = abs(coarse[0, 0].item() - math.e)
        fine_error = abs(fine[0, 0].item() - math.e)
        assert 3.8 < coarse_error / fine_error < 4.1

    def test_window_matches_full_grid_piecewise_solution_and_reverts(self):
        for direction, window in ((d, w) for d in ("nn_emp_nn", "emp_nn_emp") for w in (Window(0.23, 0.81), Window(0.0, 0.6), Window(0.4, 1.0))):
            nn = lambda x, t: 0.2 * x + (1 + t)
            emp = lambda x, t: -0.4 * x + (2 - t)
            baseline, alternate = (nn, emp) if direction == "nn_emp_nn" else (emp, nn)
            captures = (0.0, window.start, window.end, 1.0)
            grid = build_grid(12, [window], captures)
            x0 = torch.tensor([[0.5, -2.0], [3.0, 1.0]], dtype=torch.float64)
            _, base_caps = integrate_grid(baseline, x0, grid, captures)
            calls = []

            def trace(name, fn):
                def field(x, t):
                    calls.append((name, t))
                    return fn(x, t)
                return field

            got, caps = run_window(trace("nn", nn), trace("emp", emp), base_caps[window.start], grid, window, direction, captures)

            def switched(x, t):
                return (alternate if window.start <= t < window.end else baseline)(x, t)

            expected, expected_caps = integrate_grid(switched, x0, grid, captures)
            torch.testing.assert_close(got, expected, rtol=0, atol=0)
            for t, value in caps.items():
                torch.testing.assert_close(value, expected_caps[t], rtol=0, atol=0)
            assert all(t >= window.start and t < 1.0 for _, t in calls)
            base_name = "nn" if direction == "nn_emp_nn" else "emp"
            assert all(name != base_name if t < window.end else name == base_name for name, t in calls)

    def test_shared_prefix_continuation_equals_uninterrupted_baseline_exactly(self):
        field = lambda x, t: x * (0.7 + t) - 0.2
        window = Window(0.37, 0.82)
        grid = build_grid(20, [window], [window.start, 1.0])
        x0 = torch.tensor([[0.3, -0.6]], dtype=torch.float64)
        expected, captures = integrate_grid(field, x0, grid, [window.start, 1])
        continued, _ = run_window(field, field, captures[window.start], grid, window, "nn_emp_nn", [1])
        torch.testing.assert_close(continued, expected, rtol=0, atol=0)

    def test_missing_nodes_bad_grids_and_nonfinite_states_fail(self):
        x = torch.zeros(1, 2)
        field = lambda x, t: torch.ones_like(x)
        with self.assertRaisesRegex(ValueError, "capture"):
            integrate_grid(field, x, (0, 0.5, 1), (0.4,))
        with self.assertRaisesRegex(ValueError, "boundaries"):
            run_window(field, field, x, (0, 0.5, 1), Window(0.3, 0.5), "nn_emp_nn")
        with self.assertRaisesRegex(ValueError, "direction"):
            run_window(field, field, x, (0, 0.5, 1), Window(0, 0.5), "other")
        for grid in [(), (0, 0.5, 0.5, 1), (0, 0.8, 0.7, 1), (0, float("nan"), 1)]:
            with self.assertRaises(ValueError):
                integrate_grid(field, x, grid)
        with self.assertRaises(FloatingPointError):
            integrate_grid(lambda x, t: x + float("nan"), x, (0, 1))

    def test_clean_comparison_same_raw_state_relation_metrics_and_no_clipping(self):
        x = torch.tensor([[4.0, -3.0], [7.0, 2.0]], dtype=torch.float64)
        t = 0.75
        mean = torch.tensor([[2.0, -2.0], [3.0, 4.0]], dtype=torch.float64)
        v_emp = (mean - x) / (1 - t)
        v_nn = v_emp + torch.tensor([[4.0, -4.0], [8.0, 8.0]])
        queries = []

        def nn(state, time):
            queries.append((state.data_ptr(), state.clone(), time))
            return v_nn

        def emp(state, time):
            queries.append((state.data_ptr(), state.clone(), time))
            return v_emp, {"mean": mean, "ess": torch.tensor([4.0, 2.0]), "top_probability": torch.tensor([0.3, 0.8])}

        result = compare_clean(nn, emp, x, t)
        assert len(queries) == 2 and queries[0][0] == queries[1][0] == x.data_ptr()
        assert queries[0][2] == queries[1][2] == t
        torch.testing.assert_close(queries[0][1], queries[1][1])
        torch.testing.assert_close(result["clean_nn"], x + (1 - t) * v_nn)
        torch.testing.assert_close(result["clean_emp"], mean)
        torch.testing.assert_close(result["difference"], (1 - t) * (v_nn - v_emp))
        torch.testing.assert_close(result["denoiser_rmse"], (1 - t) * result["velocity_rmse"])
        torch.testing.assert_close(result["denoiser_rmse"], torch.tensor([1.0, 2.0], dtype=x.dtype))
        expected_cosine = (v_nn * v_emp).sum(1) / (v_nn.norm(dim=1) * v_emp.norm(dim=1))
        torch.testing.assert_close(result["cosine"], expected_cosine)
        assert result["clean_nn"].abs().max() > 1
        assert all(value.device.type == "cpu" for value in result.values())
        x.zero_()
        assert result["state"].abs().max() == 7
        with self.assertRaises(ValueError):
            compare_clean(nn, emp, x, 1.0)


if __name__ == "__main__":
    unittest.main()
