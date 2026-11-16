import numpy as np
import pandas as pd

from src.data.simulate import (
    SCENARIO_KEYS,
    build_parser,
    fourier_design,
    parse_regime_break,
)


def test_fourier_design_has_expected_shape_and_intercept():
    design = fourier_design(np.arange(12), period=12, n_harmonics=3)

    assert design.shape == (12, 8)
    np.testing.assert_allclose(design[:, 0], 1.0)


def test_parse_regime_break_handles_cli_values():
    assert parse_regime_break("auto") == "auto"
    assert parse_regime_break("none") is None
    assert parse_regime_break("36") == 36


def test_scenario_keys_are_stable():
    assert SCENARIO_KEYS == ("s1", "s2", "s3", "s4", "s5", "s6")


def test_simulation_cli_parser_supports_monte_carlo():
    args = build_parser().parse_args(["--monte_carlo", "--n_mc_seeds", "2", "--no_plot"])

    assert args.monte_carlo is True
    assert args.n_mc_seeds == 2
    assert args.no_plot is True
