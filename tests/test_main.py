from argparse import ArgumentTypeError

import pytest

from main import _parse_ranks, build_parser


def test_main_parser_exposes_safe_defaults():
    args = build_parser().parse_args([])

    assert args.scenario in (1, 2)
    assert args.model in {"tsmixer", "nbeats", "nhits"}
    assert args.no_plots is False


def test_parse_ranks_rejects_empty_or_non_positive_values():
    assert _parse_ranks("1, 2,3") == (1, 2, 3)

    with pytest.raises(ArgumentTypeError):
        _parse_ranks("")
    with pytest.raises(ArgumentTypeError):
        _parse_ranks("0,2")
