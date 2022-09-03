#!/usr/bin/env python3
from dsimplex import simplex
from dsimplex import args
from numpy.testing import assert_almost_equal


def test_integ_one():
    """Integration test for csv one."""
    mock_cli: str = "-m --csv ./tests/equ1.csv --delim ,"
    argparser = args.Argparser()
    argparser.parse(mock_cli.split())
    assert_almost_equal(simplex.parse_equ_csv_loop(argparser)[0], -17.0)


def test_integ_three():
    """Integration test for csv three."""
    mock_cli: str = "-m --csv ./tests/equ3.csv --delim ,"
    argparser = args.Argparser()
    argparser.parse(mock_cli.split())
    assert_almost_equal(simplex.parse_equ_csv_loop(argparser)[0], -16.0)


def test_integ_four():
    """Integration test for csv four."""
    mock_cli: str = "-m --csv ./tests/equ4.csv --delim ,"
    argparser = args.Argparser()
    argparser.parse(mock_cli.split())
    assert_almost_equal(simplex.parse_equ_csv_loop(argparser)[0], -5.4)


def test_integ_five():
    """Integration test for csv five."""
    mock_cli: str = "-m --csv ./tests/equ5.csv --delim ,"
    argparser = args.Argparser()
    argparser.parse(mock_cli.split())
    assert_almost_equal(simplex.parse_equ_csv_loop(argparser)[0], -1.25)


def test_integ_six():
    """Integration test for csv six."""
    mock_cli: str = "-m --csv ./tests/equ6.csv --delim ,"
    argparser = args.Argparser()
    argparser.parse(mock_cli.split())
    assert_almost_equal(simplex.parse_equ_csv_loop(argparser)[0], -6.0)


def test_integ_seven():
    """Integration test for csv six."""
    mock_cli: str = "--csv ./tests/equ7.csv --delim ,"
    argparser = args.Argparser()
    argparser.parse(mock_cli.split())
    assert_almost_equal(simplex.parse_equ_csv_loop(argparser)[0], 600)


def test_integ_eight():
    """Integration test for csv six."""
    mock_cli: str = "--csv ./tests/equ8.csv --delim ,"
    argparser = args.Argparser()
    argparser.parse(mock_cli.split())
    assert_almost_equal(simplex.parse_equ_csv_loop(argparser)[0], 13)


def test_integ_nine():
    """Integration test for csv nine."""
    mock_cli: str = "--csv ./tests/equ9.csv --delim ,"
    argparser = args.Argparser()
    argparser.parse(mock_cli.split())
    assert_almost_equal(simplex.parse_equ_csv_loop(argparser)[0], 400)


def test_integ_multi():
    """Integration test for multiple LP problems with the same equation."""
    mock_cli: str = "-m --csv ./tests/equ_loop.csv --delim ,"
    argparser = args.Argparser()
    argparser.parse(mock_cli.split())
    assert_almost_equal(
        simplex.parse_equ_csv_loop(argparser),
        [
            -1.25,
            -1.25,
            -1.25,
            -1.25,
        ],
    )


def test_integ_multi_diff():
    """Integration test for multiple LP problems with different equation."""
    mock_cli: str = "-m --csv ./tests/equ_loop_diff.csv --delim ,"
    argparser = args.Argparser()
    argparser.parse(mock_cli.split())
    assert_almost_equal(
        simplex.parse_equ_csv_loop(argparser), [-17.0, -6.0, -1.25]
    )
