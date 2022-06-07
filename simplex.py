#!/usr/bin/env python
"""python simplex implementation"""

import argparse
import ast
import logging
import numba
import numpy
import typing


class Argparser:  # pylint: disable=too-few-public-methods
    """argparser class"""

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--equs",
            "-e",
            type=str,
            help="the file containing the equations",
            default=False,
        )
        parser.add_argument(
            "--min",
            "-m",
            action="store_true",
            help="determines whether its a minimization problem."
            "if not, its a maximization problem",
            default=False,
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="whether to print output verbosely",
            default=False,
        )
        parser.add_argument(
            "--debug",
            "-d",
            action="store_true",
            help="whether to print debug info",
            default=False,
        )
        self.args = parser.parse_args()


def walk_expr(node: typing.Any) -> None:
    """walks over a python expression and extracts the
    operand,identifiers and literals"""
    if node is None:
        return None

    if isinstance(node, ast.Expr):
        ast_node = node.value
    else:
        ast_node = node

    match type(ast_node):
        case ast.BinOp:
            walk_expr(ast_node.left)
            walk_expr(ast_node.right)
            if isinstance(ast_node.op, ast.Mult):
                print("*")
            elif isinstance(ast_node.op, ast.Sub):
                print("-")
            elif isinstance(ast_node.op, ast.Add):
                print("+")
        case ast.Compare:
            if isinstance(ast_node.ops[0], ast.LtE):
                print("<=", ast_node.comparators[0].value)
            elif isinstance(ast_node.ops[0], ast.GtE):
                print(">=", ast_node.comparators[0].value)
            walk_expr(ast_node.left)
        case ast.Name:
            print(ast_node.id)
        case ast.Constant:
            print(ast_node.value)
        case ast.Expr:
            walk_expr(ast_node.value)
        case _:
            logging.fatal("encountered unexpected node in ast")

    return None


def parse_equations(equ_file: str, debug: bool):
    """parses the equations as pyithon expressions"""
    with open(equ_file, encoding="utf-8") as equ_expr:
        equs_unparsed = equ_expr.readlines()
    for equ in equs_unparsed:
        equ_parsed = ast.parse(equ)
        if debug:
            print(type(equ_parsed.body[0]))
            print(ast.dump(equ_parsed.body[0], indent=4))
        walk_expr(equ_parsed.body[0])


def main() -> None:
    """the entry point for the module"""
    argparser = Argparser()
    parse_equations(argparser.args.equs, argparser.args.debug)


if __name__ == "__main__":
    main()
