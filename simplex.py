#!/usr/bin/env python
"""python simplex implementation"""

import argparse
import ast
import dataclasses
import logging

# import numba
# import numpy
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


@dataclasses.dataclass
class Equation:
    vars_mults: typing.TypedDict("VarMult", {"var": str, "mult": float})
    op: str
    rhs: float

    def autofill_one_mults(self):
        for k, v in self.vars_mults:
            if v is None:
                v = 1

    def dump_var_mults(self):
        for k, v in self.vars_mults:
            print(k, v)


def walk_expr(
    node: typing.Any, equ: Equation, debug: bool
) -> (typing.Any, Equation):
    """walks over a python expression and extracts the
    operand,identifiers and literals"""
    if node is None:
        return None, equ

    if isinstance(node, ast.Expr):
        ast_node = node.value
    else:
        ast_node = node

    match type(ast_node):
        case ast.BinOp:
            if isinstance(ast_node.op, ast.Mult):
                if debug:
                    print("*")
                if ast_node.right.id not in equ.vars_mults:
                    equ.vars_mults[ast_node.right.id] = ast_node.left.value
            elif isinstance(ast_node.op, ast.Sub):
                if debug:
                    print("-")
                if isinstance(ast_node.right, ast.BinOp):
                    if isinstance(ast_node.right.op, ast.Mult):
                        if ast_node.right.right.id not in equ.vars_mults:
                            equ.vars_mults[
                                ast_node.right.right.id
                            ] = -ast_node.right.left.value
                elif isinstance(ast_node.right, ast.Name):
                    if ast_node.right.id not in equ.vars_mults:
                        equ.vars_mults[ast_node.right.id] = -1
            elif isinstance(ast_node.op, ast.Add):
                if debug:
                    print("+")
            walk_expr(ast_node.left, equ, debug)
            walk_expr(ast_node.right, equ, debug)
        case ast.Compare:
            if isinstance(ast_node.ops[0], ast.LtE):
                if debug:
                    print("<=", ast_node.comparators[0].value)
                equ.rhs = float(ast_node.comparators[0].value)
                equ.op = "<="
            elif isinstance(ast_node.ops[0], ast.GtE):
                if debug:
                    print(">=", ast_node.comparators[0].value)
                equ.rhs = float(ast_node.comparators[0].value)
                equ.op = ">="
            elif isinstance(ast_node[0], ast.Gt):
                if debug:
                    print(">", ast_node.comparators[0].value)
                equ.rhs = float(ast_node.comparators[0].value)
                equ.op = ">"
            elif isinstance(ast_node[0], ast.Lt):
                if debug:
                    print("<", ast_node.comparators[0].value)
                equ.rhs = float(ast_node.comparators[0].value)
                equ.op = "<"
            elif isinstance(ast_node[0], ast.Eq):
                if debug:
                    print("=", ast_node.comparators[0].value)
                equ.rhs = float(ast_node.comparators[0].value)
                equ.op = "="
            else:
                logging.fatal(
                    "encountered unexpected Compare node, {}, in ast".format(
                        type(ast_node.comparators[0])
                    )
                )
            walk_expr(ast_node.left, equ, debug)
        case ast.Name:
            if debug:
                print(ast_node.id)
            if ast_node.id not in equ.vars_mults:
                equ.vars_mults[ast_node.id] = 1
        case ast.Constant:
            if debug:
                print(ast_node.value)
        case ast.Expr:
            walk_expr(ast_node.value, equ, debug)
        case ast.UnaryOp:
            if isinstance(ast_node.op, ast.USub):
                if debug:
                    print("-", ast_node.operand.id)
                if ast_node.operand.id not in equ.vars_mults:
                    equ.vars_mults[ast_node.operand.id] = -1
            elif isinstance(ast_node.op, ast.UAdd):
                # Unary add operands are meaningless
                pass
            else:
                logging.fatal(
                    "encountered unexpected unary operand in ast, {}.".format(
                        type(ast_node.op)
                    )
                )
        case _:
            logging.fatal(
                "encountered unexpected node,{} , in ast".format(
                    type(ast_node)
                )
            )

    return None, equ


def parse_equations(equ_file: str, debug: bool):
    """parses the equations as pyithon expressions"""
    equs = []
    with open(equ_file, encoding="utf-8") as equ_expr:
        equs_unparsed = equ_expr.readlines()
        for equ in equs_unparsed:
            equ_parsed = ast.parse(equ)
            if debug:
                print(type(equ_parsed.body[0]))
                print(ast.dump(equ_parsed.body[0], indent=4))
            equ = Equation(dict(), "", "")
            _, res = walk_expr(equ_parsed.body[0], equ, debug)
            equs.append(res)

    if debug:
        for equ in equs:
            print("vars_mults:", equ.vars_mults)
            print("op: ", equ.op)
            print("rhs: ", equ.rhs)
            print()


def main() -> None:
    """the entry point for the module"""
    argparser = Argparser()
    parse_equations(argparser.args.equs, argparser.args.debug)


if __name__ == "__main__":
    main()
