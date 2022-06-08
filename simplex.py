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
    # postfix
    q: list
    qq: list

    def autofill_one_mults(self):
        for k, v in self.vars_mults:
            if v is None:
                v = 1

    def dump(self):
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
            walk_expr(ast_node.left, equ, debug)
            walk_expr(ast_node.right, equ, debug)
            if isinstance(ast_node.op, ast.Mult):
                if debug:
                    print("*")
                equ.q.append("*")
                equ.qq.append(ast_node.op)
            elif isinstance(ast_node.op, ast.Sub):
                if debug:
                    print("-")
                equ.q.append("-")
                equ.qq.append(ast_node.op)
            elif isinstance(ast_node.op, ast.Add):
                if debug:
                    print("+")
                equ.q.append("+")
                equ.qq.append(ast_node.op)
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
            equ.q.append(ast_node.id)
            equ.qq.append(ast_node)
        case ast.Constant:
            if debug:
                print(ast_node.value)
            equ.q.append(ast_node.value)
            equ.qq.append(ast_node)
        case ast.Expr:
            walk_expr(ast_node.value, equ, debug)
        case ast.UnaryOp:
            if isinstance(ast_node, ast.UnaryOp):
                if debug:
                    print("-", ast_node.operand.id)
                equ.q.append("-")
                equ.q.append(ast_node.operand.id)
                equ.qq.append(ast_node)
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
            equ = Equation([], "", "", [], [])
            _, res = walk_expr(equ_parsed.body[0], equ, debug)
            # print("vars_mults:", res.vars_mults)
            # print("op: ", res.op)
            # print("rhs: ", res.rhs)
            # print("q: ", res.q)
            # print()
            equs.append(res)

    for equ in equs:
        print("vars_mults:", equ.vars_mults)
        print("op: ", equ.op)
        print("rhs: ", equ.rhs)
        print("q: ", equ.q)
        print("qq: ", equ.qq)
        print()


def main() -> None:
    """the entry point for the module"""
    argparser = Argparser()
    parse_equations(argparser.args.equs, argparser.args.debug)


if __name__ == "__main__":
    main()
