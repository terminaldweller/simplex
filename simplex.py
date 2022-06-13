#!/usr/bin/env python
# ./simplex.py -e ./equ.txt -v -s s -m
"""python simplex implementation"""

import argparse
import ast
import dataclasses
import logging
import numba as nb  # type:ignore
import numpy as np
import sys
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
            "--slack",
            "-s",
            type=str,
            help="slack variable base name, names are creted"
            "by adding a number to the string",
            default="xx",
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
        parser.add_argument(
            "--numba",
            "-n",
            action="store_true",
            help="whether to print debug info",
            default=False,
        )
        self.args = parser.parse_args()


@dataclasses.dataclass
class Equation:
    """Equation class. holds the variables, the operand and binding value."""

    vars_mults: typing.Dict
    op: str
    rhs: float


@typing.no_type_check
def get_parent_node(
    node: typing.Any, root: typing.Any
) -> typing.Optional[typing.Any]:
    """Get the parent of a node in the python AST. very inefficient."""
    for subnode in ast.walk(root):
        for child in ast.iter_child_nodes(subnode):
            if child == node:
                return subnode
    return None


@typing.no_type_check
def expr_visitor(
    root: typing.Any, node: typing.Any, equ: Equation, debug: bool
) -> typing.Tuple[typing.Any, Equation]:
    """walks over a python expression and extracts the
    operands, identifiers and literals"""
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
                    if isinstance(ast_node.left, ast.UnaryOp):
                        expr_visitor(root, ast_node.left, equ, debug)
                    else:
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
            expr_visitor(root, ast_node.left, equ, debug)
            expr_visitor(root, ast_node.right, equ, debug)
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
                sys.exit(1)
            expr_visitor(root, ast_node.left, equ, debug)
        case ast.Name:
            if debug:
                print(ast_node.id)
            if ast_node.id not in equ.vars_mults:
                equ.vars_mults[ast_node.id] = 1
        case ast.Constant:
            # We don't have anything to do here
            if debug:
                print(ast_node.value)
        case ast.Expr:
            expr_visitor(root, ast_node.value, equ, debug)
        case ast.UnaryOp:
            if isinstance(ast_node.op, ast.USub):
                if isinstance(ast_node.operand, ast.Constant):
                    parent_node = get_parent_node(ast_node, root)
                    if parent_node is not None:
                        equ.vars_mults[parent_node.right.id] = (
                            -1 * ast_node.operand.value
                        )
                else:
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
                sys.exit(1)
        case _:
            logging.fatal(
                "encountered unexpected node,{} , in ast".format(
                    type(ast_node)
                )
            )
            sys.exit(1)

    return None, equ


def parse_equations(equ_file: str, debug: bool) -> typing.List[Equation]:
    """parses the equations as pyithon expressions"""
    equs = []
    with open(equ_file, encoding="utf-8") as equ_expr:
        equs_unparsed = equ_expr.readlines()
        for equ_unparsed in equs_unparsed:
            equ_parsed = ast.parse(equ_unparsed)
            if debug:
                print(type(equ_parsed.body[0]))
                print(ast.dump(equ_parsed.body[0], indent=4))
            equ = Equation(dict(), "", 0.0)
            _, res = expr_visitor(
                equ_parsed.body[0], equ_parsed.body[0], equ, debug
            )
            equs.append(res)

    if debug:
        [print(equ) for equ in equs]

    return equs


def build_A(
    equs: typing.List[Equation], verbose: bool, slack: str
) -> typing.Tuple[
    np.ndarray[typing.Any, np.dtype[np.float32]],
    np.ndarray[typing.Any, np.dtype[np.float32]],
]:
    """build the A matrix, adding the slack variable along the way"""
    var_list = dict()
    slack_counter: int = 1

    equ_b = []
    binds = []
    for i, equ in enumerate(equs):
        if len(equ.vars_mults) == 1:
            binds.append(equ)
            continue

        if equ.op == "<=" or equ.op == "<":
            if len(equ.vars_mults) > 1:
                equ.vars_mults[slack + repr(slack_counter)] = 1
                slack_counter += 1
            if equ.op == "<=":
                binds.append(Equation({slack + repr(i): 1}, ">=", 0.0))
            else:
                binds.append(Equation({slack + repr(i): 1}, ">", 0.0))
        elif equ.op == ">=" or equ.op == ">":
            if len(equ.vars_mults) > 1:
                equ.vars_mults[slack + repr(slack_counter)] = -1
                slack_counter += 1
            if equ.op == ">=":
                binds.append(Equation({slack + repr(i): 1}, ">=", 0.0))
            else:
                binds.append(Equation({slack + repr(i): 1}, ">", 0.0))
        elif equ.op == "=":
            pass
        elif equ.op == "":
            equ_b.append(equ)
        else:
            logging.fatal("found unexpected operand, {}".format(equ.op))
            sys.exit(1)

        for key, val in equ.vars_mults.items():
            var_list[key] = True

    equs.remove(equ_b[0])
    for bind in binds:
        if bind in equs:
            equs.remove(bind)

    if len(equ_b) == 0:
        logging.fatal("did not provide a cost equation.")
        sys.exit(1)

    if verbose:
        print("cost:")
        print(equ_b[0])
        print("equations:")
        [print(equ) for equ in equs]
        print("binds:")
        [print(bind) for bind in binds]
        print(var_list)

    m: int = len(equs)
    n: int = len(var_list)

    if verbose:
        print("m: {}, n: {}".format(m, n))

    var_names: typing.List = []
    for key, _ in var_list.items():
        var_names.append(key)
    var_names.sort()

    A: np.ndarray = np.ndarray((m, n), dtype=np.float32)
    b: np.ndarray = np.ndarray((m, 1), dtype=np.float32)
    for i in range(0, m):
        for j in range(0, n):
            if var_names[j] in equs[i].vars_mults:
                A[i][j] = equs[i].vars_mults[var_names[j]]
            else:
                A[i][j] = 0
            pass

    for i in range(0, m):
        b[i] = equs[i].rhs

    if verbose:
        print(A)
        print(b)

    return A, b


@nb.jit(nopython=True, cache=True)
def find_identity(
    A: np.ndarray[typing.Any, np.dtype[np.float32]],
) -> typing.Tuple[bool, typing.Optional[typing.Dict[int, int]]]:
    """Tries to find one m*m identity matrix
    inside A, returns one that it finds"""
    ones_count: int = 0
    last_one_row: int = -1
    ones: int = 0
    cancelled: bool = False
    m: int = A.shape[0]
    n: int = A.shape[1]
    col_list: typing.Dict[int, int] = dict()

    for j in range(0, n):
        for i in range(0, m):
            if A[i][j] != 1 and A[i][j] != 0:
                cancelled = True
                continue
            elif A[i][j] == 1:
                ones_count += 1
                last_one_row = i + 1
        if cancelled:
            cancelled = False
            continue
        if ones_count == 1:
            ones += last_one_row
            col_list[last_one_row] = j
        ones_count = 0

    if ones == (m * (m + 1)) / 2:
        return True, col_list
    else:
        return False, None


def find_basis(
    A, b: np.ndarray[typing.Any, np.dtype[np.float32]]
) -> np.ndarray[typing.Any, np.dtype[np.float32]]:
    m: int = A.shape[0]
    B: np.ndarray = np.ndarray((m, m), dtype=np.float32)

    has_identity, col_list = find_identity(A)
    if has_identity:
        print(col_list)
        return B

    return B


def main() -> None:
    """the entry point for the module"""
    argparser = Argparser()
    equs = parse_equations(argparser.args.equs, argparser.args.debug)
    A, b = build_A(equs, argparser.args.verbose, argparser.args.slack)
    B = find_basis(A, b)


if __name__ == "__main__":
    main()
