#!/usr/bin/env python
# ./simplex.py -e ./equ.txt -v -s s -m
"""python simplex implementation"""

import argparse
import ast
import dataclasses
import logging
import sys
import typing
import numba as nb  # type:ignore
import numpy as np


class Argparser:  # pylint: disable=too-few-public-methods
    """Argparser class."""

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
    operand: str
    rhs: float


@dataclasses.dataclass
class RoundResult:
    """The class will hold each round'd results for verbose display."""

    curr_basis: np.ndarray
    curr_x: np.ndarray
    curr_cost: float


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
    """Walks over a python expression and extracts the
    operands, identifiers and literals."""
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
                equ.operand = "<="
            elif isinstance(ast_node.ops[0], ast.GtE):
                if debug:
                    print(">=", ast_node.comparators[0].value)
                equ.rhs = float(ast_node.comparators[0].value)
                equ.operand = ">="
            elif isinstance(ast_node[0], ast.Gt):
                if debug:
                    print(">", ast_node.comparators[0].value)
                equ.rhs = float(ast_node.comparators[0].value)
                equ.operand = ">"
            elif isinstance(ast_node[0], ast.Lt):
                if debug:
                    print("<", ast_node.comparators[0].value)
                equ.rhs = float(ast_node.comparators[0].value)
                equ.operand = "<"
            elif isinstance(ast_node[0], ast.Eq):
                if debug:
                    print("=", ast_node.comparators[0].value)
                equ.rhs = float(ast_node.comparators[0].value)
                equ.operand = "="
            else:
                logging.fatal(
                    "encountered unexpected Compare node, {}, in ast %s",
                    type(ast_node.comparators[0]),
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
                    "encountered unexpected unary operand in ast, %s.",
                    type(ast_node.op),
                )
                sys.exit(1)
        case _:
            logging.fatal(
                "encountered unexpected node, %s, in ast", type(ast_node)
            )
            sys.exit(1)

    return None, equ


def parse_equations(equ_file: str, debug: bool) -> typing.List[Equation]:
    """Parses the equations as pyithon expressions."""
    equs = []
    with open(equ_file, encoding="utf-8") as equ_expr:
        equs_unparsed = equ_expr.readlines()
        for equ_unparsed in equs_unparsed:
            equ_parsed = ast.parse(equ_unparsed)
            if debug:
                print(type(equ_parsed.body[0]))
                print(ast.dump(equ_parsed.body[0], indent=4))
            equ = Equation({}, "", 0.0)
            _, res = expr_visitor(
                equ_parsed.body[0], equ_parsed.body[0], equ, debug
            )
            equs.append(res)

    if debug:
        for equ in equs:
            print(equ)

    return equs


def build_abc(
    equs: typing.List[Equation], verbose: bool, slack: str
) -> typing.Tuple[
    np.ndarray[typing.Any, np.dtype[np.float32]],
    np.ndarray[typing.Any, np.dtype[np.float32]],
    np.ndarray[typing.Any, np.dtype[np.float32]],
    typing.List[str],
]:
    """Build the A matrix, adding the slack variable along the way."""
    var_list: typing.Dict[str, bool] = {}
    var_sorted_list: typing.List[str] = []
    slack_counter: int = 1

    equ_b = []
    binds = []
    # Add the slack variables
    for i, equ in enumerate(equs):
        if len(equ.vars_mults) == 1:
            binds.append(equ)
            continue

        if equ.operand in ("<=", "<"):
            if len(equ.vars_mults) > 1:
                equ.vars_mults[slack + repr(slack_counter)] = 1
                slack_counter += 1
            if equ.operand == "<=":
                binds.append(Equation({slack + repr(i): 1}, ">=", 0.0))
            else:
                binds.append(Equation({slack + repr(i): 1}, ">", 0.0))
        elif equ.operand in (">=", ">"):
            if len(equ.vars_mults) > 1:
                equ.vars_mults[slack + repr(slack_counter)] = -1
                slack_counter += 1
            if equ.operand == ">=":
                binds.append(Equation({slack + repr(i): 1}, ">=", 0.0))
            else:
                binds.append(Equation({slack + repr(i): 1}, ">", 0.0))
        elif equ.operand == "=":
            pass
        elif equ.operand == "":
            equ_b.append(equ)
        else:
            logging.fatal("found unexpected operand, %s", equ.operand)
            sys.exit(1)

        for key, _ in equ.vars_mults.items():
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
        for equ in equs:
            print(equ)
        print("binds:")
        for bind in binds:
            print(bind)
        print(var_list)

    m: int = len(equs)
    n: int = len(var_list)

    if verbose:
        print(f"m: {m}, n: {n}")

    var_names: typing.List = []
    for key, _ in var_list.items():
        var_names.append(key)
    var_names.sort()

    A: np.ndarray = np.ndarray((m, n), dtype=np.float32)
    b: np.ndarray = np.ndarray((m, 1), dtype=np.float32)
    C: np.ndarray = np.ndarray((1, n), dtype=np.float32)
    # build A
    for i in range(0, m):
        for j in range(0, n):
            if var_names[j] in equs[i].vars_mults:
                A[i][j] = equs[i].vars_mults[var_names[j]]
            else:
                A[i][j] = 0

    # build b
    for i in range(0, m):
        b[i, 0] = equs[i].rhs

    # Add the slack variables to the cost function
    for i in range(1, slack_counter):
        equ_b[0].vars_mults[slack + repr(i)] = 0
    # build C
    for i, (_, v) in enumerate(iter(sorted(equ_b[0].vars_mults.items()))):
        C[0, i] = v

    if verbose:
        print("A:\n", A)
        print("b:\n", b)
        print("C:\n", C)

    for _, v in enumerate(iter(sorted(var_list))):
        var_sorted_list.append(v)
    if verbose:
        print("var_list: ", var_sorted_list)
    return A, b, C, var_sorted_list


@nb.jit(nopython=True, cache=True)
def find_identity(
    A: np.ndarray[typing.Any, np.dtype[np.float32]],
) -> typing.Tuple[bool, typing.Optional[typing.Dict[int, int]]]:
    """Tries to find one m*m identity matrix
    inside A, returns one that it finds."""
    ones_count: int = 0
    last_one_row: int = -1
    ones: int = 0
    cancelled: bool = False
    m: int = A.shape[0]
    n: int = A.shape[1]
    col_list: typing.Dict[int, int] = {}

    for j in range(0, n):
        for i in range(0, m):
            if A[i][j] != 1 and A[i][j] != 0:
                cancelled = True
                break
            if A[i][j] == 1:
                ones_count += 1
                last_one_row = i + 1
        if cancelled:
            cancelled = False
            continue
        if ones_count == 1:
            ones += last_one_row
            col_list[last_one_row] = j
        ones_count = 0
        last_one_row = 0

    print("ones:", ones)
    if ones == (m * (m + 1)) / 2:
        return True, col_list
    return False, None


def find_basis(
    A, b: np.ndarray[typing.Any, np.dtype[np.float32]]
) -> typing.Tuple[
    bool, np.ndarray[typing.Any, np.dtype[np.float32]], typing.List[int]
]:
    """Find a basis for A."""
    m: int = A.shape[0]
    B: np.ndarray = np.zeros((m, m), dtype=np.float32)

    has_identity, col_list = find_identity(A)
    print("col_list:", col_list)
    col_list_list: typing.List[int] = []
    for _, v in col_list.items():
        col_list_list.append(v)
    if has_identity:
        B = np.identity(m, dtype=np.float32)
        # for _, col in col_list.items():
        #     B[col][col] = 1
        return has_identity, B, col_list_list
    else:
        # two phase
        # big M
        pass

    return False, B, col_list_list


@nb.jit(nopython=True, cache=True)
def invert_matrix(M: np.ndarray) -> np.ndarray:
    """inverts a square matrix"""
    return np.linalg.inv(M)


@nb.jit(nopython=True, cache=True, parallel=True)
def get_costs(w, A, C: np.ndarray) -> np.ndarray:
    """calculates Z_j - C_j"""
    return np.dot(w, A) - C


# @nb.jit(nopython=True, cache=True)
def calculate_objective(
    basic_var_column_list: typing.List,
    A,
    B,
    C: np.ndarray[typing.Any, np.dtype[np.float32]],
    basis_is_identity: bool,
    verbose: bool,
) -> typing.Tuple[
    np.ndarray[typing.Any, np.dtype[np.float32]],
    np.ndarray[typing.Any, np.dtype[np.float32]],
]:
    """Calculate C_b*B^-1*b."""
    m: int = len(basic_var_column_list)
    C_b = np.zeros((1, m), dtype=np.float32)
    objectives = np.zeros((1, m), dtype=np.float32)

    if basis_is_identity:
        B_inv = B
    else:
        B_inv = invert_matrix(B)
    print("B_inv:\n", B_inv)

    print("basic_var_column_list: ", basic_var_column_list)
    for i, v in enumerate(basic_var_column_list):
        C_b[0, i] = C[0, v]

    if verbose:
        print("C_b:\n", C_b)

    w = np.dot(C_b, B_inv)
    if verbose:
        print("w:\n", w)
    # for j in range(0, A.shape[1]):
    #     print("zj_cj:", np.matmul(w, A[:, j : j + 1]) - C[0, j])
    objectives = get_costs(w, A, C)
    if verbose:
        print("zj_cj:\n", objectives)

    return B_inv, objectives


def get_non_negative_min(
    M,
    y_k: np.ndarray[typing.Any, np.dtype[np.float32]],
) -> typing.Tuple[float, int]:
    """Get the index of the leaving variable in the basis var list.
    This needs to get translated into the actual index of the variable."""
    n = M.shape[0]
    minimum: float = 1e9
    minimum_index: int = 0
    for i in range(0, n):
        if y_k[i, 0] > 0:
            if M[i, 0] < minimum:
                minimum = M[i, 0]
                minimum_index = i

    # TODO
    # now we have the index of the leaving var in the basis
    # we need to change this to the index of the leaving var

    return minimum, minimum_index


def determine_leaving(
    k: int,
    A,
    B,
    B_inv,
    b,
    y_k: np.ndarray[typing.Any, np.dtype[np.float32]],
    basis_col_list: typing.List[int],
    basis_is_identity,
    verbose: bool,
) -> typing.Tuple[int, np.ndarray[typing.Any, np.dtype[np.float32]]]:
    """Determines the entering and leaving variables for each round."""
    r: int = 0
    if basis_is_identity:
        b_bar = b
    else:
        b_bar = np.dot(B_inv, b)
    if verbose:
        print("b_bar:\n", b_bar)

    b_bar_plus = b_bar[b_bar > 0]
    b_bar_div_y = np.divide(b_bar_plus[:, None], y_k)
    print("b_bar_div_y:\n", b_bar_div_y)
    _, r = get_non_negative_min(b_bar_div_y, y_k)
    rr = basis_col_list[r]
    if verbose:
        print("b_bar/y_k:\n", b_bar_div_y)
        print("r: ", rr)

    B[:, r : r + 1] = A[:, k : k + 1]

    return rr, B


def get_k(
    M: np.ndarray[typing.Any, np.dtype[np.float32]],
    basis_col_list: typing.List[int],
) -> typing.Tuple[int, float]:
    """get the index of the entering variable."""
    n = M.shape[1]
    maximum: float = -1e9
    maximum_index: int = -1
    # print("k basis_col_list:\n", basis_col_list)
    # print("k objectives:\n", M)
    for i in range(0, n):
        if i not in basis_col_list:
            # print("k M[0,i]:\n", M[0, i])
            if M[0, i] > maximum:
                maximum = M[0, i]
                maximum_index = i

    return maximum_index, maximum


def solve_normal_simplex(
    A,
    b,
    C: np.ndarray[typing.Any, np.dtype[np.float32]],
    var_sorted_list: typing.List[str],
    argparser: Argparser,
) -> None:
    """Solve using the normal simplex method."""
    verbose: bool = argparser.args.verbose
    basis_is_identity, B, basis_col_list = find_basis(A, b)
    round_count: int = 0
    while True:
        round_count += 1
        B_inv, objectives = calculate_objective(
            basis_col_list, A, B, C, basis_is_identity, verbose
        )
        # k_index = np.where(objectives == np.max(objectives))
        k, _ = get_k(objectives, basis_col_list)
        # k = k_index[-1][-1]
        if verbose:
            print("k: ", k)
        max_zj_cj = objectives[0, k]
        if argparser.args.min:
            if max_zj_cj < 0:
                # we are done
                print("optimal min is:", np.sum(objectives))
                break
        else:
            if max_zj_cj > 0:
                # we aredone
                print("optimal max is :", np.sum(objectives), k)
                break

        y_k = np.dot(B_inv, A[:, k : k + 1])
        print("y_k:\n", y_k)
        if np.all(np.less_equal(y_k, 0)):
            # we are done
            print("unbounded optimal value.")
            break
        print(y_k)
        r, B = determine_leaving(
            k, A, B, B_inv, b, y_k, basis_col_list, basis_is_identity, verbose
        )
        basis_is_identity = False

        print("basis_col_list 1: ", basis_col_list)
        leaving_col: int = 0
        for i, basis in enumerate(basis_col_list):
            if basis == r:
                leaving_col = i
        basis_col_list[leaving_col] = k
        print("basis_col_list 2: ", basis_col_list)

        if verbose:
            print("B:\n", B)
        print("-------------------------------------------------")
        if round_count > 10:
            print("too many iterations.")
            break


def main() -> None:
    """The entry point for the module."""
    argparser = Argparser()
    verbose = argparser.args.verbose
    equs = parse_equations(argparser.args.equs, argparser.args.debug)
    A, b, C, var_sorted_list = build_abc(equs, verbose, argparser.args.slack)
    solve_normal_simplex(A, b, C, var_sorted_list, argparser)


if __name__ == "__main__":
    main()
