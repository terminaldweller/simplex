"""python simplex implementation"""

import argparse
import ast
import dataclasses
import logging
import sys
import typing

# import numba as nb  # type:ignore
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
            default="s",
        )
        parser.add_argument(
            "--aux",
            "-a",
            type=str,
            help="aux variable base name, names are creted"
            "by adding a number to the string",
            default="xa",
        )
        parser.add_argument(
            "--iter",
            "-i",
            type=int,
            help="maximum number of iterations",
            default=50,
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
        # TODO- not being used right now
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
class LP:
    """This class holds the information for an LP problem"""

    equs: typing.Optional[typing.List[Equation]]
    cost_equ: typing.Optional[Equation]
    binds: typing.Optional[typing.List[Equation]]
    A: typing.Optional[np.ndarray[typing.Any, np.dtype[np.float32]]]
    b: typing.Optional[np.ndarray[typing.Any, np.dtype[np.float32]]]
    B: typing.Optional[np.ndarray[typing.Any, np.dtype[np.float32]]]
    B_inv: typing.Optional[np.ndarray[typing.Any, np.dtype[np.float32]]]
    C: typing.Optional[np.ndarray[typing.Any, np.dtype[np.float32]]]
    y_k: typing.Optional[np.ndarray[typing.Any, np.dtype[np.float32]]]
    basis_column_list: typing.Optional[typing.List[int]]
    sorted_var_list: typing.Optional[typing.List[str]]
    is_minimum: typing.Optional[bool]


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
            elif isinstance(ast_node.ops[0], ast.Gt):
                if debug:
                    print(">", ast_node.comparators[0].value)
                equ.rhs = float(ast_node.comparators[0].value)
                equ.operand = ">"
            elif isinstance(ast_node.ops[0], ast.Lt):
                if debug:
                    print("<", ast_node.comparators[0].value)
                equ.rhs = float(ast_node.comparators[0].value)
                equ.operand = "<"
            elif isinstance(ast_node.ops[0], ast.Eq):
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
            # we check to make sure we are not visiting comments which have
            # an empty AST since we are parsing line by line
            if len(equ_parsed.body) > 0:
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


def add_slack_vars(
    equs: typing.List[Equation], slack: str, verbose: bool
) -> typing.Tuple[
    typing.List[Equation],
    Equation,
    typing.List[Equation],
    int,
    typing.Dict[str, bool],
]:
    """Add the slack variables, change b to all positives if necessary."""
    var_list: typing.Dict[str, bool] = {}
    slack_counter: int = 1

    equ_b = []
    binds = []
    # Add the slack variables
    for i, equ in enumerate(equs):
        # we assume that b > 0, if not we multiply the equation
        # by -1 and flip the operand accordingly
        # TODO-test me
        if equ.rhs < 0:
            for j in equ.vars_mults:
                equ.vars_mults[j] = equ.vars_mults[j] * -1
            equ.rhs = equ.rhs * -1
            if equ.operand == "<=":
                equ.operand = ">="
            elif equ.operand == "<":
                equ.operand = ">"
            elif equ.operand == ">=":
                equ.operand = "<="
            elif equ.operand == ">":
                equ.operand = "<"
            else:
                # for ==, we dont need to change the operand
                pass

        if len(equ.vars_mults) == 1 and equ.rhs == 0:
            binds.append(equ)
            continue

        if equ.operand in ("<=", "<"):
            if len(equ.vars_mults) >= 1:
                equ.vars_mults[slack + repr(slack_counter)] = 1
                slack_counter += 1
            if equ.operand == "<=":
                binds.append(Equation({slack + repr(i): 1}, ">=", 0.0))
            else:
                binds.append(Equation({slack + repr(i): 1}, ">", 0.0))
        elif equ.operand in (">=", ">"):
            if len(equ.vars_mults) >= 1:
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
        print("var_list:\n", var_list)

    return equs, equ_b[0], binds, slack_counter, var_list


def build_abc(
    equs: typing.List[Equation],
    cost_equ: Equation,
    var_list: typing.Dict[str, bool],
    slack_counter: int,
    slack: str,
    verbose: bool,
) -> typing.Tuple[
    np.ndarray[typing.Any, np.dtype[np.float32]],
    np.ndarray[typing.Any, np.dtype[np.float32]],
    np.ndarray[typing.Any, np.dtype[np.float32]],
    typing.List[str],
]:
    """Build A,b and C."""
    var_sorted_list: typing.List[str] = []
    m: int = len(equs)
    n: int = len(var_list)

    var_names: typing.List = []
    for key, _ in var_list.items():
        var_names.append(key)
    var_names.sort()

    if verbose:
        print(f"m: {m}, n: {n}")

    A: np.ndarray = np.zeros((m, n), dtype=np.float32)
    b: np.ndarray = np.zeros((m, 1), dtype=np.float32)
    C: np.ndarray = np.zeros((1, n), dtype=np.float32)
    # build A
    for i in range(0, m):
        for j in range(0, n):
            if var_names[j] in equs[i].vars_mults:
                A[i][j] = equs[i].vars_mults[var_names[j]]
            else:
                A[i][j] = 0

    # check for big M

    # build b
    for i in range(0, m):
        b[i, 0] = equs[i].rhs

    # Add the slack variables to the cost function
    for i in range(1, slack_counter):
        cost_equ.vars_mults[slack + repr(i)] = 0

    # build the list of the variables sorted lexicographically
    for _, v in enumerate(iter(sorted(var_list))):
        var_sorted_list.append(v)
    if verbose:
        print("var_list: ", var_sorted_list)

    # build C
    for i, (k, v) in enumerate(iter(sorted(cost_equ.vars_mults.items()))):
        # gets the index of the variable in the sorted list
        # we need to put zeroes for non-existant vars
        index: typing.List[int] = [
            j
            for j in range(0, len(var_sorted_list))
            if var_sorted_list[j] == k
        ]
        C[0, index[0]] = v

    if verbose:
        print("A:\n", A)
        print("b:\n", b)
        print("C:\n", C)

    return A, b, C, var_sorted_list


def construct_lp_problem(
    equs: typing.List[Equation],
    verbose: bool,
    slack: str,
    is_minimum: bool,
    aux_var_name: str,
) -> typing.Tuple[
    np.ndarray[typing.Any, np.dtype[np.float32]],
    np.ndarray[typing.Any, np.dtype[np.float32]],
    np.ndarray[typing.Any, np.dtype[np.float32]],
    np.ndarray[typing.Any, np.dtype[np.float32]],
    typing.List[Equation],
    Equation,
    typing.List[str],
    typing.Dict[str, bool],
    bool,
    typing.List[int],
]:
    """Construct the LP problem."""
    var_list: typing.Dict[str, bool] = {}
    var_sorted_list: typing.List[str] = []
    slack_counter: int = 0

    equs, cost_equ, _, slack_counter, var_list = add_slack_vars(
        equs, slack, verbose
    )

    A, b, C, var_sorted_list = build_abc(
        equs, cost_equ, var_list, slack_counter, slack, verbose
    )

    m: int = A.shape[0]
    col_list_list: typing.List[int] = []
    col_list: typing.Dict[int, int] = {}

    has_identity, col_list = find_identity(A)
    if has_identity and col_list is not None:
        print("col_list:", col_list)
        for _, v in col_list.items():
            col_list_list.append(v)
    else:
        # big M
        m_zero: float = calculate_big_m_zero(A, b, C)
        print("m_zero:", m_zero)
        ones_column_list = get_ones(A)
        print("ones_column_list:\n", ones_column_list)
        build_identity(
            A,
            equs,
            cost_equ,
            aux_var_name,
            m_zero,
            is_minimum,
            var_list,
            ones_column_list,
        )
        # update the A and C and var_list
        A, b, C, var_sorted_list = build_abc(
            equs, cost_equ, var_list, slack_counter, slack, verbose
        )
        col_list = {}
        has_identity, col_list = find_identity(A)
        if has_identity and col_list is not None:
            for _, v in col_list.items():
                col_list_list.append(v)

    # we will always have an identity basis
    B = np.identity(m, dtype=np.float32)

    return (
        A,
        b,
        B,
        C,
        equs,
        cost_equ,
        var_sorted_list,
        var_list,
        has_identity,
        col_list_list,
    )


# @nb.jit(nopython=True, cache=True)
def find_identity(
    A: np.ndarray[typing.Any, np.dtype[np.float32]],
) -> typing.Tuple[bool, typing.Dict[int, int]]:
    """Tries to find one m*m identity matrix
    inside A, returns one that it finds."""
    ones_count: int = 0
    last_one_row: int = -1
    ones: int = 0
    cancelled: bool = False
    m: int = A.shape[0]
    n: int = A.shape[1]
    col_list: typing.Dict[int, int] = {}

    # bye-bye cache locality
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
            ones_count = 0
            last_one_row = 0
            continue
        if ones_count == 1:
            ones += last_one_row
            col_list[last_one_row] = j
        ones_count = 0
        last_one_row = 0

    print("ones:", ones)
    if ones == (m * (m + 1)) / 2:
        return True, col_list
    return False, col_list


def get_ones(
    A: np.ndarray[typing.Any, np.dtype[np.float32]]
) -> typing.Dict[int, int]:
    """Extract the available ones that we can get from A."""
    m: int = A.shape[0]
    n: int = A.shape[1]
    seen_a_one: bool = False
    seen_trash: bool = False
    one_index: int = -1
    # col_sum: float = 0
    col_list: typing.Dict[int, int] = {}

    # bye-bye cache locality
    for j in range(0, n):
        for i in range(0, m):
            # col_sum += col_sum + A[i, j]
            if A[i, j] == 0:
                continue
            if A[i, j] == 1:
                if seen_a_one:
                    seen_trash = True
                    break
                seen_a_one = True
                one_index = i
            else:
                seen_trash = True
                break
        # if seen_a_one and col_sum == 1 and not seen_trash:
        if seen_a_one and not seen_trash:
            col_list[one_index] = j
        seen_a_one = False
        seen_trash = False
        one_index = -1
        # col_sum = 0

    return col_list


def build_identity(
    A: np.ndarray[typing.Any, np.dtype[np.float32]],
    equs: typing.List[Equation],
    cost: Equation,
    aux_var_name: str,
    m_zero: float,
    is_minimum: bool,
    var_list: typing.Dict[str, bool],
    col_list: typing.Dict[int, int],
) -> None:
    """Build an identity matrix by adding auxillary variables."""
    m: int = A.shape[0]
    count: int = 1
    for i in range(0, m):
        if i in col_list:
            # we dont need to do anything here, we already have a one
            # on this row
            pass
        else:
            # we need an aux value here
            equs[i].vars_mults[aux_var_name + repr(count)] = 1
            if is_minimum:
                cost.vars_mults[aux_var_name + repr(count)] = 1 * m_zero
            else:
                cost.vars_mults[aux_var_name + repr(count)] = -1 * m_zero
            var_list[aux_var_name + repr(count)] = True
            count += 1


def calculate_big_m_zero(
    A, b, C: np.ndarray[typing.Any, np.dtype[np.float32]]
) -> float:
    """Calculate big M0 according to this:
    https://www.atlantis-press.com/article/25838434.pdf
    """
    m: int = A.shape[0]
    n: int = A.shape[1]
    alpha: float = np.max(A)
    beta: float = np.max(b)
    gamma: float = np.max(C)
    m_zero: float = 2 * n * pow(m, m) * pow(alpha, m - 1) * beta * gamma
    return m_zero


# @nb.jit(nopython=True, cache=True)
def invert_matrix(M: np.ndarray) -> np.ndarray:
    """inverts a square matrix"""
    return np.linalg.inv(M)


# @nb.jit(nopython=True, cache=True, parallel=True)
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
    np.ndarray[typing.Any, np.dtype[np.float32]],
    np.ndarray[typing.Any, np.dtype[np.float32]],
]:
    """Calculate C_b*B^-1*b."""
    m: int = len(basic_var_column_list)
    C_b = np.zeros((1, m), dtype=np.float32)
    # objectives = np.zeros((1, m), dtype=np.float32)

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
    # for j in range(0, A.shape[1]):
    #     print("zj_cj:", np.matmul(w, A[:, j : j + 1]) - C[0, j])
    objectives = get_costs(w, A, C)
    if verbose:
        print("zj_cj:\n", objectives)

    return B_inv, objectives, w, C_b


# NOTE- there could be more than one minimum
# also this does not prevent cycling among extreme points
# currently unused
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

    return minimum, minimum_index


def get_leaving_var_lexi(
    M,
    B_inv,
    A,
    y_k: np.ndarray[typing.Any, np.dtype[np.float32]],
) -> typing.Tuple[float, int]:
    """Calculates the leaving variable using the lexicographic rule."""
    n = M.shape[0]
    m = M.shape[1]
    minimum: float = 1e100
    minimum_index: int = 0
    minimum_count: int = 0
    min_indexes: typing.List[int] = []
    for j in range(0, n):
        if y_k[j, 0] > 0:
            if M[j, 0] < minimum:
                minimum = M[j, 0]
                minimum_index = j
                minimum_count = 1
                min_indexes.clear()
                min_indexes.append(j)
            elif M[j, 0] == minimum:
                minimum_count += 1
                min_indexes.append(j)

    if minimum_count == 1:
        return minimum, minimum_index

    minimum = 1e100
    minimum_index = 0
    minimum_count = 0
    min_next_indexes: typing.List[int] = []
    for i in range(0, m):
        y_i = np.dot(B_inv, A[:, i : i + 1])
        y_i_bar_y_k = y_i / y_k
        for j in range(0, n):
            if j in min_indexes:
                if y_i_bar_y_k[j, 0] < minimum:
                    minimum = y_i_bar_y_k[j, 0]
                    minimum_index = j
                    minimum_count = 1
                    min_next_indexes.clear()
                    min_next_indexes.append(j)
                elif y_i_bar_y_k[j, 0] == minimum:
                    minimum_count += 1
                    min_next_indexes.append(j)
        # we expect to return until i=(m-1) i.e. for the mth column.
        # if we don't, then the columns of B_inv were not linearly
        # independent. We know the columns of B_inv are linearly
        # independent so it's mathematically fine.
        if len(min_next_indexes) == 1:
            return minimum, minimum_index
        min_indexes = min_next_indexes
        min_next_indexes = []

        minimum = 1e100
        minimum_index = 0
        minimum_count = 0

    # we should never return from here
    return 0, -1


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

    b_bar_div_y = np.divide(b_bar[:, :], y_k)
    print("b_bar_div_y:\n", b_bar_div_y)
    _, r = get_leaving_var_lexi(b_bar_div_y, B_inv, A, y_k)
    rr = basis_col_list[r]
    if verbose:
        print("b_bar/y_k:\n", b_bar_div_y)
        print("r: ", rr)

    B[:, r : r + 1] = A[:, k : k + 1]

    return rr, B


def get_k_for_min(
    M: np.ndarray[typing.Any, np.dtype[np.float32]],
    basis_col_list: typing.List[int],
) -> typing.Tuple[int, float]:
    """get the index of the entering variable for a minimization problem."""
    n = M.shape[1]
    maximum: float = -1e9
    maximum_index: int = -1
    for i in range(0, n):
        if i not in basis_col_list:
            if M[0, i] > maximum:
                maximum = M[0, i]
                maximum_index = i

    return maximum_index, maximum


def get_k_for_max(
    M: np.ndarray[typing.Any, np.dtype[np.float32]],
    basis_col_list: typing.List[int],
) -> typing.Tuple[int, float]:
    """get the index of the entering variable for a maximization problem."""
    n = M.shape[1]
    minimum: float = 1e9
    minimum_index: int = -1
    for i in range(0, n):
        if i not in basis_col_list:
            if M[0, i] < minimum:
                minimum = M[0, i]
                minimum_index = i

    return minimum_index, minimum


def calculate_optimal(
    b: np.ndarray[typing.Any, np.dtype[np.float32]],
    B_inv: np.ndarray[typing.Any, np.dtype[np.float32]],
    C_b: np.ndarray[typing.Any, np.dtype[np.float32]],
    basis_col_list: typing.List[int],
    var_sorted_list: typing.List[str],
) -> float:
    """Calculates the optimal value.
    B * x_b = b
    Z  = C_b * x_b
    """
    optim: float = 0.0

    x_b = np.dot(B_inv, b)
    print("x_b:\n", x_b)
    print("optimal solution point:")
    for basis in basis_col_list:
        print(var_sorted_list[basis])
    Z = np.dot(C_b, x_b)
    print("Z:\n", Z)

    optim = Z[0, 0]

    return optim


def solve_normal_simplex(
    A,
    b,
    C,
    B: np.ndarray[typing.Any, np.dtype[np.float32]],
    basis_is_identity: bool,
    basis_col_list: typing.List,
    var_sorted_list: typing.List[str],
    argparser: Argparser,
) -> None:
    """Solve using the normal simplex method."""
    verbose: bool = argparser.args.verbose
    round_count: int = 0
    while True:
        round_count += 1
        B_inv, objectives, w, C_b = calculate_objective(
            basis_col_list, A, B, C, basis_is_identity, verbose
        )
        if verbose:
            print("w:\n", w)

        if argparser.args.min:
            k, _ = get_k_for_min(objectives, basis_col_list)
        else:
            k, _ = get_k_for_max(objectives, basis_col_list)
        if verbose:
            print("k: ", k)
        extrmem_zj_cj = objectives[0, k]

        if argparser.args.min:
            if extrmem_zj_cj < 0:
                # we are done
                print(
                    "optimal min is:",
                    calculate_optimal(
                        b, B_inv, C_b, basis_col_list, var_sorted_list
                    ),
                )
                break
        else:
            if extrmem_zj_cj > 0:
                # we are done
                print(
                    "optimal max is:",
                    calculate_optimal(
                        b, B_inv, C_b, basis_col_list, var_sorted_list
                    ),
                )
                break

        y_k = np.dot(B_inv, A[:, k : k + 1])
        print("y_k:\n", y_k)
        if np.all(np.less_equal(y_k, 0)):
            # we are done
            # TODO- print the direction along which the value is unbounded
            print("unbounded optimal value.")
            break
        print(y_k)
        r, B = determine_leaving(
            k, A, B, B_inv, b, y_k, basis_col_list, basis_is_identity, verbose
        )
        basis_is_identity = False

        leaving_col: int = 0
        for i, basis in enumerate(basis_col_list):
            if basis == r:
                leaving_col = i
        basis_col_list[leaving_col] = k

        if verbose:
            print("B:\n", B)
        print("-------------------------------------------------")
        if round_count > argparser.args.iter:
            print("too many iterations.")
            break


def dsimplex() -> None:
    """The entry point for the module."""
    argparser = Argparser()
    verbose = argparser.args.verbose
    equs = parse_equations(argparser.args.equs, argparser.args.debug)
    (
        A,
        b,
        B,
        C,
        equs,
        _,
        var_sorted_list,
        _,
        basis_is_identity,
        basis_col_list,
    ) = construct_lp_problem(
        equs,
        verbose,
        argparser.args.slack,
        argparser.args.min,
        argparser.args.aux,
    )
    solve_normal_simplex(
        A,
        b,
        C,
        B,
        basis_is_identity,
        basis_col_list,
        var_sorted_list,
        argparser,
    )
