from dataclasses import dataclass, field
from typing import List, Optional

from arithmetic import (
    MultiplicationResult,
    PartialProduct,
    AdditionResult,
    DigitWithCarry,
    digit_to_char,
)
from config import ColumnMulConfig, round_up_to_multiple


@dataclass
class Cell:
    row: int
    col: int
    char: str
    is_carry: bool = False


@dataclass
class HLine:
    row: int
    col_left: int
    col_right: int


@dataclass
class Layout:
    cells: List[Cell] = field(default_factory=list)
    lines: List[HLine] = field(default_factory=list)
    num_rows: int = 0
    num_cols: int = 0


def _add_digit_cells(
    cells: List[Cell],
    row: int,
    digits: List[DigitWithCarry],
    shift: int = 0,
) -> None:
    all_zero = all(d.digit == 0 for d in digits)
    if all_zero:
        cells.append(Cell(row=row, col=shift, char="0", is_carry=False))
        return

    for dwc in digits:
        global_col = dwc.col + shift
        cells.append(
            Cell(row=row, col=global_col, char=digit_to_char(dwc.digit), is_carry=False)
        )
        if dwc.carry > 0:
            cells.append(
                Cell(
                    row=row,
                    col=global_col,
                    char=digit_to_char(dwc.carry),
                    is_carry=True,
                )
            )


def create_layout(
    result: MultiplicationResult,
    config: ColumnMulConfig,
    input_only: bool = False,
) -> Layout:
    cells: List[Cell] = []
    lines: List[HLine] = []

    m = len(result.a_digits)
    n = len(result.b_digits)

    mul_sign_col = n

    # Row 0: a
    for j, digit in enumerate(result.a_digits):
        cells.append(Cell(row=0, col=j, char=digit_to_char(digit), is_carry=False))

    # Row 1: × b
    cells.append(Cell(row=1, col=mul_sign_col, char="\u00d7", is_carry=False))
    for j, digit in enumerate(result.b_digits):
        cells.append(Cell(row=1, col=j, char=digit_to_char(digit), is_carry=False))

    # Row 2: separator line
    lines.append(HLine(row=2, col_left=max(m, n), col_right=0))

    fixed_num_cols = m + n + 1

    if input_only:
        num_rows = _compute_total_rows(n)
        return Layout(
            cells=cells,
            lines=lines,
            num_rows=num_rows,
            num_cols=fixed_num_cols,
        )

    current_row = 3

    # P_0
    _add_digit_cells(cells, current_row, result.partial_products[0].digits, result.partial_products[0].shift)
    current_row += 1

    if n >= 2:
        # P_1
        _add_digit_cells(cells, current_row, result.partial_products[1].digits, result.partial_products[1].shift)
        current_row += 1

        # Addition line
        lines.append(HLine(row=current_row, col_left=0, col_right=0))
        current_row += 1

        # S_1
        _add_digit_cells(cells, current_row, result.additions[0].digits)
        current_row += 1

        for i in range(2, n):
            # P_i
            _add_digit_cells(
                cells, current_row, result.partial_products[i].digits, result.partial_products[i].shift
            )
            current_row += 1

            # Addition line
            lines.append(HLine(row=current_row, col_left=0, col_right=0))
            current_row += 1

            # S_i (additions[i-1])
            _add_digit_cells(cells, current_row, result.additions[i - 1].digits)
            current_row += 1

    num_rows = current_row

    for line in lines:
        line.col_left = fixed_num_cols - 2

    return Layout(
        cells=cells,
        lines=lines,
        num_rows=num_rows,
        num_cols=fixed_num_cols,
    )


def _compute_total_rows(n: int) -> int:
    if n == 1:
        return 4
    return 3 * n + 1


def _compute_max_col(result: MultiplicationResult, mul_sign_col: int) -> int:
    m = len(result.a_digits)
    n = len(result.b_digits)
    max_col = max(m - 1, n - 1, mul_sign_col)

    for pp in result.partial_products:
        if pp.digits:
            col = max(d.col for d in pp.digits) + pp.shift
            max_col = max(max_col, col)

    for add in result.additions:
        if add.digits:
            col = max(d.col for d in add.digits)
            max_col = max(max_col, col)

    return max_col


def _compute_max_col_from_cells(cells: List[Cell], mul_sign_col: int) -> int:
    if not cells:
        return mul_sign_col
    return max(c.col for c in cells)
