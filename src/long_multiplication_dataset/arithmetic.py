from dataclasses import dataclass, field
from typing import List, Dict, Optional


def digit_to_char(d: int) -> str:
    if 0 <= d <= 9:
        return str(d)
    elif 10 <= d <= 35:
        return chr(ord('A') + d - 10)
    else:
        raise ValueError(f"Digit {d} cannot be represented as a single character")


def get_digits(n: int, base: int = 10) -> List[int]:
    if n == 0:
        return [0]
    digits = []
    while n > 0:
        digits.append(n % base)
        n //= base
    return digits


@dataclass
class DigitWithCarry:
    col: int
    digit: int
    carry: int = 0


@dataclass
class PartialProduct:
    shift: int
    digits: List[DigitWithCarry]
    b_digit: int


@dataclass
class AdditionResult:
    digits: List[DigitWithCarry]


@dataclass
class MultiplicationResult:
    a: int
    b: int
    a_digits: List[int]
    b_digits: List[int]
    base: int
    partial_products: List[PartialProduct]
    additions: List[AdditionResult]
    result: int


def compute_partial_product(
    a_digits: List[int], b_digit: int, base: int
) -> List[DigitWithCarry]:
    result = []
    carry = 0
    for j, a_digit in enumerate(a_digits):
        product = a_digit * b_digit + carry
        digit = product % base
        carry = product // base
        result.append(DigitWithCarry(col=j, digit=digit, carry=carry))
    if carry > 0:
        result.append(DigitWithCarry(col=len(a_digits), digit=carry, carry=0))
    return result


def compute_addition(
    a_digits: List[DigitWithCarry],
    a_shift: int,
    b_digits: List[DigitWithCarry],
    b_shift: int,
    base: int,
) -> AdditionResult:
    a_map: Dict[int, int] = {d.col + a_shift: d.digit for d in a_digits}
    b_map: Dict[int, int] = {d.col + b_shift: d.digit for d in b_digits}

    all_cols = set(a_map.keys()) | set(b_map.keys())
    if not all_cols:
        return AdditionResult(digits=[])

    max_col = max(all_cols)
    result = []
    carry = 0
    for col in range(max_col + 1):
        da = a_map.get(col, 0)
        db = b_map.get(col, 0)
        total = da + db + carry
        digit = total % base
        carry = total // base
        result.append(DigitWithCarry(col=col, digit=digit, carry=carry))
    if carry > 0:
        result.append(DigitWithCarry(col=max_col + 1, digit=carry, carry=0))

    return AdditionResult(digits=result)


def compute_multiplication(a: int, b: int, base: int = 10) -> MultiplicationResult:
    a_digits = get_digits(a, base)
    b_digits = get_digits(b, base)

    partial_products: List[PartialProduct] = []
    for i, b_digit in enumerate(b_digits):
        pp_digits = compute_partial_product(a_digits, b_digit, base)
        partial_products.append(PartialProduct(shift=i, digits=pp_digits, b_digit=b_digit))

    additions: List[AdditionResult] = []
    if len(b_digits) >= 2:
        s = compute_addition(
            partial_products[0].digits,
            partial_products[0].shift,
            partial_products[1].digits,
            partial_products[1].shift,
            base,
        )
        additions.append(s)

        for i in range(2, len(b_digits)):
            s = compute_addition(
                additions[-1].digits,
                0,
                partial_products[i].digits,
                partial_products[i].shift,
                base,
            )
            additions.append(s)

    return MultiplicationResult(
        a=a,
        b=b,
        a_digits=a_digits,
        b_digits=b_digits,
        base=base,
        partial_products=partial_products,
        additions=additions,
        result=a * b,
    )
