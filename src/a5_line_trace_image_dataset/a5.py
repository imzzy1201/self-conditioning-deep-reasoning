"""Utilities for the alternating group A5.

Permutations are represented as 5-tuples of integers in 1..5.  A tuple ``p``
means ``p(k) = p[k - 1]``.
"""

from __future__ import annotations

from itertools import permutations
import json
import re


Permutation = tuple[int, int, int, int, int]

IDENTITY: Permutation = (1, 2, 3, 4, 5)


def inversion_count(perm: Permutation) -> int:
    count = 0
    for i, left in enumerate(perm):
        for right in perm[i + 1:]:
            if left > right:
                count += 1
    return count


def is_even_permutation(perm: Permutation) -> bool:
    validate_permutation(perm)
    return inversion_count(perm) % 2 == 0


def validate_permutation(perm: Permutation) -> None:
    if len(perm) != 5 or sorted(perm) != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected a permutation of 1..5, got {perm}")


def validate_a5_element(perm: Permutation) -> None:
    validate_permutation(perm)
    if not is_even_permutation(perm):
        raise ValueError(f"Expected an even permutation, got {perm}")


ALL_A5: tuple[Permutation, ...] = tuple(
    perm for perm in permutations((1, 2, 3, 4, 5)) if inversion_count(perm) % 2 == 0
)


def compose(left: Permutation, right: Permutation) -> Permutation:
    """Return ``left o right``.

    With the tuple convention above, ``(left o right)(k) = left(right(k))``.
    """

    validate_permutation(left)
    validate_permutation(right)
    return tuple(left[value - 1] for value in right)  # type: ignore[return-value]


def prefix_compositions(sequence: list[Permutation]) -> list[Permutation]:
    """Compute ``h_i = g_i o g_{i-1} o ... o g_1`` for each prefix."""

    current = IDENTITY
    prefixes: list[Permutation] = []
    for element in sequence:
        validate_a5_element(element)
        current = compose(element, current)
        prefixes.append(current)
    return prefixes


def format_permutation(perm: Permutation) -> str:
    return "[" + ",".join(str(value) for value in perm) + "]"


def format_sequence(sequence: list[Permutation]) -> str:
    return ",".join(format_permutation(perm) for perm in sequence)


def build_prompt(sequence: list[Permutation]) -> str:
    return f"Compose A5 prefixes. Elements: {format_sequence(sequence)}"


_PERM_RE = re.compile(r"\[([1-5],[1-5],[1-5],[1-5],[1-5])\]")


def parse_sequence_text(text: str) -> list[Permutation]:
    """Parse a sequence from either the raw sequence string or full prompt."""

    matches = _PERM_RE.findall(text)
    if not matches:
        raise ValueError("No A5 permutation brackets found")
    sequence: list[Permutation] = []
    for match in matches:
        perm = tuple(int(part) for part in match.split(","))  # type: ignore[assignment]
        validate_a5_element(perm)
        sequence.append(perm)
    return sequence


def sequence_to_json(sequence: list[Permutation]) -> str:
    return json.dumps([list(perm) for perm in sequence], separators=(",", ":"))


def sequence_from_json(value: str) -> list[Permutation]:
    raw = json.loads(value)
    sequence = [tuple(int(item) for item in perm) for perm in raw]
    for perm in sequence:
        validate_a5_element(perm)  # type: ignore[arg-type]
    return sequence  # type: ignore[return-value]

