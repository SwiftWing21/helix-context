"""A simple calculator module for testing code chunking."""

import math
from typing import List, Optional


class Calculator:
    """Basic arithmetic calculator with history tracking."""

    def __init__(self):
        self.history: List[str] = []
        self._last_result: Optional[float] = None

    def add(self, a: float, b: float) -> float:
        result = a + b
        self._record(f"{a} + {b} = {result}")
        return result

    def subtract(self, a: float, b: float) -> float:
        result = a - b
        self._record(f"{a} - {b} = {result}")
        return result

    def multiply(self, a: float, b: float) -> float:
        result = a * b
        self._record(f"{a} * {b} = {result}")
        return result

    def divide(self, a: float, b: float) -> float:
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self._record(f"{a} / {b} = {result}")
        return result

    def _record(self, entry: str):
        self.history.append(entry)
        self._last_result = float(entry.split("= ")[1])


class ScientificCalculator(Calculator):
    """Extended calculator with scientific functions."""

    def square_root(self, x: float) -> float:
        if x < 0:
            raise ValueError("Cannot take square root of negative number")
        result = math.sqrt(x)
        self._record(f"sqrt({x}) = {result}")
        return result

    def power(self, base: float, exponent: float) -> float:
        result = math.pow(base, exponent)
        self._record(f"{base} ^ {exponent} = {result}")
        return result

    def factorial(self, n: int) -> int:
        if n < 0:
            raise ValueError("Factorial not defined for negative numbers")
        result = math.factorial(n)
        self._record(f"{n}! = {result}")
        return result

    def logarithm(self, x: float, base: float = math.e) -> float:
        if x <= 0:
            raise ValueError("Logarithm not defined for non-positive numbers")
        result = math.log(x, base)
        self._record(f"log_{base}({x}) = {result}")
        return result


def parse_expression(expr: str) -> float:
    """Parse and evaluate a simple arithmetic expression.

    Supports: +, -, *, / with parentheses.
    This is intentionally naive for testing purposes.
    """
    # Strip whitespace
    expr = expr.replace(" ", "")

    # Find innermost parentheses and evaluate
    while "(" in expr:
        start = expr.rfind("(")
        end = expr.find(")", start)
        inner = expr[start + 1 : end]
        result = _evaluate_flat(inner)
        expr = expr[:start] + str(result) + expr[end + 1 :]

    return _evaluate_flat(expr)


def _evaluate_flat(expr: str) -> float:
    """Evaluate a flat expression without parentheses."""
    # Handle addition and subtraction (lowest precedence)
    terms = []
    current = ""
    for i, ch in enumerate(expr):
        if ch in "+-" and i > 0 and expr[i - 1] not in "*/":
            terms.append(current)
            current = ch
        else:
            current += ch
    terms.append(current)

    total = 0.0
    for term in terms:
        total += _evaluate_term(term)
    return total


def _evaluate_term(term: str) -> float:
    """Evaluate a term with only multiplication and division."""
    factors = term.replace("/", "*/").split("*")
    result = 1.0
    for f in factors:
        if not f:
            continue
        if f.startswith("/"):
            divisor = float(f[1:])
            if divisor == 0:
                raise ValueError("Division by zero")
            result /= divisor
        else:
            result *= float(f)
    return result
