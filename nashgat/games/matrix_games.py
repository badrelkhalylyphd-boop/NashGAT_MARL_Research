"""Two-player bimatrix games with exact Nash equilibria and exploitability.

These are the smallest settings in which the exploitability (NashConv) used in the
NashGAT-MARL objective can be computed in closed form; they serve as unit-test
oracles for the estimators used on larger games.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class BimatrixGame:
    """Payoff matrices A (row player) and B (column player), both shape (m, n)."""

    name: str
    A: np.ndarray
    B: np.ndarray

    def payoffs(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        return float(x @ self.A @ y), float(x @ self.B @ y)

    def best_response_values(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Value of the best unilateral deviation of each player."""
        return float(np.max(self.A @ y)), float(np.max(x @ self.B))

    def nash_conv(self, x: np.ndarray, y: np.ndarray) -> float:
        """NashConv(x, y) = sum_i [max_{pi_i'} J_i(pi_i', pi_-i) - J_i(pi)] >= 0."""
        u1, u2 = self.payoffs(x, y)
        b1, b2 = self.best_response_values(x, y)
        return (b1 - u1) + (b2 - u2)

    def welfare(self, x: np.ndarray, y: np.ndarray) -> float:
        return sum(self.payoffs(x, y))


def prisoners_dilemma(T: float = 5.0, R: float = 3.0, P: float = 1.0, S: float = 0.0) -> BimatrixGame:
    """Actions (Cooperate, Defect); requires T > R > P > S."""
    assert T > R > P > S
    A = np.array([[R, S], [T, P]], dtype=float)
    return BimatrixGame("prisoners_dilemma", A, A.T.copy())


def matching_pennies() -> BimatrixGame:
    A = np.array([[1.0, -1.0], [-1.0, 1.0]])
    return BimatrixGame("matching_pennies", A, -A)


def battle_of_sexes() -> BimatrixGame:
    A = np.array([[2.0, 0.0], [0.0, 1.0]])
    B = np.array([[1.0, 0.0], [0.0, 2.0]])
    return BimatrixGame("battle_of_sexes", A, B)


def pure_nash_equilibria(game: BimatrixGame) -> list[tuple[int, int]]:
    m, n = game.A.shape
    eq = []
    for i, j in product(range(m), range(n)):
        if game.A[i, j] >= game.A[:, j].max() - 1e-12 and game.B[i, j] >= game.B[i, :].max() - 1e-12:
            eq.append((i, j))
    return eq


def mixed_nash_2x2(game: BimatrixGame) -> tuple[np.ndarray, np.ndarray] | None:
    """Fully mixed equilibrium of a 2x2 game via the indifference conditions.

    Player 2 mixes q so that player 1 is indifferent:
        q A[0,0] + (1-q) A[0,1] = q A[1,0] + (1-q) A[1,1].
    Returns None when no interior solution exists.
    """
    A, B = game.A, game.B
    den_q = A[0, 0] - A[0, 1] - A[1, 0] + A[1, 1]
    den_p = B[0, 0] - B[1, 0] - B[0, 1] + B[1, 1]
    if abs(den_q) < 1e-12 or abs(den_p) < 1e-12:
        return None
    q = (A[1, 1] - A[0, 1]) / den_q
    p = (B[1, 1] - B[1, 0]) / den_p
    if not (0.0 < p < 1.0 and 0.0 < q < 1.0):
        return None
    return np.array([p, 1.0 - p]), np.array([q, 1.0 - q])
