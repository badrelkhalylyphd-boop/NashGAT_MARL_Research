import numpy as np
import pytest

from nashgat.games.matrix_games import (battle_of_sexes, matching_pennies, mixed_nash_2x2, prisoners_dilemma,
                                        pure_nash_equilibria)
from nashgat.games.polymatrix import (grid_anti_coordination, networked_prisoners_dilemma,
                                      train_softmax_profile)

C, D = np.array([1.0, 0.0]), np.array([0.0, 1.0])


def test_pd_unique_nash_is_defect():
    g = prisoners_dilemma()
    assert pure_nash_equilibria(g) == [(1, 1)]
    assert g.nash_conv(D, D) == pytest.approx(0.0)
    # (C,C): each player gains T-R = 2 by defecting
    assert g.nash_conv(C, C) == pytest.approx(4.0)
    assert g.welfare(C, C) > g.welfare(D, D)


def test_matching_pennies_mixed_equilibrium():
    g = matching_pennies()
    assert pure_nash_equilibria(g) == []
    x, y = mixed_nash_2x2(g)
    np.testing.assert_allclose(x, [0.5, 0.5])
    np.testing.assert_allclose(y, [0.5, 0.5])
    assert g.nash_conv(x, y) == pytest.approx(0.0, abs=1e-12)
    assert g.nash_conv(np.array([0.8, 0.2]), y) == pytest.approx(0.6)  # player 2 deviates: 0.6


def test_battle_of_sexes_equilibria():
    g = battle_of_sexes()
    assert set(pure_nash_equilibria(g)) == {(0, 0), (1, 1)}
    x, y = mixed_nash_2x2(g)
    np.testing.assert_allclose(x, [2 / 3, 1 / 3])
    np.testing.assert_allclose(y, [1 / 3, 2 / 3])
    assert g.nash_conv(x, y) == pytest.approx(0.0, abs=1e-12)


def test_nash_conv_nonnegative_random_profiles():
    rng = np.random.default_rng(0)
    for g in (prisoners_dilemma(), matching_pennies(), battle_of_sexes()):
        for _ in range(100):
            x, y = rng.dirichlet([1, 1]), rng.dirichlet([1, 1])
            assert g.nash_conv(x, y) >= -1e-12


def test_networked_pd_enumeration():
    g = networked_prisoners_dilemma(2, 3)
    assert g.pure_nash_equilibria() == [(1,) * 6]
    eye = np.eye(2)
    assert g.nash_conv(eye[[1] * 6]) == pytest.approx(0.0)
    # all-cooperate: every agent gains (T-R) per neighbour by defecting
    deg = np.array([len(g.neighbours(i)) for i in range(6)])
    assert g.nash_conv(eye[[0] * 6]) == pytest.approx(2.0 * deg.sum())


def test_anti_coordination_checkerboards_are_nash():
    g = grid_anti_coordination(2, 2)
    eq = set(g.pure_nash_equilibria())
    assert {(0, 1, 1, 0), (1, 0, 0, 1)} <= eq


def test_gradient_learners_behave_as_theory_predicts():
    g = networked_prisoners_dilemma(2, 2)
    ind = train_softmax_profile(g, "independent", steps=1500, seed=1)
    wel = train_softmax_profile(g, "welfare", steps=1500, seed=1)
    assert ind["nash_conv"] < 0.05  # independent gradient play reaches the dominant-strategy NE
    assert wel["welfare"] > ind["welfare"]  # welfare ascent cooperates ...
    assert wel["nash_conv"] > 1.0  # ... and is exploitable
    reg = train_softmax_profile(g, "welfare_gap", lam=2.0, steps=1500, seed=1)
    assert reg["nash_conv"] < wel["nash_conv"]


def test_semi_gradient_gap_equals_reward_mixing():
    """Proposition 4: with Q held fixed, grad(-lam*gap) = lam * grad(own payoff)."""
    g = networked_prisoners_dilemma(2, 2)
    a = train_softmax_profile(g, "welfare_gap", lam=0.7, steps=300, seed=3)
    b = train_softmax_profile(g, "welfare_mix", lam=0.7, steps=300, seed=3)
    np.testing.assert_allclose(a["x"], b["x"], atol=1e-10)


def test_dense_action_values_match_loop():
    import torch
    g = networked_prisoners_dilemma(2, 3)
    x = np.random.default_rng(0).dirichlet([1, 1], size=6)
    np.testing.assert_allclose(g.torch_action_values(torch.as_tensor(x)).numpy(), g.action_values(x))
