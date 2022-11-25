import numpy as np

import gameoflife_ndimage.simulation as sim


def test_gol_64_zero_result():
    rules = sim.Rules2D.classic()
    size = (64, 64)
    state = sim.State2D.random(rules, size, seed=64)
    res = state.after(100).field[:5, :5]

    target = np.zeros((5, 5), dtype=bool)
    assert res.tolist() == target.tolist()


def test_gol_5x5_basic():
    rules = sim.Rules2D.classic()
    state = sim.State2D(
        np.array(
            [
                [0] * 5,
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0] * 5,
            ],
            dtype=bool,
        ),
        rules,
    )
    res = state.step().field

    target = np.array(
        [
            [0] * 5,
            [0] * 5,
            [0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=bool,
    )

    assert res.tolist() == target.tolist()


def test_static_cube():
    rules = sim.Rules2D.classic()
    state = sim.State2D(
        np.array(
            [
                [0] * 4,
                [0, 1, 1, 0],
                [0, 1, 1, 0],
                [0] * 4,
            ],
            dtype=bool,
        ),
        rules,
    )

    assert state.step().field.tolist() == state.field.tolist()


def test_oscillator_blinker():
    rules = sim.Rules2D.classic()
    state = sim.State2D(
        np.array(
            [
                [0] * 5,
                [0] * 5,
                [0, 1, 1, 1, 0],
                [0] * 5,
                [0] * 5,
            ],
            dtype=bool,
        ),
        rules,
    )
    _, s1, s2 = map(lambda x: x.field, state.run(2))
    assert s1.tolist() == state.field.T.tolist()  # rotates 90 degrees
    assert s2.tolist() == state.field.tolist()  # and returns to starting state
