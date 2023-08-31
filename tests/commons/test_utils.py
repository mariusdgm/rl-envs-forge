import pytest
import random
import numpy as np

from rl_envs.envs.common.utils import set_random_seeds

def test_set_random_seeds_consistency():
    # Set seed
    set_random_seeds(42)

    # Generate random values
    random_val1 = random.random()
    np_random_val1 = np.random.rand()

    # Reset seed
    set_random_seeds(42)

    # Generate another set of random values
    random_val2 = random.random()
    np_random_val2 = np.random.rand()

    # Check that the random values are consistent after resetting the seed
    assert random_val1 == random_val2
    assert np_random_val1 == np_random_val2


def test_set_random_seeds_different_seed():
    # Set seed to 42
    set_random_seeds(42)

    # Generate random values
    random_val1 = random.random()
    np_random_val1 = np.random.rand()

    # Reset seed to 24 (a different seed)
    set_random_seeds(24)

    # Generate another set of random values
    random_val2 = random.random()
    np_random_val2 = np.random.rand()

    # Check that the random values are not the same after changing the seed
    assert random_val1 != random_val2
    assert np_random_val1 != np_random_val2