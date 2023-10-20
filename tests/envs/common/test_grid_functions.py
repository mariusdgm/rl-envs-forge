import pytest
import numpy as np
import random

from rl_envs_forge.envs.common.grid_functions import on_line


class TestGridFucntions:
    def test_on_line_positive_case(self):
        p = (1, 1)
        q = (2, 2)
        r = (3, 3)
        assert on_line(p, q, r) == True

    def test_on_line_outside_line_segment(self):
        p = (1, 1)
        q = (4, 4)
        r = (3, 3)
        assert on_line(p, q, r) == False

    def test_on_line_not_collinear(self):
        p = (1, 1)
        q = (2, 3)
        r = (3, 3)
        assert on_line(p, q, r) == False
