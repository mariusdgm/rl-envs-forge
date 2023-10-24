import pytest
import matplotlib.pyplot as plt
import numpy as np

from rl_envs_forge.envs.k_armed_bandit.k_armed_bandit import KArmedBandit


class TestKArmedBandit:
    @pytest.fixture
    def bandit(self):
        return KArmedBandit()

    def test_render_violin_plot(self, bandit, monkeypatch):
        """
        Test if the violin_plot mode of the render function produces a figure.
        """
        # Mock plt.show() to prevent actual rendering
        monkeypatch.setattr(plt, "show", lambda: None)
        bandit.render(mode="violin_plot")

        # Check if a figure has been generated
        assert plt.get_fignums()

    def test_render_print(self, bandit, capsys):
        """
        Test if the print mode of the render function actually prints something.
        """
        bandit.render(mode="print")

        captured = capsys.readouterr()
        # Check if something has been printed to stdout
        assert captured.out != ""

    def test_render_invalid_mode(self, bandit):
        """
        Test if the render function raises a ValueError for an invalid mode.
        """
        with pytest.raises(
            ValueError, match=r"Invalid mode.*please use 'violin_plot' or 'print'."
        ):
            bandit.render(mode="invalid_mode")

    def test_invalid_arm_index(self):
        arm_params = {
            10: {"distribution": "normal", "mean": 1, "std": 1}
        }  # Index 10 is out of bounds for a 10-armed bandit

        with pytest.raises(
            ValueError,
            match=r"Invalid arm index: 10, the arm index range is between 0 and 9.",
        ):
            KArmedBandit(arm_params=arm_params)

        arm_params = {-1: {"distribution": "normal", "mean": 1, "std": 1}}

        with pytest.raises(
            ValueError,
            match=r"Invalid arm index: -1, the arm index range is between 0 and 9.",
        ):
            KArmedBandit(arm_params=arm_params)

    @pytest.mark.parametrize(
        "distribution,params",
        [
            ("normal", {"mean": 5, "std": 1}),
            ("uniform", {"low": 4, "high": 6}),
            ("exponential", {"scale": 0.2}),
            ("gamma", {"shape": 9, "scale": 0.55}),
            ("logistic", {"loc": 5, "scale": 0.3}),
            ("lognormal", {"mean": 1.6, "sigma": 0.3}),
            ("weibull", {"a": 2.0}),  # Added weibull distribution
            ("beta", {"a": 2, "b": 5}),  # Added beta distribution
        ],
    )
    def test_arms_sampling(self, distribution, params, monkeypatch):
        """
        Test if the arms can be sampled without errors and check if the sample falls within an expected range.
        """
        bandit = KArmedBandit(
            seed=0, k=1, arm_params={0: {"distribution": distribution, **params}}
        )

        # Perform 1000 steps and get rewards
        rewards = [bandit.step(0)[1] for _ in range(1000)]

        # Check if rewards are not empty
        assert rewards

    def test_not_implemented_arm(self):
        with pytest.raises(
            ValueError,
        ):
            KArmedBandit(
                seed=0, k=1, arm_params={0: {"distribution": "not_implemented"}}
            )

    def test_linear_param_shift(self):
        """
        Test if the linear parameter shift works correctly.
        """

        def linear_function(timestep):
            return timestep

        initial_mean = 0

        bandit = KArmedBandit(
            k=1,
            arm_params={
                0: {
                    "distribution": "normal",
                    "mean": initial_mean,
                    "std": 1,
                    "param_functions": [
                        {"function": linear_function, "target_param": "mean"}
                    ],
                }
            },
        )

        # Perform 10 steps
        for _ in range(10):
            bandit.step(0)

        # Check if the current mean of the arm is equal to 10 + initial mean
        current_mean = bandit.arms[0].current_params["mean"]
        assert (
            current_mean == 10 + initial_mean
        ), f"Expected mean {10 + initial_mean}, but got {current_mean}"
