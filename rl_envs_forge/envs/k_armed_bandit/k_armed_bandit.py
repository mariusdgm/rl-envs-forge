from typing import Optional, List, Dict, Any, Union, Callable
import numpy as np
import gymnasium as gym

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


class Arm:
    IMPLEMENTED_DISTRIBUTIONS = [
        "normal",
        "uniform",
        "exponential",
        "gamma",
        "beta",
        "logistic",
        "weibull",
        "lognormal",
    ]

    def __init__(
        self,
        distribution: str = "normal",
        param_functions: Optional[List[Dict[str, Union[Callable, str]]]] = None,
        **kwargs,
    ):
        """
        Represents a single arm in the k-armed bandit.

        Args:
            distribution (str): The distribution type for this arm. E.g., "normal", "uniform", etc.
            kwargs: Parameters for the distribution. E.g., for a normal distribution, you might have `mean` and `std`.
            param_functions (List[Dict], optional): List of dictionaries. Each dictionary contains:
                - 'function': Callable that takes the current timestep and returns a new value for its target parameter.
                - 'target_param': The parameter of the distribution to be updated by the function.

        """
        self.distribution = distribution
        if self.distribution not in self.IMPLEMENTED_DISTRIBUTIONS:
            raise ValueError(
                f"Invalid distribution: {self.distribution}. Please use one of {self.IMPLEMENTED_DISTRIBUTIONS}."
            )
        self.init_params = (
            kwargs.copy()
        )  # keep a copy of the initial params for updating
        self.current_params = (
            kwargs.copy()
        )  # these are the actual params used for sampling
        self.param_functions = param_functions if param_functions else []

    def update_param(self, timestep: int):
        """
        Update parameters using the provided param_functions.

        Args:
            timestep (int): The current timestep.
        """
        for func_dict in self.param_functions:
            shift_value = func_dict["function"](timestep)
            self.current_params[func_dict["target_param"]] = (
                self.init_params[func_dict["target_param"]] + shift_value
            )

    def sample(self, np_random) -> float:
        """
        Sample a reward from this arm.

        Args:
            np_random: An instance of numpy's random, for reproducibility.

        Returns:
            float: The sampled reward.
        """
        if self.distribution == "normal":
            mean = self.current_params.get("mean", 0)
            std = self.current_params.get("std", 1)
            return np_random.normal(mean, std)

        elif self.distribution == "uniform":
            low = self.current_params.get("low", 0)
            high = self.current_params.get("high", 1)
            return np_random.uniform(low, high)

        elif self.distribution == "exponential":
            scale = self.current_params.get("scale", 1)
            return np_random.exponential(scale)

        elif self.distribution == "gamma":
            shape = self.current_params.get("shape")
            scale = self.current_params.get("scale")
            if shape is None or scale is None:
                raise ValueError(
                    "For gamma distribution, both 'shape' and 'scale' params are required."
                )
            return np_random.gamma(shape, scale)

        elif self.distribution == "beta":
            a = self.current_params.get("a")
            b = self.current_params.get("b")
            if a is None or b is None:
                raise ValueError(
                    "For beta distribution, both 'a' and 'b' params are required."
                )
            return np_random.beta(a, b)

        elif self.distribution == "logistic":
            loc = self.current_params.get("loc", 0)
            scale = self.current_params.get("scale", 1)
            return np_random.logistic(loc, scale)

        elif self.distribution == "weibull":
            a = self.current_params.get("a")
            if a is None:
                raise ValueError("For weibull distribution, 'a' param is required.")
            return np_random.weibull(a)

        elif self.distribution == "lognormal":
            mean = self.current_params.get("mean", 0)
            sigma = self.current_params.get("sigma", 1)
            return np_random.lognormal(mean, sigma)


class KArmedBandit(gym.Env):
    def __init__(
        self,
        k: int = 10,
        arm_params: Optional[Dict[int, Dict[str, Any]]] = None,
        seed: Optional[int] = None,
    ):
        """
        k-armed bandit environment for reinforcement learning.

        Args:
            k (int): Number of arms of the bandit.
            arm_params (Dict[int, Dict[str, Any]], optional): Dictionary with arm index as key and its parameters as value.
            seed (int, optional): Seed for reproducibility.
        """

        super().__init__()

        self.k = k
        self.seed = seed
        self.np_random = np.random.RandomState(self.seed)
        self.timestep = 0
        self.arm_params = arm_params

        self.action_space = gym.spaces.Discrete(k)
        self.observation_space = gym.spaces.Discrete(1)
        
        self.arms = None
        self._init_arms()

    def _init_arms(self):
        self._create_default_arms()
        if self.arm_params:
            self._create_custom_arms(self.arm_params)

    def _create_default_arms(self):
        """By default instantiate normal distribution arms."""
        default_means = self.np_random.randn(self.k)
        self.arms = [
            Arm(distribution="normal", mean=mean, std=1) for mean in default_means
        ]

    def _create_custom_arms(self, arm_params):
        for idx, params in arm_params.items():
            if idx < 0 or (idx > self.k - 1):
                raise ValueError(
                    f"Invalid arm index: {idx}, the arm index range is between 0 and {self.k-1}."
                )
            self.arms[idx] = Arm(**params)

    def step(self, action: int):
        """
        Take a step using the chosen action.

        Args:
            action (int): The chosen arm.

        Returns:
            tuple: observation, reward, done, info
        """
        
        assert 0 <= action < self.k, "Invalid action, must be between 0 and k-1."

        # Get reward by sampling from the chosen arm
        reward = self.arms[action].sample(self.np_random)

        # Increment timestep
        self.timestep += 1

        # Update the parameters of all arms based on timestep
        for arm in self.arms:
            arm.update_param(self.timestep)

        done = True
        truncated = False
        return action, reward, done, truncated, {}

    def reset(self, seed: int = None) -> int:
        """
        Reset the environment.

        For k-armed bandit, the environment state remains unchanged between resets.

        Args:
            seed (int, optional): Seed for reproducibility.
        Returns:
            int: An initial observation, which is always 0.
        """
        if seed is not None:
            self.seed = seed
        self.np_random = np.random.RandomState(self.seed)
        self._init_arms()
        return 0

    def render(self, mode="violin_plot"):
        """
        Render the environment.

        Args:
            mode (str): The mode for rendering. Can be 'violin_plot' or 'print'.
        """
        if mode == "violin_plot":
            self.make_violinplot()
        elif mode == "print":
            self.print_arms_info()
        else:
            raise ValueError(
                f"Invalid mode: {mode}, please use 'violin_plot' or 'print'."
            )

    def make_violinplot(self):
        """
        Create a violin plot for the reward distributions of each arm.
        """
        # Generating samples for each arm
        samples = []
        for arm in self.arms:
            arm_samples = [
                arm.sample(self.np_random) for _ in range(1000)
            ]  # 1000 samples per arm
            samples.append(arm_samples)

        # Using seaborn for the violin plot
        sns.violinplot(data=samples)
        plt.xlabel("Arm")
        plt.ylabel("Reward Distribution")
        plt.title("Reward Distributions of Arms")
        plt.show()

    def print_arms_info(self):
        """
        Print information about the reward distributions of each arm.
        """
        for i, arm in enumerate(self.arms):
            print(f"Arm {i}:")
            print(f"\tDistribution: {arm.distribution.capitalize()}")
            for param, value in arm.current_params.items():
                print(f"\t{param.capitalize()}: {value:.2f}")
            print("-" * 30)

    def close(self):
        """
        Close the environment.

        For this simple environment, there's nothing to close, but it's added for API consistency.
        """
        pass

    @property
    def state(self):
        """
        Returns the current state of the k-armed bandit environment.

        Returns:
            tuple: (current timestep, list of current parameters for all arms)
        """
        arm_parameters = [arm.current_params for arm in self.arms]
        return self.timestep, arm_parameters
