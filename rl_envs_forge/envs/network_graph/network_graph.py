import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.linalg import expm
import networkx as nx
from .visualize import draw_network_graph, plot_centralities_sorted
from .graph_utils import (
    compute_centrality,
    get_weighted_adjacency_matrix,
    compute_laplacian,
    compute_eigenvector_centrality,
)


class NetworkGraph(gym.Env):
    metadata = {"render.modes": ["human", "matplotlib"]}

    def __init__(
        self,
        num_agents=10,
        connection_prob_range=(0.1, 0.3),
        max_u=0.1,
        desired_opinion=1.0,
        tau=1.0,
        initial_opinion_range=(0.0, 1.0),
        initial_opinions=None,
        control_resistance=None,
        connectivity_matrix=None,
        use_weighted_edges=False,
        weight_range=(0.1, 1.0),
        budget=10.0,
        max_steps=100,
        opinion_end_tolerance=0.01,
        control_beta=0.4,
    ):
        super(NetworkGraph, self).__init__()

        # Store the initialization parameters
        if connectivity_matrix is not None:
            self.num_agents = connectivity_matrix.shape[0]
        else:
            self.num_agents = num_agents
        self.connection_prob_range = connection_prob_range
        self.use_weighted_edges = use_weighted_edges
        self.weight_range = weight_range

        self.max_u = max_u
        self.desired_opinion = desired_opinion
        self.tau = tau
        self.initial_opinion_range = initial_opinion_range
        self.initial_opinions = initial_opinions
        self.control_resistance = control_resistance

        self.budget = budget
        self.max_steps = max_steps
        self.opinion_end_tolerance = opinion_end_tolerance
        self.control_beta = control_beta

        # Determine adjacency matrix
        if connectivity_matrix is not None:
            self.connectivity_matrix = connectivity_matrix
        else:
            # Generate a random adjacency matrix using Erdos-Renyi graph
            self.graph = nx.erdos_renyi_graph(
                self.num_agents, np.random.uniform(*connection_prob_range)
            )
            self.connectivity_matrix = nx.to_numpy_array(self.graph)

            if self.use_weighted_edges:
                # Assign random weights to the edges within the specified range
                for i in range(self.num_agents):
                    for j in range(i + 1, self.num_agents):
                        if self.connectivity_matrix[i, j] == 1:
                            weight = np.random.uniform(*self.weight_range)
                            self.connectivity_matrix[i, j] = weight
                            self.connectivity_matrix[j, i] = weight  # Ensure symmetry

        # Compute Laplacian
        self.L = compute_laplacian(self.connectivity_matrix)
        self.centralities = compute_eigenvector_centrality(self.L)

        # Check if the graph is fully connected
        G = nx.from_numpy_array(self.connectivity_matrix)
        if not nx.is_connected(G):
            print("Warning: The generated graph is not fully connected.")

        # Define or generate initial opinions
        if initial_opinions is not None:
            self.opinions = np.array(initial_opinions)
        else:
            self.opinions = np.random.uniform(
                *self.initial_opinion_range, size=self.num_agents
            )

        # Define or generate impulse resistance (cost) for each agent
        if control_resistance is not None:
            self.control_resistance = np.array(control_resistance)
        else:
            self.control_resistance = np.zeros(self.num_agents)

        # Define the action and observation spaces
        self.action_space = spaces.Box(
            low=0, high=self.max_u, shape=(self.num_agents,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_agents,), dtype=np.float32
        )

        self.current_step = 0
        self.total_spent = 0.0

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        if self.initial_opinions is None:
            self.opinions = np.random.uniform(
                *self.initial_opinion_range, size=self.num_agents
            )
        else:
            self.opinions = np.array(self.initial_opinions)
        self.current_step = 0
        self.total_spent = 0.0
        return self.opinions

    def compute_dynamics(self, current_state, control_action, step_duration):
        """
        Compute the new state of the network given the current state, control action, and step duration.

        Parameters:
            current_state (numpy.ndarray): The current state of the network.
            control_action (numpy.ndarray): The control action to apply to the network.
            step_duration (float): The duration of the step.

        Returns:
            numpy.ndarray: The new state of the network.
        """
        # Ensure step duration is positive
        if step_duration <= 0:
            raise ValueError("step_duration must be positive")

        effective_control = control_action * (1 - self.control_resistance)

        # Apply control to blend current opinions with the desired opinion
        opinions = (
            effective_control * self.desired_opinion
            + (1 - effective_control) * current_state
        )

        # Compute opinion propagation with the influence of the Laplacian
        expL_remaining = expm(-self.L * step_duration)
        propagated_opinions = expL_remaining @ opinions

        # Clip the resulting opinions to be within [0, 1]
        new_states = np.clip(propagated_opinions, 0, 1)
        return new_states

    def reward_function(self, x, u, d, beta):
        return -np.abs(d - x).sum() - beta * np.sum(u)

    def step(self, action, step_duration=None):
        """
        Execute one time step within the environment.

        Args:
            action (np.array): Control inputs for influencing the agents.
            step_duration (float, optional): Total duration over which the opinions are propagated.
                                            Defaults to self.tau.

        Returns:
            observation (np.array): Updated opinions.
            reward (float): Reward for the step.
            done (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated (e.g., due to reaching the max number of steps).
            info (dict): Additional information.
        """
        if step_duration is None:
            step_duration = self.tau

        # Clip the control action to within [0, max_u]
        action = np.clip(action, 0, self.max_u)

        # Use the compute_dynamics function to determine the next state
        next_opinions = self.compute_dynamics(self.opinions, action, step_duration)

        # Update environment state with the new opinions
        self.opinions = next_opinions
        self.total_spent += np.sum(action)

        # Compute reward based on closeness to the desired opinions and budget spent
        reward = self.reward_function(
            self.opinions, action, self.desired_opinion, self.control_beta
        )

        # Increment step counter and check if the episode is done or truncated
        self.current_step += 1
        done = self.total_spent >= self.budget
        truncated = self.current_step >= self.max_steps

        # Info dictionary can be used to pass additional information
        info = {
            "current_step": self.current_step,
            "total_spent": self.total_spent,
            "remaining_budget": self.budget - self.total_spent,
            "action_applied": action,
        }

        return self.opinions, reward, done, truncated, info

    def render(self, mode="matplotlib"):
        """
        Render the environment.

        Args:
            mode (str): The mode to render with. Supported modes: "human", "matplotlib".
        """
        if mode == "matplotlib":
            draw_network_graph(self.connectivity_matrix, self.centralities)
        elif mode == "centralities":
            plot_centralities_sorted(self.centralities)
        elif mode == "human":
            print(
                f"Step: {self.current_step}, Opinions: {self.opinions}, Total Spent: {self.total_spent}"
            )
        else:
            raise NotImplementedError(f"Render mode '{mode}' is not supported.")

    def close(self):
        """
        Clean up the environment.
        """
        pass
