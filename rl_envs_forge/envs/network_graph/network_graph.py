import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx

from .visualize import draw_network_graph
from .aux_functions import compute_centrality


class NetworkGraph(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        num_agents=100,
        connection_prob_range=(0.05, 0.15),
        max_u=0.1,
        budget=10.0,
        desired_opinion=1.0,
        tau=1.0,
        initial_opinion_range=(0.0, 1.0),
        max_steps=100,
        custom_adjacency_matrix=None,
        initial_opinions=None,
        impulse_resistance=None,
    ):
        """
        Initialize the Network Graph environment.

        Args:
            num_agents (int): Number of agents in the network.
            connection_prob_range (tuple): Range of probabilities for connections between any two agents.
            max_u (float): Maximum control input for influencing agents.
            budget (float): Total budget for the marketing campaigns.
            desired_opinion (float): Desired opinion value that the external entity wants the agents to adopt.
            tau (float): Time step for the evolution dynamics.
            initial_opinion_range (tuple): Range of initial opinions for agents.
            max_steps (int): Maximum number of steps per episode.
            custom_adjacency_matrix (np.array, optional): Custom adjacency matrix to define connections.
            initial_opinions (np.array, optional): Custom initial opinions for the agents.
            impulse_resistance (np.array, optional): Array representing resistance to impulse for each agent.
        """
        super(NetworkGraph, self).__init__()

        self.num_agents = num_agents
        self.connection_prob_range = connection_prob_range
        self.max_u = max_u
        self.budget = budget
        self.desired_opinion = desired_opinion
        self.tau = tau
        self.initial_opinions = initial_opinions
        self.initial_opinion_range = initial_opinion_range
        self.max_steps = max_steps

        # Define or generate the adjacency matrix
        if custom_adjacency_matrix is not None:
            self.adjacency_matrix = custom_adjacency_matrix
        else:
            self.graph = nx.erdos_renyi_graph(
                num_agents, np.random.uniform(*connection_prob_range)
            )
            self.adjacency_matrix = nx.to_numpy_array(self.graph)

        self.L = (
            np.diag(np.sum(self.adjacency_matrix, axis=1)) - self.adjacency_matrix
        )  # Laplacian matrix

        # Define or generate initial opinions
        if initial_opinions is not None:
            self.opinions = np.array(initial_opinions)
        else:
            self.opinions = np.random.uniform(
                *self.initial_opinion_range, size=num_agents
            )

        # Define or generate impulse resistance (cost) for each agent
        if impulse_resistance is not None:
            self.impulse_resistance = np.array(impulse_resistance)
        else:
            self.impulse_resistance = np.ones(
                num_agents
            )  # Default resistance is 1 for all agents

        # Define the action and observation spaces
        self.action_space = spaces.Box(
            low=0, high=self.max_u, shape=(num_agents,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(num_agents,), dtype=np.float32
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

    def step(self, action):
        """
        Execute one time step within the environment.

        Args:
            action (np.array): Control inputs for influencing the agents.

        Returns:
            observation (np.array): Updated opinions.
            reward (float): Reward for the step.
            done (bool): Whether the episode has ended.
            truncated (bool): Whether the episode was truncated (e.g., due to reaching the max number of steps).
            info (dict): Additional information.
        """
        # Apply control action (impulsive dynamics at tk)
        self.opinions = action * self.desired_opinion + (1 - action) * self.opinions
        self.total_spent += np.sum(
            action * self.impulse_resistance
        )  # Apply resistance cost

        # Update opinions based on network influence (evolution dynamics between tk and tk+1)
        self.opinions += self.tau * (-self.L @ self.opinions)

        # Reward based on how close opinions are to the desired value and penalize budget
        reward = -np.sum((self.opinions - self.desired_opinion) ** 2) - 0.01 * np.sum(
            action * self.impulse_resistance
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
        }

        return self.opinions, reward, done, truncated, info

    def render(self, mode="human"):
        # Compute centralities based on the Laplacian matrix
        centralities = compute_centrality(self.L)
        draw_network_graph(self.adjacency_matrix, centralities)
        print(
            f"Step: {self.current_step}, Opinions: {self.opinions}, Total Spent: {self.total_spent}"
        )

    def close(self):
        """
        Clean up the environment.
        """
        pass
