import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
from .visualize import draw_network_graph
from .graph_utils import (
    compute_centrality,
    get_weighted_adjacency_matrix,
    compute_laplacian,
)


class NetworkGraph(gym.Env):
    metadata = {"render.modes": ["human", "matplotlib"]}

    def __init__(
        self,
        num_agents=10,
        connection_prob_range=(0.1, 0.3),
        max_u=0.1,
        budget=10.0,
        desired_opinion=1.0,
        tau=1.0,
        initial_opinion_range=(0.0, 1.0),
        max_steps=100,
        custom_adjacency_matrix=None,
        initial_opinions=None,
        impulse_resistance=None,
        connectivity_matrix=None,
        desired_centrality=None,
        agent_sizes=None,  # New parameter
        use_weighted_edges=False,
        weight_range=(0.1, 1.0),
    ):
        super(NetworkGraph, self).__init__()

        # Store the initialization parameters
        if connectivity_matrix is not None:
            self.num_agents = connectivity_matrix.shape[0]
        elif custom_adjacency_matrix is not None:
            self.num_agents = custom_adjacency_matrix.shape[0]
        elif agent_sizes is not None:
            self.num_agents = len(agent_sizes)
        else:
            self.num_agents = num_agents
        self.connection_prob_range = connection_prob_range
        self.max_u = max_u
        self.budget = budget
        self.desired_opinion = desired_opinion
        self.tau = tau
        self.initial_opinion_range = initial_opinion_range
        self.max_steps = max_steps
        self.initial_opinions = initial_opinions
        self.impulse_resistance = impulse_resistance
        self.use_weighted_edges = use_weighted_edges
        self.weight_range = weight_range
        self.agent_sizes = agent_sizes  # Store the agent sizes if provided

        # Check for conflicting initialization methods
        if (
            connectivity_matrix is not None or desired_centrality is not None
        ) and custom_adjacency_matrix is not None:
            raise ValueError(
                "Cannot provide both a custom_adjacency_matrix and a connectivity_matrix with desired_centrality. "
                "Please choose one method of initialization."
            )
            
        # Determine adjacency matrix
        if connectivity_matrix is not None:
            if desired_centrality is None:
                desired_centrality = np.ones(self.num_agents)/self.num_agents

            self.adjacency_matrix, self.centralities = get_weighted_adjacency_matrix(
                connectivity_matrix, desired_centrality
            )
        elif custom_adjacency_matrix is not None:
            self.adjacency_matrix = custom_adjacency_matrix
        else:
            # Generate a random adjacency matrix using Erdos-Renyi graph
            self.graph = nx.erdos_renyi_graph(
                self.num_agents, np.random.uniform(*connection_prob_range)
            )
            self.adjacency_matrix = nx.to_numpy_array(self.graph)

            if self.use_weighted_edges:
                # Assign random weights to the edges within the specified range
                for i in range(self.num_agents):
                    for j in range(i + 1, self.num_agents):
                        if self.adjacency_matrix[i, j] == 1:
                            weight = np.random.uniform(*self.weight_range)
                            self.adjacency_matrix[i, j] = weight
                            self.adjacency_matrix[j, i] = weight  # Ensure symmetry

        # Compute Laplacian
        self.L = compute_laplacian(self.adjacency_matrix)

        # Compute centralities if agent sizes are not provided
        if self.agent_sizes is None:
            self.centralities = compute_centrality(self.L, self.adjacency_matrix)
        else:
            self.centralities = (
                None  # Centralities are not needed if agent sizes are provided
            )

        # Check if the graph is fully connected
        G = nx.from_numpy_array(self.adjacency_matrix)
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
        if impulse_resistance is not None:
            self.impulse_resistance = np.array(impulse_resistance)
        else:
            self.impulse_resistance = np.ones(self.num_agents)

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
        self.total_spent += np.sum(action)

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

    def render(self, mode="matplotlib"):
        """
        Render the environment.

        Args:
            mode (str): The mode to render with. Supported modes: "human", "matplotlib".
        """
        if mode == "matplotlib":
            if self.agent_sizes is not None:
                draw_network_graph(self.adjacency_matrix, self.agent_sizes)
            else:
                draw_network_graph(self.adjacency_matrix, self.centralities)
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
