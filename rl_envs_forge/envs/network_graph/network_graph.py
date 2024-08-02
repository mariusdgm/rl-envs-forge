import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx

class NetworkGraph(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, num_agents=100, connection_prob=0.1, max_u=0.1, budget=10.0, desired_opinion=1.0, tau=1.0):
        """
        Initialize the Network Graph environment.

        Args:
            num_agents (int): Number of agents in the network.
            connection_prob (float): Probability of connection between any two agents.
            max_u (float): Maximum control input for influencing agents.
            budget (float): Total budget for the marketing campaigns.
            desired_opinion (float): Desired opinion value that the external entity wants the agents to adopt.
            tau (float): Time step for the evolution dynamics.
        """
        super(NetworkGraph, self).__init__()

        self.num_agents = num_agents
        self.connection_prob = connection_prob
        self.max_u = max_u
        self.budget = budget
        self.desired_opinion = desired_opinion
        self.tau = tau

        self.graph = nx.erdos_renyi_graph(num_agents, connection_prob)
        self.adjacency_matrix = nx.to_numpy_array(self.graph)
        self.L = np.diag(np.sum(self.adjacency_matrix, axis=1)) - self.adjacency_matrix  # Laplacian matrix

        # Initial opinions of agents
        self.opinions = np.random.rand(num_agents)

        # Define the action and observation spaces
        self.action_space = spaces.Box(low=0, high=self.max_u, shape=(num_agents,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_agents,), dtype=np.float32)

        self.current_step = 0
        self.total_spent = 0.0
        self.max_steps = 100

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.opinions = np.random.rand(self.num_agents)
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
            info (dict): Additional information.
        """
        # Apply control action (impulsive dynamics at tk)
        self.opinions = action * self.desired_opinion + (1 - action) * self.opinions
        self.total_spent += np.sum(action)

        # Update opinions based on network influence (evolution dynamics between tk and tk+1)
        self.opinions += self.tau * (-self.L @ self.opinions)

        # Reward based on how close opinions are to the desired value and penalize budget
        reward = -np.sum((self.opinions - self.desired_opinion) ** 2) - 0.01 * np.sum(action)

        self.current_step += 1
        done = (self.current_step >= self.max_steps) or (self.total_spent >= self.budget)

        return self.opinions, reward, done, False, {}

    def render(self, mode="human"):
        """
        Render the environment.
        """
        print(f"Step: {self.current_step}, Opinions: {self.opinions}")

    def close(self):
        """
        Clean up the environment.
        """
        pass

