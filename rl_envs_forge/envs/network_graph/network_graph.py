import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.linalg import expm
import networkx as nx
from .visualize import draw_network_graph, plot_centralities_sorted
from .graph_utils import (
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
        t_campaign=1.0,              
        t_s=0.1,                     
        initial_opinion_range=(0.0, 1.0),
        initial_opinions=None,
        control_resistance=None,
        connectivity_matrix=None,
        graph_connection_distribution="uniform",
        graph_connection_params=None,
        use_weighted_edges=False,
        weight_range=(0.1, 1.0),
        bidirectional_prob: float = 0.5,  # 1.0 = fully undirected, 0.0 = fully directed
        dynamics_model="laplacian", # "laplacian" or "coca"
        budget=None,
        max_steps=100,
        opinion_end_tolerance=0.01,
        control_beta=0.4,
        normalize_reward=False,
        terminal_reward=0.0,
        terminate_when_converged: bool = True,
        seed=None,
    ):
        super(NetworkGraph, self).__init__()

        self.seed = seed
        self._rng = np.random.default_rng(seed)
        
        if connectivity_matrix is not None:
            self.num_agents = connectivity_matrix.shape[0]
        else:
            self.num_agents = num_agents
            
        self.connection_prob_range = connection_prob_range
        self.use_weighted_edges = use_weighted_edges
        self.weight_range = weight_range

        if np.isscalar(max_u):
            self.max_u = np.full(self.num_agents, max_u, dtype=np.float32)
        else:
            assert len(max_u) == self.num_agents, "max_u vector must match number of agents"
            self.max_u = np.array(max_u, dtype=np.float32)

        self.desired_opinion = desired_opinion
        if np.isscalar(desired_opinion):
            self.desired_opinion = desired_opinion
            self.desired_opinion_vector = np.full(self.num_agents, desired_opinion)
        else:
            assert len(desired_opinion) == self.num_agents, "desired_opinion vector must match number of agents"
            self.desired_opinion = np.mean(desired_opinion)  # or any meaningful aggregate
            self.desired_opinion_vector = np.array(desired_opinion)
    
        # Store time parameters and validate
        self.t_campaign = t_campaign
        self.t_s = t_s

        if self.t_campaign <= 0:
            raise ValueError(f"t_campaign must be positive, got {self.t_campaign}")
        if self.t_s <= 0:
            raise ValueError(f"t_s must be positive, got {self.t_s}")
        if abs(self.t_campaign / self.t_s - round(self.t_campaign / self.t_s)) > 1e-8:
            raise ValueError(f"t_s = {self.t_s} must evenly divide t_campaign = {self.t_campaign}")
        
        self.initial_opinion_range = initial_opinion_range
        self.initial_opinions = initial_opinions
        self.control_resistance = control_resistance
        self.bidirectional_prob = bidirectional_prob
        self.dynamics_model = dynamics_model

        self.budget = budget
        self.max_steps = max_steps
        self.opinion_end_tolerance = opinion_end_tolerance
        self.control_beta = control_beta
        self.normalize_reward = normalize_reward
        self.terminal_reward = terminal_reward
        self.terminate_when_converged = terminate_when_converged
        
        if connectivity_matrix is not None:
            self.connectivity_matrix = connectivity_matrix
            if self.use_weighted_edges:
                for i in range(self.num_agents):
                    for j in range(self.num_agents):
                        if self.connectivity_matrix[i, j] == 1:
                            weight = self._rng.uniform(*self.weight_range)
                            self.connectivity_matrix[i, j] = weight
        else:
            graph_connection_params = graph_connection_params or {}

            # Generate probability matrix if needed
            if graph_connection_distribution == "uniform":
                prob = self._rng.uniform(*connection_prob_range)
                probs = np.full((self.num_agents, self.num_agents), prob)
            elif graph_connection_distribution == "normal":
                mean = graph_connection_params.get("mean", 0.2)
                std = graph_connection_params.get("std", 0.05)
                probs = np.clip(
                    self._rng.normal(mean, std, (self.num_agents, self.num_agents)), 0, 1
                )
            elif graph_connection_distribution == "exponential":
                scale = graph_connection_params.get("scale", 0.2)
                probs = np.clip(
                    self._rng.exponential(scale, (self.num_agents, self.num_agents)), 0, 1
                )
            else:
                raise ValueError(f"Unknown graph_connection_distribution: {graph_connection_distribution}")

            # Build directed graph with probabilistic bidirectionality
            if self.bidirectional_prob >= 1.0:
                # Fully undirected
                self.graph = nx.Graph()
                self.graph.add_nodes_from(range(self.num_agents))
                for i in range(self.num_agents):
                    for j in range(i + 1, self.num_agents):
                        if self._rng.random() < probs[i, j]:
                            self.graph.add_edge(i, j)
            else:
                # Directed graph with probabilistic bidirectionality
                self.graph = nx.DiGraph()
                self.graph.add_nodes_from(range(self.num_agents))
                for i in range(self.num_agents):
                    for j in range(self.num_agents):
                        if i == j:
                            continue
                        if self._rng.random() < probs[i, j]:
                            self.graph.add_edge(i, j)
                            if self._rng.random() < self.bidirectional_prob:
                                self.graph.add_edge(j, i)

            self.connectivity_matrix = nx.to_numpy_array(self.graph)

            # Ensure at least weak connectivity
            if isinstance(self.graph, nx.DiGraph):
                G = nx.DiGraph(self.connectivity_matrix)
            else:
                G = nx.Graph(self.connectivity_matrix)

            for node in range(self.num_agents):
                if (G.in_degree(node) + G.out_degree(node)) if isinstance(G, nx.DiGraph) else G.degree(node) == 0:
                    candidates = list(range(self.num_agents))
                    candidates.remove(node)
                    neighbor = self._rng.choice(candidates)
                    G.add_edge(node, neighbor)

            self.connectivity_matrix = nx.to_numpy_array(G)
        
        if not nx.is_connected(nx.from_numpy_array(self.connectivity_matrix)):
            print("Warning: The generated graph is not fully connected.")
            
        # Laplacian and centrality
        self.L = compute_laplacian(self.connectivity_matrix)
        self.centralities = compute_eigenvector_centrality(self.L)
        
        # Precompute neighbor indices for COCA dynamics
        self.neighbor_lists = []
        for i in range(self.num_agents):
            neighbors = np.nonzero(self.connectivity_matrix[i])[0]
            self.neighbor_lists.append(neighbors)

        if initial_opinions is not None:
            self.opinions = np.array(initial_opinions)
        else:
            self.opinions = self._rng.uniform(*self.initial_opinion_range, size=self.num_agents)

        if control_resistance is not None:
            self.control_resistance = np.array(control_resistance)
        else:
            self.control_resistance = np.zeros(self.num_agents)

        self.action_space = spaces.Box(
            low=np.zeros(self.num_agents, dtype=np.float32),
            high=self.max_u,
            shape=(self.num_agents,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.num_agents,), dtype=np.float32
        )

        self.current_step = 0
        self.total_spent = 0.0

    @property
    def state(self):
        """
        Getter for opinions, allowing access via self.state.
        """
        return self.opinions

    @state.setter
    def state(self, value):
        """
        Setter for opinions, allowing assignment via self.state.
        """
        self.opinions = np.array(value)
        
    def reset(self, randomize_opinions=False):
        """
        Reset the environment to its initial state.

        Args:
            randomize_opinions (bool): If True, ignore self.initial_opinions and use random initialization instead.
        """
        if randomize_opinions or self.initial_opinions is None:
            self.opinions = self._rng.uniform(*self.initial_opinion_range, size=self.num_agents)
            info = {"random_opinions": True}
        else:
            self.opinions = np.array(self.initial_opinions)
            info = {"random_opinions": False}

        self.current_step = 0
        self.total_spent = 0.0
        return self.opinions, info

    def compute_dynamics(self, current_state, control_action, t_campaign, t_s):
        """
        Compute the new state of the network given the current state, control action, and simulate over t_campaign.

        Parameters:
            current_state (numpy.ndarray): The current state of the network.
            control_action (numpy.ndarray): The control action to apply.
            t_campaign (float): Total duration to simulate (until next control).
            t_s (float): Time increment for integration steps.

        Returns:
            final_state (np.ndarray): State at the end of t_campaign.
            intermediate_states (List[np.ndarray]): All intermediate states including initial state after control.
        """
        if t_campaign <= 0 or t_s <= 0:
            raise ValueError("Both t_campaign and t_s must be positive.")

        if abs(t_campaign / t_s - round(t_campaign / t_s)) > 1e-8:
            raise ValueError(f"t_s = {t_s} must evenly divide t_campaign = {t_campaign}")

        # --- Apply impulse control (instantaneous) ---
        effective_control = control_action * (1 - self.control_resistance)
        controlled_state = effective_control * self.desired_opinion_vector + (1 - effective_control) * current_state

        # Store trajectory: initial controlled state
        intermediate_states = [controlled_state.copy()]

        num_steps = int(round(t_campaign / t_s))
        current = controlled_state.copy()

        for _ in range(num_steps):
            if self.dynamics_model == "laplacian":
                expL = expm(-self.L * t_s)
                current = expL @ current
                current = np.clip(current, 0, 1)

            elif self.dynamics_model == "coca":
                next_state = np.copy(current)
                for i in range(self.num_agents):
                    neighbors = self.neighbor_lists[i]
                    if len(neighbors) == 0:
                        continue
                    pi = current[i]
                    neighbor_opinions = current[neighbors]
                    sum_diff = np.sum(neighbor_opinions - pi)
                    delta = t_s * (pi * (1 - pi) / len(neighbors)) * sum_diff
                    next_state[i] = np.clip(pi + delta, 0, 1)
                current = next_state

            else:
                raise ValueError(f"Unknown dynamics model: {self.dynamics_model}")

            intermediate_states.append(current.copy())

        return current, np.array(intermediate_states)

    def reward_function(self, x, u, d, beta, done: bool = False):
        raw_reward = -np.abs(d - x).sum() - beta * np.sum(u)

        if self.normalize_reward:
            normalized_reward = raw_reward / self.num_agents
            reward = float(normalized_reward)
        else:
            reward = float(raw_reward)

        # Add terminal bonus if we're done due to reaching target
        if done:
            reward += self.terminal_reward

        return reward

    def step(self, action, t_campaign=None, t_s=None):
        """
        Execute one time step: apply control and simulate propagation.

        Args:
            action (np.array): Control inputs.
            t_campaign (float, optional): Time until next control (default: self.t_campaign).
            t_s (float, optional): Time step for internal dynamics (default: self.t_s).

        Returns:
            observation (np.array): State after t_campaign.
            reward (float): Reward for this step.
            done (bool): Whether the episode has ended.
            truncated (bool): Whether max steps were reached.
            info (dict): Additional info including intermediate_states.
        """
        t_campaign = t_campaign if t_campaign is not None else self.t_campaign
        t_s = t_s if t_s is not None else self.t_s

        original_action = np.array(action, copy=True)
        clipped_action = np.clip(original_action, 0, self.max_u)

        final_state, intermediate_states = self.compute_dynamics(self.opinions, clipped_action, t_campaign, t_s)

        self.total_spent += np.sum(original_action)
        self.current_step += 1
        self.opinions = final_state

        done = (
            self.terminate_when_converged
            and np.abs(np.mean(final_state) - self.desired_opinion) <= self.opinion_end_tolerance
        )
        truncated = self.current_step >= self.max_steps
        terminal_success = done and not truncated

        reward = self.reward_function(
            self.opinions, original_action, self.desired_opinion, self.control_beta, done=terminal_success
        )

        info = {
            "current_step": self.current_step,
            "total_spent": self.total_spent,
            "action_applied_raw": original_action,
            "action_applied_clipped": clipped_action,
            "terminal_reward_applied": terminal_success,
            "intermediate_states": intermediate_states  # Shape: (num_substeps+1, num_agents)
        }

        if self.budget is not None:
            info["remaining_budget"] = self.budget - self.total_spent

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
