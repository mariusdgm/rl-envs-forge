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
        dynamics_model="laplacian",  # "laplacian" or "coca"
        budget=None,
        max_steps=100,
        opinion_end_tolerance=0.01,
        control_beta=0.4,
        normalize_reward=False,
        use_delta_shaping: bool = False,
        delta_lambda: float = 0.0,
        terminal_reward=0.0,
        terminate_when_converged: bool = True,
        seed=None,
    ):
        """Initializes the Network Opinion Dynamics environment.

        This constructor sets up a Gymnasium environment for simulating the control
        of opinions on a social network. It handles the generation of the network
        graph, the initialization of agent opinions, and the configuration of
        the dynamics, control, and reward systems.

        The environment can be configured in two main ways:
        1.  By providing a pre-defined `connectivity_matrix`.
        2.  By specifying parameters to dynamically generate a graph. In this
            case, `num_agents` and connection probabilities are used.

        Args:
            num_agents (int, optional): The number of agents in the network.
                This parameter is ignored if `connectivity_matrix` is provided.
                Defaults to 10.
            connection_prob_range (tuple[float, float], optional): The lower and
                upper bounds for sampling the connection probability when
                generating a random graph with a 'uniform' distribution.
                Defaults to (0.1, 0.3).
            max_u (float | np.ndarray, optional): The maximum control effort
                that can be applied to any single agent in one step. Can be a
                scalar to set a uniform limit for all agents, or a numpy array
                of shape (num_agents,) to specify per-agent limits. This defines
                the upper bound of the action space. Defaults to 0.1.
            desired_opinion (float | np.ndarray, optional): The target opinion
                value. The goal of the agent is to drive all opinions towards
                this value. Can be a scalar for a uniform target or a numpy
                array of shape (num_agents,) for agent-specific targets.
                Defaults to 1.0.
            t_campaign (float, optional): The total duration of the autonomous
                opinion propagation phase between two control actions.
                Defaults to 1.0.
            t_s (float, optional): The time step for the numerical integration
                of the opinion dynamics during the propagation phase. This value
                must evenly divide `t_campaign`. Defaults to 0.1.
            initial_opinion_range (tuple[float, float], optional): The lower and
                upper bounds for randomly initializing agent opinions at the
                start of an episode. Used only if `initial_opinions` is not
                provided. Defaults to (0.0, 1.0).
            initial_opinions (np.ndarray | None, optional): A specific vector of
                initial opinions for the agents. If provided, this overrides
                `initial_opinion_range` and ensures a deterministic start.
                Defaults to None.
            control_resistance (np.ndarray | None, optional): A vector of shape
                (num_agents,) where each element in [0, 1) represents an agent's
                "stubbornness" or resistance to external control. A value of 0
                means no resistance, while a value close to 1 means high
                resistance. Defaults to a vector of zeros.
            connectivity_matrix (np.ndarray | None, optional): A pre-defined
                adjacency matrix of shape (num_agents, num_agents). If provided,
                it overrides all other graph generation parameters like
                `num_agents` and `connection_prob_range`. Defaults to None.
            graph_connection_distribution (str, optional): The probability
                distribution to use for generating graph edges. Supported values
                are 'uniform', 'normal', and 'exponential'. Defaults to 'uniform'.
            graph_connection_params (dict | None, optional): A dictionary of
                parameters for the chosen `graph_connection_distribution`. For
                'normal', this can include 'mean' and 'std'; for 'exponential',
                it can include 'scale'. Defaults to None.
            use_weighted_edges (bool, optional): If True, edges in the graph
                will be assigned weights sampled uniformly from `weight_range`.
                This affects the Laplacian matrix calculation. Defaults to False.
            weight_range (tuple[float, float], optional): The range of possible
                weights for graph edges if `use_weighted_edges` is True.
                Defaults to (0.1, 1.0).
            bidirectional_prob (float, optional): The probability that an edge
                (i, j) will have a corresponding reverse edge (j, i). A value
                of 1.0 creates a fully undirected graph, while 0.0 creates a
                fully directed graph (where bidirectionality only occurs by
                chance). Defaults to 0.5.
            dynamics_model (str, optional): The model to use for opinion
                propagation. Supported models are 'laplacian' (linear) and
                'coca' (non-linear). Defaults to 'laplacian'.
            budget (float | None, optional): An optional total budget for control
                effort over an entire episode. The environment does not enforce
                this budget but tracks spending against it. Defaults to None.
            max_steps (int, optional): The maximum number of steps allowed in an
                episode before it is truncated. Defaults to 100.
            opinion_end_tolerance (float, optional): The tolerance for the mean
                opinion to be considered "converged" to the `desired_opinion`.
                If the mean opinion is within this tolerance, the episode ends.
                Defaults to 0.01.
            control_beta (float, optional): The weight factor (lambda) used in
                the reward function to penalize the total control effort applied
                in a step. Defaults to 0.4.
            normalize_reward (bool, optional): If True, the raw reward is divided
                by the number of agents. This can help stabilize training across
                environments with different numbers of agents. Defaults to False.
            terminal_reward (float, optional): A bonus reward added at the final
                step if the episode terminates because the opinions have
                successfully converged (i.e., not due to truncation).
                Defaults to 0.0.
            terminate_when_converged (bool, optional): If True, the episode ends
                with `done=True` as soon as the mean opinion is within the
                `opinion_end_tolerance`. If False, the episode runs until
                `max_steps` is reached. Defaults to True.
            seed (int | None, optional): A seed for the random number generator
                to ensure reproducibility of graph generation and initial
                opinion sampling. Defaults to None.

        Raises:
            ValueError: If `t_s` is not a positive value or does not evenly
                divide `t_campaign`.
            ValueError: If `graph_connection_distribution` is an unknown type.
        """
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
            assert (
                len(max_u) == self.num_agents
            ), "max_u vector must match number of agents"
            self.max_u = np.array(max_u, dtype=np.float32)

        self.desired_opinion = desired_opinion
        if np.isscalar(desired_opinion):
            self.desired_opinion = desired_opinion
            self.desired_opinion_vector = np.full(self.num_agents, desired_opinion)
        else:
            assert (
                len(desired_opinion) == self.num_agents
            ), "desired_opinion vector must match number of agents"
            self.desired_opinion = np.mean(
                desired_opinion
            )  # or any meaningful aggregate
            self.desired_opinion_vector = np.array(desired_opinion)

        # Store time parameters and validate
        self.t_campaign = t_campaign
        self.t_s = t_s

        if self.t_campaign < 0:
            raise ValueError(f"t_campaign must be positive, got {self.t_campaign}")
        if self.t_s <= 0:
            raise ValueError(f"t_s must be positive, got {self.t_s}")
        if abs(self.t_campaign / self.t_s - round(self.t_campaign / self.t_s)) > 1e-8:
            raise ValueError(
                f"t_s = {self.t_s} must evenly divide t_campaign = {self.t_campaign}"
            )

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
        self.use_delta_shaping = use_delta_shaping
        self.delta_lambda = float(delta_lambda)

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
                    self._rng.normal(mean, std, (self.num_agents, self.num_agents)),
                    0,
                    1,
                )
            elif graph_connection_distribution == "exponential":
                scale = graph_connection_params.get("scale", 0.2)
                probs = np.clip(
                    self._rng.exponential(scale, (self.num_agents, self.num_agents)),
                    0,
                    1,
                )
            else:
                raise ValueError(
                    f"Unknown graph_connection_distribution: {graph_connection_distribution}"
                )

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
                if (
                    (G.in_degree(node) + G.out_degree(node))
                    if isinstance(G, nx.DiGraph)
                    else G.degree(node) == 0
                ):
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
            self.opinions = self._rng.uniform(
                *self.initial_opinion_range, size=self.num_agents
            )

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
            self.opinions = self._rng.uniform(
                *self.initial_opinion_range, size=self.num_agents
            )
            info = {"random_opinions": True}
        else:
            self.opinions = np.array(self.initial_opinions)
            info = {"random_opinions": False}

        self.current_step = 0
        self.total_spent = 0.0
        return self.opinions, info

    def compute_dynamics(
        self, current_state, control_action, t_campaign=None, t_s=None
    ):
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
        if t_campaign is None:
            t_campaign = self.t_campaign
        if t_s is None:
            t_s = self.t_s

        if t_campaign < 0 or t_s <= 0:
            raise ValueError("Both t_campaign and t_s must be positive.")

        if abs(t_campaign / t_s - round(t_campaign / t_s)) > 1e-8:
            raise ValueError(
                f"t_s = {t_s} must evenly divide t_campaign = {t_campaign}"
            )

        # --- Apply impulse control (instantaneous) ---
        effective_control = control_action * (1 - self.control_resistance)
        controlled_state = (
            effective_control * self.desired_opinion_vector
            + (1 - effective_control) * current_state
        )

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

    def reward_function(self, x_prev, x_next, u_original, d, beta, done: bool = False):
        # absolute term on resulting state
        abs_term = -np.abs(d - x_next).sum()

        # optional delta shaping (progress this step)
        if self.use_delta_shaping and self.delta_lambda != 0.0:
            delta = np.abs(d - x_prev).sum() - np.abs(d - x_next).sum()
            shaped = abs_term + self.delta_lambda * delta
        else:
            shaped = abs_term

        raw_reward = shaped - beta * np.sum(u_original)

        reward = (
            raw_reward / self.num_agents if self.normalize_reward else float(raw_reward)
        )
        if done:
            reward += self.terminal_reward
        return float(reward)

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

        x_prev = self.opinions.copy()
        original_action = np.array(action, copy=True)
        clipped_action = np.clip(original_action, 0, self.max_u)

        x_next, intermediate_states = self.compute_dynamics(
            x_prev, clipped_action, t_campaign, t_s
        )

        self.total_spent += np.sum(original_action)
        self.current_step += 1

        done = (
            self.terminate_when_converged
            and np.abs(np.mean(x_next) - self.desired_opinion)
            <= self.opinion_end_tolerance
        )
        truncated = self.current_step >= self.max_steps
        terminal_success = done and not truncated

        reward = self.reward_function(
            x_prev,
            x_next,
            original_action,
            self.desired_opinion,
            self.control_beta,
            done=terminal_success,
        )

        self.opinions = x_next

        info = {
            "current_step": self.current_step,
            "total_spent": self.total_spent,
            "action_applied_raw": original_action,
            "action_applied_clipped": clipped_action,
            "terminal_reward_applied": terminal_success,
            "intermediate_states": intermediate_states,  # Shape: (num_substeps+1, num_agents)
        }

        if self.budget is not None:
            info["remaining_budget"] = self.budget - self.total_spent

        return self.opinions, reward, done, truncated, info

    def render(self, mode="matplotlib", **kwargs):
        """
        Render the environment.

        Args:
            mode (str): "human", "matplotlib", "centralities"
            **kwargs: forwarded to the plotting functions.

        Examples:
            env.render(mode="matplotlib", layout_try=0)
            env.render(mode="matplotlib", layout_try=3)
            env.render(mode="matplotlib", randomize_layout=True)

            env.render(mode="centralities")  # kwargs ignored unless you add support there too
        """
        if mode == "matplotlib":
            # Forward kwargs so you can do layout retries without changing the env seed/graph
            fig = draw_network_graph(
                self.connectivity_matrix, self.centralities, **kwargs
            )

        elif mode == "centralities":
            fig = plot_centralities_sorted(
                self.centralities
            )  # (you can also forward **kwargs if you want)

        elif mode == "human":
            print(
                f"Step: {self.current_step}, Opinions: {self.opinions}, Total Spent: {self.total_spent}"
            )

        else:
            raise NotImplementedError(f"Render mode '{mode}' is not supported.")

        return fig

    def close(self):
        """
        Clean up the environment.
        """
        pass
