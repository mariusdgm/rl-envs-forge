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
    ):
        super(NetworkGraph, self).__init__()

        # Store the initialization parameters
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

        # Check if conflicting initialization methods are used
        if (
            connectivity_matrix is not None or desired_centrality is not None
        ) and custom_adjacency_matrix is not None:
            raise ValueError(
                "Cannot provide both a custom_adjacency_matrix and a connectivity_matrix with desired_centrality. "
                "Please choose one method of initialization."
            )

        # Check if connectivity_matrix and desired_centrality are provided
        if connectivity_matrix is not None and desired_centrality is not None:
            self.adjacency_matrix, _ = inverse_centrality(
                connectivity_matrix, desired_centrality
            )
            self.num_agents = connectivity_matrix.shape[0]
        elif custom_adjacency_matrix is not None:
            # Use the provided custom adjacency matrix
            self.adjacency_matrix = custom_adjacency_matrix
            self.num_agents = custom_adjacency_matrix.shape[0]
        else:
            # Generate a random adjacency matrix using Erdos-Renyi graph
            self.graph = nx.erdos_renyi_graph(
                num_agents, np.random.uniform(*connection_prob_range)
            )
            self.adjacency_matrix = nx.to_numpy_array(self.graph)

        # Check if the graph is fully connected
        G = nx.from_numpy_array(self.adjacency_matrix)
        if not nx.is_connected(G):
            print("Warning: The generated graph is not fully connected.")

        # Only proceed if there are actual edges in the graph
        if np.sum(self.adjacency_matrix) > 0:
            # Compute the Laplacian matrix using the shared utility function
            self.L = compute_laplacian(
                np.ones(int(np.sum(self.adjacency_matrix[self.adjacency_matrix == 1]))), self.adjacency_matrix
            )

            # Compute centralities based on the Laplacian matrix
            self.centralities = compute_centrality_with_isolated(self.L)
        else:
            # Handle the case of a completely disconnected graph
            self.L = np.zeros((self.num_agents, self.num_agents))
            self.centralities = np.zeros(self.num_agents)

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
            self.impulse_resistance = np.ones(
                self.num_agents
            )  # Default resistance is 1 for all agents

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
            self.render_matplotlib()
        elif mode == "human":
            self.render_human()
        else:
            raise NotImplementedError(f"Render mode '{mode}' is not supported.")

    def render_human(self):
        """
        Render the environment using print statements.
        """
        print(
            f"Step: {self.current_step}, Opinions: {self.opinions}, Total Spent: {self.total_spent}"
        )

    def render_matplotlib(self):
        """
        Render the environment using matplotlib.
        """
        draw_network_graph(self.adjacency_matrix, self.centralities)

    def close(self):
        """
        Clean up the environment.
        """
        pass
