import pytest
from rl_envs_forge.envs.acml.car_rental.car_rental import JacksCarRental

@pytest.fixture
def fixture_env():
    """Fixture to create a default Jack's Car Rental environment for each test."""
    return JacksCarRental(
        max_cars=4,
        max_move_cars=2,
        request_lambda=[2, 2],
        return_lambda=[2, 2],
    )


class TestJacksCarRental:
    def test_initialization(self, fixture_env):
        assert fixture_env.max_cars == 4
        assert fixture_env.max_move_cars == 2
        assert isinstance(fixture_env.state, tuple)
        assert 0 <= fixture_env.state[0] <= fixture_env.max_cars
        assert 0 <= fixture_env.state[1] <= fixture_env.max_cars


    def test_reset(self, fixture_env):
        fixture_env.step(0)
        state, _ = fixture_env.reset()
        assert isinstance(state, tuple)


    def test_poisson_probabilities(self, fixture_env):
        # Test if the precomputed probabilities are correctly set
        assert 0 <= fixture_env._get_probability_of_rental_requests(0, 0) <= 1
        assert 0 <= fixture_env._get_probability_of_car_returns(0, 0) <= 1

    def test_mdp_build(self, fixture_env):
        t_r_dict = fixture_env.build_mdp()
        assert isinstance(t_r_dict, dict)  # MDP representation should be a dictionary
        assert len(t_r_dict) > 0  # MDP representation should not be empty

    def test_render(self, fixture_env, monkeypatch):
        """
        Test if the render function produces the expected output.
        """
        # Mock the print function to capture its output
        captured_output = []
        monkeypatch.setattr("builtins.print", lambda x: captured_output.append(x))
        fixture_env.render()
        assert len(captured_output) > 0  
