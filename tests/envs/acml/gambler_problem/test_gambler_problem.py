import pytest
import numbers
from rl_envs_forge.envs.acml.gambler_problem.gambler_problem import GamblersProblem

@pytest.fixture
def fixture_env():
    """Fixture to create a default Jack's Car Rental environment for each test."""
    return GamblersProblem(
        goal_amount=100, win_probability=0.40, start_capital=10
    )


class TestGamblersProblem:
    def test_initialization(self, fixture_env):
        assert fixture_env.state == 10


    def test_reset(self, fixture_env):
        fixture_env.step(5)
        state, _ = fixture_env.reset()
        assert isinstance(state, numbers.Number)

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
