import toml
import os

# Define the path to the pyproject.toml
current_init_file_path = os.path.dirname(__file__)
pyproject_path = os.path.join(current_init_file_path, "..", "pyproject.toml")

# Read the version from the pyproject.toml
with open(pyproject_path, "r") as f:
    pyproject_data = toml.load(f)
    __version__ = pyproject_data["tool"]["poetry"]["version"]