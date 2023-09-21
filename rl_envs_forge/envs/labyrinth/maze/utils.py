import numpy as np
from ..constants import *


def generate_ellipse(
    rows: int,
    cols: int,
    a: int,
    b: int,
    center_x: int,
    center_y: int,
    value_true: float = PATH,
    value_false: float = WALL,
) -> np.ndarray:
    """
    Generate an ellipse shape in a 2D grid.

    Parameters:
        rows (int): The number of rows in the grid.
        cols (int): The number of columns in the grid.
        a (int): The horizontal radius of the ellipse.
        b (int): The vertical radius of the ellipse.
        center_x (int): The x-axis offset of the ellipse.
        center_y (int): The y-axis offset of the ellipse.
        value_true (float): The value to assign to the center of the ellipse.
        value_false (float): The value to assign to the outside of the ellipse.

    Returns:
        np.ndarray: An array representing the generated ellipse shape in the grid.
    """
    ellipse = np.ones((rows, cols), dtype=int) * value_false

    for i in range(rows):
        for j in range(cols):
            y, x = i - center_y, j - center_x

            if (x**2 / a**2) + (y**2 / b**2) <= 1:
                ellipse[i][j] = value_true

    return ellipse


def clamp(value, min_value: float, max_value: float) -> float:
    """
    Clamp a value between a minimum and maximum value.

    Parameters:
        value (float): The value to be clamped.
        min_value (float): The minimum value that the value parameter can be.
        max_value (float): The maximum value that the value parameter can be.

    Returns:
        float: The clamped value.

    """
    return max(min_value, min(value, max_value))
