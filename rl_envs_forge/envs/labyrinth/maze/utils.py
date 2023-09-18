import numpy as np
from ..constants import *




def generate_ellipse(
    rows, cols, a, b, center_x, center_y, value_true=PATH, value_false=WALL
):
    """
    Generate an ellipse shape in a 2D grid.

    Parameters:
        rows (int): The number of rows in the grid.
        cols (int): The number of columns in the grid.
        a (int): The horizontal radius of the ellipse.
        b (int): The vertical radius of the ellipse.
        center_x (int): The x-axis offset of the ellipse.
        center_y (int): The y-axis offset of the ellipse.

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


def clamp(value, min_value, max_value):
    """Clamp a value between a minimum and a maximum."""
    return max(min_value, min(value, max_value))
