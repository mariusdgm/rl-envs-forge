import numpy as np
from ..constants import *

# Utility function to count the number of neighboring PATH cells for a given cell in a grid
def count_neighboring_path_cells(grid, cell):
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    count = 0
    
    rows, cols = grid.shape
    
    for d in directions:
        new_row, new_col = cell[0] + d[0], cell[1] + d[1]
        if 0 <= new_row < rows and 0 <= new_col < cols and grid[new_row][new_col] == PATH:
            count += 1
    return count

# Utility function to check if a cell is a true perimeter cell
def is_perimeter(grid, cell):
    neighbors = count_neighboring_path_cells(grid, cell)
    return 0 < neighbors < 4

def generate_ellipse(rows, cols, a, b, offset_y, offset_x, value_true=PATH, value_false=WALL):
        """
        Generate an ellipse shape in a 2D grid.

        Parameters:
            rows (int): The number of rows in the grid.
            cols (int): The number of columns in the grid.
            a (int): The horizontal radius of the ellipse.
            b (int): The vertical radius of the ellipse.
            offset_y (int): The y-axis offset of the ellipse.
            offset_x (int): The x-axis offset of the ellipse.

        Returns:
            np.ndarray: An array representing the generated ellipse shape in the grid.
        """
        ellipse = np.ones((rows, cols), dtype=int) * value_false

        for i in range(rows):
            for j in range(cols):
                y, x = i - offset_y, j - offset_x

                if (x**2 / a**2) + (y**2 / b**2) <= 1:
                    ellipse[i][j] = value_true

        return ellipse

def clamp(value, min_value, max_value):
    """Clamp a value between a minimum and a maximum."""
    return max(min_value, min(value, max_value))