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