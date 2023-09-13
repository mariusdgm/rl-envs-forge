import numpy as np
import random
from collections import deque
from queue import PriorityQueue
from typing import List, Tuple, Union, Optional

from ..constants import WALL, PATH, START, TARGET, CorridorMoveStatus, CorridorCosts

from ...common.grid_functions import on_line


class CustomPriorityQueue:
    """Custom Priority Queue to allow for keeping the insertion order consistent."""

    def __init__(self):
        self.q = PriorityQueue()
        self.counter = 0

    def put(self, priority, item):
        count = self.counter
        self.counter += 1
        self.q.put((priority, count, item))

    def get(self):
        return self.q.get()[2]

    def empty(self):
        return self.q.empty()

    @property
    def queue(self):
        return self.q.queue


class CorridorBuilder:
    def __init__(self, maze):
        self.maze = maze
        self.rows = self.maze.rows
        self.cols = self.maze.cols

    ########## Prim's algorithm logic ########################
    def generate_corridor_prim(self):
        """Generate corridors using an adaptation of Prim's algorithm.

        Builds the corridor around the rooms, without touching the rooms, so that
        the access points of the rooms need to be connected to the corridors.
        """
        grid = np.full((self.rows, self.cols), WALL)
        directions = [(2, 0), (-2, 0), (0, 2), (0, -2)]

        # Initialize the starting position
        current_position = self.maze.start_position
        grid[current_position] = PATH

        walls = []

        for d in directions:
            pos_valid, move_status = self.is_valid_next_position(
                grid, current_position, d
            )
            if pos_valid or move_status in [
                CorridorMoveStatus.ROOM_BOUNDARY,
                CorridorMoveStatus.MAZE_BOUNDARY,
            ]:
                walls.append((current_position, d))

        while walls:
            # Randomly select a wall from the list
            current_position, direction = self.maze.py_random.choice(walls)

            next_pos = (
                current_position[0] + direction[0],
                current_position[1] + direction[1],
            )

            intermediary_pos = (
                current_position[0] + direction[0] // 2,
                current_position[1] + direction[1] // 2,
            )

            is_valid, status = self.is_valid_next_position(
                grid, current_position, direction
            )

            if is_valid:
                grid[intermediary_pos[0], intermediary_pos[1]] = PATH
                grid[next_pos[0], next_pos[1]] = PATH

                for d in directions:
                    pos_valid, move_status = self.is_valid_next_position(
                        grid, next_pos, d
                    )
                    if pos_valid:
                        walls.append((next_pos, d))

            if not is_valid and self.maze.corridor_grid_connect_option:
                if (
                    0 <= intermediary_pos[0] < self.rows
                    and 0 <= intermediary_pos[1] < self.cols
                    and grid[intermediary_pos[0], intermediary_pos[1]] == WALL
                    and not self.is_adjacent_to_room(intermediary_pos)
                ):
                    grid[intermediary_pos[0], intermediary_pos[1]] = PATH

            # Remove the wall from the list
            walls.remove((current_position, direction))

        return grid

    def is_out_of_bounds_maze(self, position):
        if (
            position[0] < 0
            or position[0] >= self.rows
            or position[1] < 0
            or position[1] >= self.cols
        ):
            return True
        return False

    def is_valid_next_position(self, grid, position, direction):
        new_pos = (position[0] + direction[0], position[1] + direction[1])

        # Detect if new position is out of maze bounds
        if self.is_out_of_bounds_maze(new_pos):
            return (False, CorridorMoveStatus.MAZE_BOUNDARY)

        # Detect if new position or its neighbors are a path (either corridor or room)
        if (
            grid[new_pos[0], new_pos[1]] == PATH
            or self.maze.room_grid[new_pos[0], new_pos[1]] == PATH
        ):
            return (False, CorridorMoveStatus.INVALID)

        # Check for room padding
        if self.is_adjacent_to_room(new_pos):
            return (False, CorridorMoveStatus.ROOM_BOUNDARY)

        return (True, CorridorMoveStatus.VALID_MOVE)

    def is_adjacent_to_room(self, position):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if (
                    0 <= position[0] + dx < self.rows
                    and 0 <= position[1] + dy < self.cols
                    and self.maze.room_grid[position[0] + dx, position[1] + dy] == PATH
                ):
                    return True
        return False

    def greedy_sort(self, access_points, start_position):
        # Initialize the sorted list with the start_position
        sorted_points = [start_position]

        # Remaining points to be sorted
        remaining_points = access_points.copy()

        while remaining_points:
            current_point = sorted_points[-1]  # Get the last added point
            closest_point = min(
                remaining_points, key=lambda point: self.heuristic(current_point, point)
            )

            sorted_points.append(closest_point)
            remaining_points.remove(closest_point)

        # Return the sorted list without the start_position
        return sorted_points[1:]

    ####### Connect access points to paths ############

    def connect_rooms_to_paths(self):
        """Connect the rooms to the corridors via the shortest path."""

        for room in self.maze.rooms:
            for access_point in room.access_points:
                global_access_point = (
                    room.global_position[0] + access_point[0],
                    room.global_position[1] + access_point[1],
                )
                connection = self.bfs_to_find_closest_path(global_access_point)

                if not connection:
                    # If BFS failed to find a connection, try growing a path
                    connection = self.grow_path_from(global_access_point)

                # If we found a connection, draw a path to it
                if connection:
                    self.plot_path_from_to(global_access_point, connection)

    def bfs_to_find_closest_path(self, access_point):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        visited = np.zeros((self.rows, self.cols), dtype=bool)
        queue = deque([access_point])

        # If the access point is already on a corridor path
        if self.maze.corridor_grid[access_point[0], access_point[1]] == PATH:
            return access_point

        while queue:
            current = queue.popleft()

            for d in directions:
                new_pos = (current[0] + d[0], current[1] + d[1])

                # Check boundaries
                if 0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols:
                    # If the new position is an existing path, we found a connection
                    if self.maze.corridor_grid[new_pos[0], new_pos[1]] == PATH:
                        return new_pos

                    # If not visited, not a room cell, not inside any room (excluding the access point), and not yet part of the corridor
                    if (
                        not visited[new_pos[0], new_pos[1]]
                        and not self.maze.is_inside_any_room(
                            new_pos, exception=access_point
                        )
                        and self.maze.corridor_grid[new_pos[0], new_pos[1]] == WALL
                    ):
                        visited[new_pos[0], new_pos[1]] = True
                        queue.append(new_pos)

        return None

    def plot_path_from_to(self, p1, p2):
        # Directly connect if the cells are adjacent and not diagonally so
        diff_x = abs(p1[0] - p2[0])
        diff_y = abs(p1[1] - p2[1])
        if diff_x <= 1 and diff_y <= 1 and not (diff_x == 1 and diff_y == 1):
            self.maze.corridor_grid[p2[0], p2[1]] = PATH
            return

        while p1 != p2:
            # First adjust horizontally
            if p1[1] != p2[1]:
                p1 = (p1[0], p1[1] + np.sign(p2[1] - p1[1]))
            # Then adjust vertically
            else:
                p1 = (p1[0] + np.sign(p2[0] - p1[0]), p1[1])

            if not self.maze.is_inside_any_room(p1, exception=None):
                self.maze.corridor_grid[p1[0], p1[1]] = PATH

    ####### Grow path logic #########
    def grow_path_from(
        self, start_pos, max_attempts=5000
    ):  # Temporarily increased max_attempts
        """Grow a path from start_pos until it reaches a corridor or max_attempts are reached."""

        current_pos = start_pos
        attempt = 0
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        visited = set()
        visited.add(start_pos)

        # Get a list of all corridor positions
        corridor_positions = np.argwhere(self.maze.corridor_grid == PATH)

        while attempt < max_attempts:
            potential_directions = []

            # Get the closest corridor position to the current position
            distances = [
                np.linalg.norm(np.array(current_pos) - np.array(corridor_pos))
                for corridor_pos in corridor_positions
            ]
            closest_corridor_pos = corridor_positions[np.argmin(distances)]

            for d in directions:
                new_pos = (current_pos[0] + d[0], current_pos[1] + d[1])

                if new_pos in visited:
                    continue

                # Check boundaries
                if not (0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols):
                    continue  # Go to the next direction if this is out of bounds

                # If the new position is an existing path, validate and then return
                if self.maze.corridor_grid[new_pos[0], new_pos[1]] == PATH:
                    diff_x = new_pos[0] - current_pos[0]
                    diff_y = new_pos[1] - current_pos[1]

                    # If this is not diagonal, then validate
                    if not (abs(diff_x) == 1 and abs(diff_y) == 1):
                        if not self.is_line_segment_intersecting_room(
                            start_pos, new_pos
                        ):
                            return new_pos
                    continue  # Go to the next direction

                # If not inside any room and not yet part of the corridor
                if (
                    not self.maze.is_inside_any_room(new_pos)
                    and self.maze.corridor_grid[new_pos[0], new_pos[1]] == WALL
                ):
                    potential_directions.append(
                        (
                            new_pos,
                            self.direction_cost(current_pos, d, closest_corridor_pos),
                        )
                    )

            # Sort the potential directions by their cost
            potential_directions.sort(key=lambda x: x[1])

            if not potential_directions:
                break

            # Select the best direction
            next_pos = potential_directions[0][0]
            visited.add(next_pos)

            # Update the grid
            self.maze.corridor_grid[next_pos[0], next_pos[1]] = PATH
            current_pos = next_pos
            attempt += 1

        # If we reached here, we didn't find a corridor
        return None

    def direction_cost(self, current_pos, direction, target_pos):
        new_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
        base_cost = abs(new_pos[0] - target_pos[0]) + abs(new_pos[1] - target_pos[1])

        # Penalize if the new position is close to a room.
        if self.maze.is_inside_any_room(new_pos):
            base_cost += 10

        return base_cost

    def is_line_segment_intersecting_room(self, p1, p2):
        """Check if the line segment p1p2 intersects any room cell."""
        for room in self.maze.rooms:
            for i in range(room.rows):
                for j in range(room.cols):
                    if room.grid[i][j] == PATH:
                        global_cell_pos = (
                            room.global_position[0] + i,
                            room.global_position[1] + j,
                        )

                        # Skip if it's the starting point
                        if global_cell_pos == p1:
                            continue

                        if on_line(p1, global_cell_pos, p2):
                            return True
        return False

    ###### Post Processing logic ######

    def post_process_maze(self):
        # Initialize the global room mask
        global_room_mask = self.maze.generate_global_room_mask()

        # Collect border cells
        border_cells = []

        for i in range(self.rows):
            for j in [0, self.cols - 1]:  # Left and right borders
                border_cells.append((i, j))

        for j in range(self.cols):
            for i in [0, self.rows - 1]:  # Top and bottom borders
                border_cells.append((i, j))

        self.maze.py_random.shuffle(border_cells)  # Shuffle border cells

        for cell in border_cells:
            self._attempt_fill_path(cell, global_room_mask)

        # Collect the perimeter cells of all rooms
        room_perimeter_cells = []

        for room in self.maze.rooms:
            perimeter_cells = room.get_perimeter_cells(padding=2)
            for cell in perimeter_cells:
                # Translate to global maze coordinates
                global_pos = (
                    cell[0] + room.global_position[0],
                    cell[1] + room.global_position[1],
                )
                room_perimeter_cells.append(global_pos)

        self.maze.py_random.shuffle(room_perimeter_cells)  # Shuffle perimeter cells

        for cell in room_perimeter_cells:
            self._attempt_fill_path(cell, global_room_mask)

    def _attempt_fill_path(self, pos, global_room_mask):
        if pos[0] < 0 or pos[0] >= self.rows or pos[1] < 0 or pos[1] >= self.cols:
            return

        adjacent_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        all_directions = adjacent_dirs + [(1, 1), (-1, 1), (-1, -1), (1, -1)]

        path_count = sum(
            [
                1
                for dx, dy in adjacent_dirs
                if 0 <= pos[0] + dx < self.rows
                and 0 <= pos[1] + dy < self.cols
                and self.maze.corridor_grid[pos[0] + dx, pos[1] + dy] == PATH
            ]
        )

        inside_or_adjacent_to_room = False
        for dx, dy in all_directions:
            new_pos = (pos[0] + dx, pos[1] + dy)
            if 0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols:
                if global_room_mask[new_pos[0], new_pos[1]]:
                    inside_or_adjacent_to_room = True
                    break

        if not inside_or_adjacent_to_room and path_count == 1:
            self.maze.corridor_grid[pos[0], pos[1]] = PATH

    ####### Greedy best first search logic #########
    def generate_corridor_gbfs(self):
        """Generate corridors using a slight variation of greedy best first search algorithm.
        Instead of using only the heuristic we are also considering the cost matrix.
        """

        # Initialize a grid filled with walls
        corridor_grid = np.full((self.rows, self.cols), WALL)
        current_endpoint = self.maze.start_position

        access_points = self.get_global_access_points(current_endpoint)

        cost_grid = self.make_cost_grid(access_points, corridor_grid)

        for access_point in access_points:
            path = self.gbfs(current_endpoint, access_point, cost_grid)
            for point in path:
                corridor_grid[point[0], point[1]] = PATH
            current_endpoint = access_point

            # Recompute grid with the newly generated coridors
            cost_grid = self.make_cost_grid(access_points, corridor_grid)

        return corridor_grid

    def gbfs(self, start, goal, cost_grid):
        frontier = CustomPriorityQueue()
        frontier.put(0, start)

        came_from = {}
        came_from[start] = None

        while not frontier.empty():
            current = frontier.get()
           
            if current == goal:
                break

            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            self.maze.py_random.shuffle(directions)

            for dx, dy in directions:
                next_node = (current[0] + dx, current[1] + dy)

                if self.is_out_of_bounds_maze(next_node):
                    continue

                if next_node not in came_from:
                    priority = self.heuristic(goal, next_node, cost_grid)
                    frontier.put(priority, next_node)
                    came_from[next_node] = current

        path = self.reconstruct_path(came_from, start, goal)

        return path

    ####### A* direct corridor generation logic #########
    def generate_corridor_a_star(self):
        """Generate corridors using the A* algorithm."""

        # Initialize a grid filled with walls
        corridor_grid = np.full((self.rows, self.cols), WALL)
        current_endpoint = self.maze.start_position

        access_points = self.get_global_access_points(current_endpoint)

        cost_grid = self.make_cost_grid(access_points, corridor_grid)

        for access_point in access_points:
            path = self.a_star_path(current_endpoint, access_point, cost_grid)
            for point in path:
                corridor_grid[point[0], point[1]] = PATH
            current_endpoint = access_point

            # Recompute grid with the newly generated coridors
            cost_grid = self.make_cost_grid(access_points, corridor_grid)

        return corridor_grid

    def a_star_path(self, start, goal, cost_grid):
        frontier = CustomPriorityQueue()
        frontier.put(0, start)

        came_from = {}
        came_from[start] = None
        cost_so_far = {}
        cost_so_far[start] = 0

        visited = set()
        visited.add(start)

        while not frontier.empty():
            current = frontier.get()

            if current == goal:
                break

            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                next_node = (current[0] + dx, current[1] + dy)

                if next_node in visited:
                    continue

                if self.is_out_of_bounds_maze(next_node):
                    continue

                cell_cost = cost_grid[next_node[0], next_node[1]]
                new_cost = cost_so_far[current] + cell_cost
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self.heuristic(goal, next_node)
                    frontier.put(priority, next_node)
                    came_from[next_node] = current

        path = self.reconstruct_path(came_from, start, goal)

        return path

    ### Common astar and gbfs helper functions ###
    def get_global_access_points(self, current_endpoint):
        """
        Retrieves the access points in the rooms of the maze in global coordinates.
        Sorts the access points by heuristic to get even more direct paths.

        Parameters:
            current_endpoint (Endpoint): The current endpoint in the maze.

        Returns:
            list: A list of global access points in the maze.
        """

        access_points = []
        for room in self.maze.rooms:
            access_points_global = [
                (x + room.global_position[0], y + room.global_position[1])
                for x, y in room.access_points
            ]
            access_points.extend(access_points_global)

        # Sort the access points to get even more direct paths
        if self.maze.corridor_sort_access_points_option:
            access_points = self.greedy_sort(access_points, current_endpoint)
        else:
            self.maze.py_random.shuffle(access_points)

        return access_points

    def make_cost_grid(self, access_points, corridor_grid):
        cost_grid = np.full((self.rows, self.cols), CorridorCosts.BASE)

        global_room_mask = self.maze.generate_global_room_mask()
        cost_grid[global_room_mask] = CorridorCosts.ROOM_CELL

        for room in self.maze.rooms:
            global_pos = room.global_position
            perimeter_cells = room.get_perimeter_cells(padding=1)
            perimeter_cells_global = [
                (x + global_pos[0], y + global_pos[1]) for x, y in perimeter_cells
            ]
            for cell in perimeter_cells_global:
                if self.is_next_to_access_point(cell, access_points):
                    cost_grid[cell[0], cell[1]] = CorridorCosts.ADJACENT_ACCESS_POINT
                else:
                    cost_grid[cell[0], cell[1]] = CorridorCosts.ADJACENT_ROOM

            for access_point in access_points:
                cost_grid[access_point[0], access_point[1]] = CorridorCosts.ACCESS_POINT

        cost_grid[corridor_grid == PATH] = CorridorCosts.CORRIDOR

        return cost_grid

    def reconstruct_path(self, came_from, start, goal):
        current = goal
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()

        # Remove goal because the path joins it to the corridor
        path.remove(goal)

        return path

    def is_next_to_access_point(self, position, access_points=None):
        """Checks if the given position is adjacent to any of the room's access points."""
        if access_points is None:
            access_points = []
            for room in self.maze.rooms:
                access_points_global = [
                    (x + room.global_position[0], y + room.global_position[1])
                    for x, y in room.access_points
                ]
                access_points.extend(access_points_global)

        for access_point in access_points:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                if position == (access_point[0] + dx, access_point[1] + dy):
                    return True
        return False

    def heuristic(self, goal, next_node, cost_grid=None):
        """Manhattan distance heuristic."""
        heuristic = abs(goal[0] - next_node[0]) + abs(goal[1] - next_node[1])
        if cost_grid is not None:
            heuristic += cost_grid[next_node[0], next_node[1]]
        return heuristic
