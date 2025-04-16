import math
import heapq
import numpy as np
import time

class a_star:
    def __init__(self, obstacles, start, goal):
        self.cell_size = 0.25  # Smaller cell size for better resolution
        self.grid = self.Grid(obstacles, self.cell_size)
        self.start = start
        self.goal = goal
        self.node = None  # Will be set by the planner
        print(f"Grid initialized with {len(obstacles)} obstacles")

    class Node:
        def __init__(self, x, y, obstacle=False):
            self.x = x
            self.y = y
            self.obstacle = obstacle
            self.parent = None
            self.g = float('inf')
            self.h = float('inf')

        def __lt__(self, other):
            return (self.g + self.h) < (other.g + other.h)
        
    class Grid:
        def __init__(self, obstacles, cell_size = 1):
            self.cell_size = cell_size
            # Expand boundaries slightly to ensure all obstacles can be placed
            self.width_min = -62.0
            self.width_max = 27.0
            self.height_min = -5.0
            self.height_max = 40.0
            width_range = np.linspace(self.width_min, self.width_max, int((self.width_max - self.width_min) / cell_size) + 1)
            height_range = np.linspace(self.height_min, self.height_max, int((self.height_max - self.height_min) / cell_size) + 1)
            self.nodes = {(x,y): a_star.Node(x,y) for x in width_range for y in height_range}
            self.place_obstacles(obstacles)

        def is_within_bounds(self, x, y):
            return (self.width_min <= x <= self.width_max and 
                    self.height_min <= y <= self.height_max)

        def place_obstacles(self, obstacles):
            obstacle_count = 0
            for obstacle in obstacles:
                x = obstacle[0]
                y = obstacle[1]
                rad = obstacle[2] if len(obstacle) > 2 else 0.75  # Use provided radius or default to 0.75
                x_min = x - rad
                x_max = x + rad
                y_min = y - rad
                y_max = y + rad
                
                # Round to nearest grid points
                cells_per_meter = 1 / self.cell_size
                x_min_rounded = math.floor(x_min * cells_per_meter) / cells_per_meter
                x_max_rounded = math.ceil(x_max * cells_per_meter) / cells_per_meter
                y_min_rounded = math.floor(y_min * cells_per_meter) / cells_per_meter
                y_max_rounded = math.ceil(y_max * cells_per_meter) / cells_per_meter
                
                # Ensure points are within bounds
                x_min_rounded = max(x_min_rounded, self.width_min)
                x_max_rounded = min(x_max_rounded, self.width_max)
                y_min_rounded = max(y_min_rounded, self.height_min)
                y_max_rounded = min(y_max_rounded, self.height_max)
                
                x_num_points = int((x_max_rounded - x_min_rounded) / self.cell_size) + 1
                y_num_points = int((y_max_rounded - y_min_rounded) / self.cell_size) + 1
                
                if x_num_points > 0 and y_num_points > 0:
                    x_locs = np.linspace(x_min_rounded, x_max_rounded, x_num_points)
                    y_locs = np.linspace(y_min_rounded, y_max_rounded, y_num_points)
                    
                    for x_loc in x_locs:
                        for y_loc in y_locs:
                            # Check if point is actually within the circular obstacle
                            if ((x_loc - x)**2 + (y_loc - y)**2 <= rad**2 and 
                                self.is_within_bounds(x_loc, y_loc) and 
                                (x_loc, y_loc) in self.nodes):
                                self.nodes[(x_loc, y_loc)].obstacle = True
                                obstacle_count += 1
            
            print(f"Placed {obstacle_count} obstacle cells in grid")

        def __getitem__(self, coordinates):
            x,y = coordinates
            return self.nodes[(x,y)]
        
        def get_neighbors(self, node):
            neighbors = []
            # Check all 8 directions
            directions = [
                (self.cell_size, 0),          # right
                (self.cell_size, self.cell_size),    # up-right
                (0, self.cell_size),          # up
                (-self.cell_size, self.cell_size),   # up-left
                (-self.cell_size, 0),         # left
                (-self.cell_size, -self.cell_size),  # down-left
                (0, -self.cell_size),         # down
                (self.cell_size, -self.cell_size),   # down-right
            ]

            for dx, dy in directions:
                new_x, new_y = node.x + dx, node.y + dy
                if self.is_within_bounds(new_x, new_y):
                    neighbor_coords = (new_x, new_y)
                    if neighbor_coords in self.nodes and not self.nodes[neighbor_coords].obstacle:
                        # For diagonal movements, check if both adjacent cells are free
                        if abs(dx) == self.cell_size and abs(dy) == self.cell_size:
                            # Check horizontal neighbor
                            h_coords = (node.x + dx, node.y)
                            # Check vertical neighbor
                            v_coords = (node.x, node.y + dy)
                            if (h_coords in self.nodes and not self.nodes[h_coords].obstacle and
                                v_coords in self.nodes and not self.nodes[v_coords].obstacle):
                                neighbors.append((self.nodes[neighbor_coords], 
                                               self.cell_size * math.sqrt(2)))
                        else:
                            neighbors.append((self.nodes[neighbor_coords], self.cell_size))

            return neighbors
        
        def get_node_from_loc(self, loc):
            x,y = loc
            half_cell = self.cell_size / 2
            for grid_loc in self.nodes:
                grid_x, grid_y = grid_loc
                if (grid_x - half_cell < x < grid_x + half_cell) and (grid_y - half_cell < y < grid_y + half_cell):
                    return self.nodes[grid_loc]
                
    def heuristic(self, node, goal):
        return math.sqrt((node.x - goal.x)**2 + (node.y - goal.y)**2)
    
    def reconstruct_path(self, current_node):
        path = []
        while current_node is not None:
            path.append((current_node.x, current_node.y))
            current_node = current_node.parent
        return path[::-1]
    
    def plan(self, start_loc, goal_loc):
        start_time = time.time()
        
        open_set = []
        closed_set = set()
        start = self.grid.get_node_from_loc(start_loc)
        goal = self.grid.get_node_from_loc(goal_loc)
        start.g = 0
        start.h = self.heuristic(start, goal)
        heapq.heappush(open_set, start)

        while open_set:
            current = heapq.heappop(open_set)
            if current == goal:
                path = self.reconstruct_path(current)
                planning_time = time.time() - start_time
                path_length = sum(np.linalg.norm(np.array(path[i]) - np.array(path[i-1])) for i in range(1, len(path)))
                if hasattr(self, 'node') and self.node:
                    self.node.get_logger().info(f"A* planning took {planning_time:.3f} seconds")
                    self.node.get_logger().info(f"A* path length: {path_length:.2f} meters")
                return path
            
            closed_set.add(current)

            for neighbor, cost in self.grid.get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g = current.g + cost
                if tentative_g < neighbor.g:
                    neighbor.parent = current
                    neighbor.g = tentative_g
                    neighbor.h = self.heuristic(neighbor, goal)
                    if neighbor not in open_set:
                        heapq.heappush(open_set, neighbor)
        
        planning_time = time.time() - start_time
        if hasattr(self, 'node') and self.node:
            self.node.get_logger().info(f"A* planning took {planning_time:.3f} seconds")
            self.node.get_logger().error("A* found no path")
        return None
        

        
            


