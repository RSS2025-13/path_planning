import numpy as np
import random
import math
import time

class rrt:
    class Node:
        def __init__(self, p):
            self.p = np.array(p)
            self.parent = None

    def __init__(self, start, goal, obstacle_list, x_bounds, y_bounds, max_extend_length = 3.0, path_resolution = 0.5, goal_sample_rate = 0.05, max_iter = 1000):
        self.start = self.Node(start)
        self.goal = self.Node(goal)
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.max_extend_length = max_extend_length
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

    def plan(self):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_node = self.get_nearest_node(self.node_list, rnd_node)
            new = self.steer(nearest_node, rnd_node, max_extend_length = self.max_extend_length)
            if not self.collision(nearest_node, new, self.obstacle_list):
                self.node_list.append(new)

            if self.dist_to_goal(self.node_list[-1].p) <= self.max_extend_length:
                final_node = self.steer(self.node_list[-1], self.goal, max_extend_length = self.max_extend_length)
                if not self.collision(final_node, self.node_list[-1], self.obstacle_list):
                    return self.final_path(len(self.node_list) - 1)
                
        return None
    
    def steer(self, from_node, to_node, max_extend_length = np.inf):
        new_node = self.Node(to_node.p)
        d = from_node.p - to_node.p
        dist = np.linalg.norm(d)
        if dist > max_extend_length:
            new_node.p = from_node.p - d * (max_extend_length / dist)
        new_node.parent = from_node
        return new_node
    
    def dist_to_goal(self, p):
        return np.linalg.norm(p - self.goal.p)
    
    def get_random_node(self):
        if np.random.rand() > self.goal_sample_rate:
            return self.Node([np.random.rand() * (self.x_bounds[1] - self.x_bounds[0]) + self.x_bounds[0], np.random.rand() * (self.y_bounds[1] - self.y_bounds[0]) + self.y_bounds[0],])
        else:
            return self.Node(self.goal.p)
    
    @staticmethod
    def get_nearest_node(node_list, node):
        dist_list = [np.sum(np.square((node.p - n.p))) for n in node_list]
        min_dist_idx = dist_list.index(min(dist_list))
        return node_list[min_dist_idx]
    
    @staticmethod
    def collision(node1, node2, obstacle_list):
        p1 = node2.p
        p2 = node1.p

        for o in obstacle_list:
            center_circle = o[0:2]
            radius = o[2]
            d12 = p2 - p1
            d1c = center_circle - p1
            t = d12.dot(d1c) / (d12.dot(d12) + 1e-6)
            t = max(0, min(1, t))
            d = p1 + d12 * t
            is_collide = np.sum(np.square(d - center_circle)) <= np.square(radius)
            if is_collide:
                return True
        return False
    

    def final_path(self, goal_ind):
        path = [self.goal.p]
        node = self.node_list[goal_ind]
        while node != self.start:
            path.append(node.p)
            node = node.parent
        path.append(node.p)
        return path
    
class rrt_star(rrt):
    class Node(rrt.Node):
        def __init__(self, p):
            super().__init__(p)
            self.cost = 0.0
    
    def __init__(self, start, goal, obstacle_list, x_bounds, y_bounds, max_extend_length = 3.0, path_resolution = 0.5, goal_sample_rate = 0.05, max_iter = 1000, connect_circle_dist = 50.0):
        super().__init__(start, goal, obstacle_list, x_bounds, y_bounds, max_extend_length, path_resolution, goal_sample_rate, max_iter)
        self.connect_circle_dist = connect_circle_dist
        self.goal = self.Node(goal)
        # Precompute obstacle information for faster collision checking
        self.obstacle_centers = np.array([o[0:2] for o in obstacle_list])
        self.obstacle_radii = np.array([o[2] for o in obstacle_list])
        # Increase goal sampling rate for faster convergence
        self.goal_sample_rate = 0.1
        # Reduce max iterations for faster planning
        self.max_iter = 500
        # Add early termination if we find a good path
        self.early_termination_cost = 1.5  # Terminate if we find a path within 1.5x the straight-line distance
        self.node = None  # Will be set by the planner

    def plan(self):
        start_time = time.time()
        
        self.node_list = [self.start]
        straight_line_dist = np.linalg.norm(self.start.p - self.goal.p)
        best_cost = float('inf')
        best_path = None
        
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_node = self.get_nearest_node(self.node_list, rnd_node)
            new = self.steer(nearest_node, rnd_node, max_extend_length = self.max_extend_length)
            if not self.collision(new, nearest_node, self.obstacle_list):
                near_inds = self.near_nodes_inds(new)
                new = self.choose_parent(new, near_inds)
                self.node_list.append(new)
                self.rewire(new, near_inds)
                
                # Check if we can connect to goal
                if self.dist_to_goal(new.p) <= self.max_extend_length:
                    if not self.collision(new, self.goal, self.obstacle_list):
                        # Calculate path cost
                        path_cost = new.cost + self.dist_to_goal(new.p)
                        if path_cost < best_cost:
                            best_cost = path_cost
                            best_path = self.final_path(len(self.node_list) - 1)
                            
                            # Early termination if we found a good path
                            if path_cost < straight_line_dist * self.early_termination_cost:
                                planning_time = time.time() - start_time
                                path_length = sum(np.linalg.norm(np.array(best_path[i]) - np.array(best_path[i-1])) for i in range(1, len(best_path)))
                                if hasattr(self, 'node') and self.node:
                                    self.node.get_logger().info(f"RRT* planning took {planning_time:.3f} seconds")
                                    self.node.get_logger().info(f"RRT* path length: {path_length:.2f} meters")
                                return best_path, best_cost

        # If we didn't find a path through early termination, return the best path found
        planning_time = time.time() - start_time
        if best_path is not None:
            path_length = sum(np.linalg.norm(np.array(best_path[i]) - np.array(best_path[i-1])) for i in range(1, len(best_path)))
            if hasattr(self, 'node') and self.node:
                self.node.get_logger().info(f"RRT* planning took {planning_time:.3f} seconds")
                self.node.get_logger().info(f"RRT* path length: {path_length:.2f} meters")
            return best_path, best_cost
        if hasattr(self, 'node') and self.node:
            self.node.get_logger().info(f"RRT* planning took {planning_time:.3f} seconds")
            self.node.get_logger().error("RRT* found no path")
        return None, float('inf')
    
    def choose_parent(self, new_node, near_inds):
        min_cost = np.inf
        best_near_node = None
        for i in near_inds:
            node = self.node_list[i]
            if not self.collision(new_node, node, self.obstacle_list):
                new_cost = self.new_cost(node, new_node)
                if new_cost < min_cost:
                    min_cost = new_cost
                    best_near_node = node
        new_node.cost = min_cost
        new_node.parent = best_near_node
        return new_node
    
    def rewire(self, new_node, near_inds):
        for i in near_inds:
            node = self.node_list[i]
            new_cost = self.new_cost(new_node, node)
            if not self.collision(node, new_node, self.obstacle_list) and new_cost < node.cost:
                node.parent = new_node
                node.cost = new_cost
        self.propagate_cost_to_leaves(new_node)

    def best_goal_node_index(self):
        min_cost = np.inf
        best_goal_node_idx = None
        for i in range(len(self.node_list)):
            node = self.node_list[i]
            if self.dist_to_goal(node.p) <= self.max_extend_length:
                if not self.collision(self.goal, node, self.obstacle_list):
                    cost = node.cost + self.dist_to_goal(node.p)
                    if cost < min_cost:
                        min_cost = cost
                        best_goal_node_idx = i
        return best_goal_node_idx, min_cost
    
    def near_nodes_inds(self, new_node):
        # Limit the number of nodes to check for rewiring
        nnode = len(self.node_list) + 1
        r = min(self.connect_circle_dist * np.sqrt(np.log(nnode) / nnode), 5.0)  # Cap the search radius
        dist_list = [np.sum(np.square((new_node.p - n.p))) for n in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        # Limit the number of nodes to rewire
        if len(near_inds) > 10:
            near_inds = sorted(near_inds, key=lambda i: dist_list[i])[:10]
        return near_inds
    
    def new_cost(self, from_node, to_node):
        d = np.linalg.norm(from_node.p - to_node.p)
        return from_node.cost + d
    
    def propagate_cost_to_leaves(self, parent_node):
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)
    
    def collision(self, node1, node2, obstacle_list):
        # Optimized collision checking using vectorized operations
        p1 = node1.p
        p2 = node2.p
        
        # Vectorized calculation for all obstacles at once
        d12 = p2 - p1
        d12_norm_squared = np.sum(np.square(d12))
        
        # Avoid division by zero
        if d12_norm_squared < 1e-10:
            return False
            
        # Calculate distance from line segment to each obstacle center
        d1c = self.obstacle_centers - p1
        t = np.clip(np.sum(d12 * d1c, axis=1) / d12_norm_squared, 0, 1)
        
        # Calculate closest point on line segment to each obstacle center
        closest_points = p1 + np.outer(t, d12)
        
        # Calculate squared distances from closest points to obstacle centers
        dist_squared = np.sum(np.square(closest_points - self.obstacle_centers), axis=1)
        
        # Check if any obstacle is closer than its radius
        return np.any(dist_squared <= np.square(self.obstacle_radii))
    
    
