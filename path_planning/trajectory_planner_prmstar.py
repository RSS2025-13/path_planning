import rclpy
from rclpy.node import Node
import numpy as np
from scipy.spatial import KDTree
import heapq
from typing import List, Tuple, Set
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose
from nav_msgs.msg import OccupancyGrid
from .utils import LineTrajectory
from scipy import ndimage
import matplotlib.pyplot as plt


class PRMStar(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose using PRM* algorithm.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

        # Subscribe to map, goal, and pose topics
        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_cb,
            1)

        self.goal_sub = self.create_subscription(
            PoseStamped,
            "/goal_pose",
            self.goal_cb,
            10
        )

        self.traj_pub = self.create_publisher(
            PoseArray,
            "/trajectory/current",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")
        
        # PRM* specific attributes
        self.n_samples = 1000  # Increased number of samples
        self.robot_radius = 0.2  # Robot radius in meters
        self.occupancy_grid = None
        self.dilated_grid = None  # New attribute for dilated obstacle grid
        self.map_resolution = None
        self.map_origin = None
        self.current_pose = None
        self.samples = []
        self.graph = {}
        self.kdtree = None

    def map_cb(self, msg: OccupancyGrid):
        """Process incoming occupancy grid map and dilate obstacles"""
        self.get_logger().info("Received new map")
        
        # Convert map to binary grid (0 = free, 1 = occupied)
        height, width = msg.info.height, msg.info.width
        self.occupancy_grid = np.array(msg.data).reshape(height, width)
        binary_grid = np.zeros_like(self.occupancy_grid)
        binary_grid[self.occupancy_grid > 50] = 1  # Occupied space
        binary_grid[self.occupancy_grid < 0] = 1   # Unknown space
        
        # Dilate obstacles
        robot_radius_pixels = int(self.robot_radius / msg.info.resolution)
        struct_element = ndimage.generate_binary_structure(2, 2)
        self.dilated_grid = ndimage.binary_dilation(
            binary_grid, 
            structure=struct_element,
            iterations=robot_radius_pixels
        )
        
        self.map_resolution = msg.info.resolution
        self.map_origin = msg.info.origin
        
        # Debug visualization
        # plt.imshow(self.dilated_grid, cmap='gray')
        # plt.show()

    def pose_cb(self, pose: PoseWithCovarianceStamped):
        """Store current pose of the robot"""
        self.current_pose = pose.pose.pose
        
    def goal_cb(self, msg: PoseStamped):
        """Handle new goal pose and plan path"""
        if self.current_pose is None:
            self.get_logger().warn("No current pose received yet")
            return
            
        if self.occupancy_grid is None:
            self.get_logger().warn("No map received yet")
            return
            
        start_point = (self.current_pose.position.x, self.current_pose.position.y)
        goal_point = (msg.pose.position.x, msg.pose.position.y)
        
        path = self.plan_path(start_point, goal_point)
        if path is not None:
            self.publish_path(path)
        else:
            self.get_logger().warn("No path found!")

    def world_to_grid(self, point: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to grid coordinates"""
        x = int((point[0] - self.map_origin.position.x) / self.map_resolution)
        y = int((point[1] - self.map_origin.position.y) / self.map_resolution)
        return (x, y)

    def grid_to_world(self, point: Tuple[int, int]) -> Tuple[float, float]:
        """Convert grid coordinates to world coordinates"""
        x = point[0] * self.map_resolution + self.map_origin.position.x
        y = point[1] * self.map_resolution + self.map_origin.position.y
        return (x, y)

    def is_collision_free(self, point: Tuple[float, float]) -> bool:
        """Check if a point is collision-free using dilated grid"""
        grid_point = self.world_to_grid(point)
        
        # Check if point is within map bounds
        if (0 <= grid_point[0] < self.dilated_grid.shape[1] and 
            0 <= grid_point[1] < self.dilated_grid.shape[0]):
            return not self.dilated_grid[grid_point[1], grid_point[0]]
        return False

    def is_path_collision_free(self, start: Tuple[float, float], end: Tuple[float, float]) -> bool:
        """Check if path between two points is collision-free"""
        start = np.array(start)
        end = np.array(end)
        vec = end - start
        dist = np.linalg.norm(vec)
        
        if dist == 0:
            return True
        
        # Check points along the path
        n_checks = max(3, int(dist / self.map_resolution))
        for i in range(n_checks):
            point = tuple(start + (i/n_checks) * vec)
            if not self.is_collision_free(point):
                return False
        return True

    def sample_free_space(self) -> None:
        """Generate random collision-free samples focusing on free space"""
        self.samples = []
        
        # Get free space coordinates
        free_space = np.where(self.dilated_grid == 0)
        if len(free_space[0]) == 0:
            self.get_logger().error("No free space found in map!")
            return
        
        # Sample points from free space
        while len(self.samples) < self.n_samples:
            # Randomly select an index from free space
            idx = np.random.randint(0, len(free_space[0]))
            grid_y = free_space[0][idx]
            grid_x = free_space[1][idx]
            
            # Convert to world coordinates with small random offset
            world_x, world_y = self.grid_to_world((grid_x, grid_y))
            # Add small random offset within the grid cell
            offset = self.map_resolution * 0.5
            world_x += np.random.uniform(-offset, offset)
            world_y += np.random.uniform(-offset, offset)
            
            self.samples.append((world_x, world_y))
        
        self.samples = np.array(self.samples)
        self.kdtree = KDTree(self.samples)

    def build_graph(self) -> None:
        """Build the roadmap graph with adaptive radius"""
        n = len(self.samples)
        d = 2  # 2D space
        
        # Calculate free space volume
        free_space_cells = np.sum(self.dilated_grid == 0)
        volume = free_space_cells * (self.map_resolution ** 2)
        
        # Calculate connection radius
        gamma = 2 * (1 + 1/d)**(1/d) * (volume/np.pi)**(1/d)
        r = gamma * (np.log(n)/n)**(1/d)
        
        # Ensure minimum connection radius
        r = max(r, self.map_resolution * 5)
        
        self.graph = {i: set() for i in range(len(self.samples))}
        
        # Connect nodes within radius
        for i in range(len(self.samples)):
            neighbors = self.kdtree.query_ball_point(self.samples[i], r)
            for j in neighbors:
                if i != j and self.is_path_collision_free(
                    tuple(self.samples[i]), tuple(self.samples[j])
                ):
                    self.graph[i].add(j)
                    self.graph[j].add(i)

    def find_path(self, start: Tuple[float, float], goal: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Find path using A* search with improved start/goal connections"""
        if not (self.is_collision_free(start) and self.is_collision_free(goal)):
            self.get_logger().warn("Start or goal position is in collision!")
            return None
            
        # Add start and goal to graph
        self.samples = np.vstack([self.samples, [start, goal]])
        start_idx = len(self.samples) - 2
        goal_idx = len(self.samples) - 1
        
        # Connect start and goal to nearby nodes
        self.kdtree = KDTree(self.samples)
        k = min(20, len(self.samples))  # Increased number of connections
        
        for idx in [start_idx, goal_idx]:
            _, indices = self.kdtree.query(self.samples[idx], k=k)
            self.graph[idx] = set()
            for j in indices:
                if idx != j and self.is_path_collision_free(
                    tuple(self.samples[idx]), tuple(self.samples[j])
                ):
                    self.graph[idx].add(j)
                    self.graph[j].add(idx)
        
        # Check if start and goal are connected
        if not self.graph[start_idx] or not self.graph[goal_idx]:
            self.get_logger().warn("Could not connect start or goal to roadmap!")
            return None
        
        # A* search
        def heuristic(node: int) -> float:
            return np.linalg.norm(self.samples[node] - self.samples[goal_idx])
        
        open_set = [(0 + heuristic(start_idx), 0, start_idx, [start_idx])]
        closed_set = set()
        
        while open_set:
            _, cost, current, path = heapq.heappop(open_set)
            
            if current == goal_idx:
                return [tuple(self.samples[i]) for i in path]
            
            if current in closed_set:
                continue
                
            closed_set.add(current)
            
            for neighbor in self.graph[current]:
                if neighbor not in closed_set:
                    new_cost = cost + np.linalg.norm(
                        self.samples[current] - self.samples[neighbor]
                    )
                    heapq.heappush(
                        open_set,
                        (new_cost + heuristic(neighbor), new_cost, neighbor, path + [neighbor])
                    )
        
        return None

    def plan_path(self, start_point: Tuple[float, float], goal_point: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Complete planning process"""
        self.sample_free_space()
        self.build_graph()
        return self.find_path(start_point, goal_point)

    def publish_path(self, path: List[Tuple[float, float]]) -> None:
        """Convert path to PoseArray and publish"""
        self.trajectory.clear()
        for point in path:
            self.trajectory.addPoint(point)
        
        self.traj_pub.publish(self.trajectory.toPoseArray())
        self.trajectory.publish_viz()


def main(args=None):
    rclpy.init(args=args)
    planner = PRMStar()
    rclpy.spin(planner)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
