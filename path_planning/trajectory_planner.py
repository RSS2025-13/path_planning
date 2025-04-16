import rclpy
from rclpy.node import Node

assert rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped, PoseArray, Pose
from nav_msgs.msg import OccupancyGrid, Odometry
from .utils import LineTrajectory
import numpy as np
from .rrt_star_functions import rrt_star, rrt
from .a_star_functions import a_star
import tf_transformations as tf

class PathPlan(Node):
    """ Listens for goal pose published by RViz and uses it to plan a path from
    current car pose.
    """

    def __init__(self):
        super().__init__("trajectory_planner")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('map_topic', "default")
        self.declare_parameter('initial_pose_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.map_topic = self.get_parameter('map_topic').get_parameter_value().string_value
        self.initial_pose_topic = self.get_parameter('initial_pose_topic').get_parameter_value().string_value

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

        self.obstacles_pub = self.create_publisher(
            PoseArray,
            "/obstacles",
            10
        )

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            self.initial_pose_topic,
            self.pose_cb,
            10
        )

        self.trajectory = LineTrajectory(node=self, viz_namespace="/planned_trajectory")

        self.x_bounds = (-61, 25) # in meters
        self.y_bounds = (-16, 48) # in meters
        self.obstacles = [] #x,y,radius (in meters)
        self.start = None

    def map_cb(self, msg):
        self.get_logger().info("Processing Map")
        T = self.pose_to_T(msg.info.origin)
        table = np.array(msg.data)
        table = table.reshape((msg.info.height, msg.info.width))
        
        # Convert to binary map (0 for free, 1 for occupied, -1 for unknown)
        binary_map = np.zeros_like(table)
        binary_map[table == -1] = -1  # Unknown cells
        binary_map[table > 65] = 1    # Occupied cells (lower threshold to be more conservative)
        
        # Create larger kernel for more aggressive dilation
        kernel_size = 5  # Increased from 3 to 5
        
        # First pass: Fill in unknown areas near obstacles
        filled_map = binary_map.copy()
        for i in range(2, binary_map.shape[0]-2):
            for j in range(2, binary_map.shape[1]-2):
                if binary_map[i,j] == -1:  # If cell is unknown
                    window = binary_map[i-2:i+3, j-2:j+3]
                    if np.any(window == 1):  # If any nearby cell is occupied
                        filled_map[i,j] = 1
        
        # Dilate to expand obstacles
        dilated = np.zeros_like(filled_map)
        for i in range(2, filled_map.shape[0]-2):
            for j in range(2, filled_map.shape[1]-2):
                window = filled_map[i-2:i+3, j-2:j+3]
                # More aggressive dilation - mark as obstacle if any cell in 5x5 window is occupied
                dilated[i,j] = 1 if np.any(window == 1) else 0
        
        # Clear previous obstacles
        self.obstacles = []
        
        # Create obstacles with smaller spacing but larger radius
        obstacle_count = 0
        obstacle_radius = 0.25  # Increased radius for better coverage
        
        for i, row in enumerate(dilated):
            for j, val in enumerate(row):
                if val == 1:
                    px = np.eye(3)
                    px[1,2] = i*msg.info.resolution
                    px[0,2] = j*msg.info.resolution
                    p = T@px
                    # Add obstacle with increased radius
                    self.obstacles.append([p[0,2], p[1,2], obstacle_radius])
                    obstacle_count += 1
        
        # Publish obstacles for visualization
        obstacle_msg = self.arr_to_pose_arr(self.obstacles)
        self.obstacles_pub.publish(obstacle_msg)
        
        self.get_logger().info(f"Map processed: {obstacle_count} obstacles created with radius {obstacle_radius}m")

    def pose_cb(self, pose):
        self.start = [pose.pose.pose.position.x, pose.pose.pose.position.y]

    def goal_cb(self, msg):
        self.method = 'a_star'
        goal = [msg.pose.position.x, msg.pose.position.y]

        if self.method == 'a_star':
            self.get_logger().info("Start: (%s,%s)" % (self.start[0],self.start[1]))
            self.get_logger().info("Goal: (%s,%s)" % (goal[0],goal[1]))
            astar = a_star(self.obstacles, self.start, goal)        
            astar.node = self  # Pass the ROS node to A*
            # self.get_logger().info(",".join(str(loc)+str(astar.grid.nodes[loc].obstacle) for loc in astar.grid.nodes))
            self.get_logger().info("Finding path with A*")
            traj = astar.plan(self.start, goal)
            self.trajectory.points = traj
            # self.get_logger().info(str(traj))
            self.get_logger().info(f"Path found")
            self.traj_pub.publish(self.trajectory.toPoseArray())
            self.trajectory.publish_viz()
        
        elif self.method == "rrt_star":
            rrt_inst = rrt_star(self.start, goal, self.obstacles, self.x_bounds, self.y_bounds)
            rrt_inst.node = self  # Pass the ROS node to RRT*
            # rrt = rrt_star(self.start, goal, self.obstacles, self.x_bounds, self.y_bounds)
            
            self.get_logger().info("Finding path with RRT*")
            traj_result = rrt_inst.plan()
            
            # RRT* returns a tuple of (path, cost)
            if traj_result[0] is not None:
                traj = traj_result[0]  # Extract just the path
                self.trajectory.points = traj
                self.get_logger().info(f"Path found with cost {traj_result[1]}")
                self.traj_pub.publish(self.trajectory.toPoseArray())
                self.trajectory.publish_viz()
            else:
                self.get_logger().error("No path found with RRT*")

    def pose_to_T(self, pose_msg):
        th = tf.euler_from_quaternion([
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w,
        ])[2]
        x, y = pose_msg.position.x, pose_msg.position.y
        return np.array([
            [np.cos(th), -np.sin(th), x],
            [np.sin(th),  np.cos(th), y],
            [         0,           0, 1],
        ])
    
    def arr_to_pose_arr(self, arr):
        msg = PoseArray()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        a = []
        for i in arr:
            pose = Pose()
            pose.position.x = i[0]
            pose.position.y = i[1]
            pose.position.z = 0.0
            pose.orientation.w = 1.0
            a.append(pose)
        msg.poses = a
        return msg

    # def plan_path(self, start_point, end_point, map):
    #     self.traj_pub.publish(self.trajectory.toPoseArray())
    #     self.trajectory.publish_viz()


def main(args=None):
    rclpy.init(args=args)
    planner = PathPlan()
    rclpy.spin(planner)
    rclpy.shutdown()
