import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
import numpy as np

from .utils import LineTrajectory


class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 0  # FILL IN #
        self.speed = 0  # FILL IN #
        self.wheelbase_length = 0  # FILL IN #
        self.location = [0,0,0]

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        self.location_sub = self.create_subscription(Odometry, "/pf/pose/odom", self.location_callback, 1)
        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)

    def pose_callback(self, odometry_msg):
        #probably needs to be in a new function, but just put it here
        #FINDS closest line segment to car's location. Assumes the particle filter is publishing average pose
        points = np.array(self.trajectory.points)
        num_points = len(points)
        distances = np.zeros(num_points-1)#distance from location to line segment, defined by first point
        for i in range(num_points-1):
            a1 = self.location[:2] - points[i]
            a2 = points[i+1] - points[i]
            L = self.dist(a2,0)
            if L > 1e-5:
                coeff = max(0,min(1,self.dot_prod(a1,a2)/L))
                proj_point = points[i]+coeff*a2
            else:
                proj_point = points[i]
            distances[i] = self.dist(proj_point,self.location[:2])
        closest_index = np.argmin(distances)
        segment = points[closest_index:closest_index+2]
    
    def dot_prod(p1,p2):
        #numpy operations
        return sum(np.multiply(p1,p2))
    
    def dist(p1,p2):
        #numpy operations
        return sum(np.square(p1-p2))**0.5
    
    def location_callback(self, msg):
        x = msg.pose.pose.position.x
        y=msg.pose.pose.position.y
        theta=2*np.arctan2(msg.pose.pose.orientation.z,msg.pose.pose.orientation.w)
        self.location = np.array([x,y,theta])

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()
