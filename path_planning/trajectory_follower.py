import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry
from rclpy.node import Node
import numpy as np

from .utils import LineTrajectory

#wasn't part of imports originally
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import numpy as np

class PurePursuit(Node):
    """ Implements Pure Pursuit trajectory tracking with a fixed lookahead and speed.
    """

    def __init__(self):
        super().__init__("trajectory_follower")
        self.declare_parameter('odom_topic', "default")
        self.declare_parameter('drive_topic', "default")

        self.odom_topic = self.get_parameter('odom_topic').get_parameter_value().string_value
        self.drive_topic = self.get_parameter('drive_topic').get_parameter_value().string_value

        self.lookahead = 2.0  # (meters) EXPERIMENT WITH VALUES (implement a changing lookahead distance based on curvature?) #
        self.speed = 2.0  # (meters/second) ADJUST AS NEEDED #
        self.wheelbase_length = 0.34  # (meters) FIND THE CORRECT VALUE #
        self.x_car, self.y_car, self.yaw = 0,0,0

        self.timer = self.create_timer(1.0/20.0,self.timer_callback)

        self.trajectory = LineTrajectory("/followed_trajectory")

        self.drive_pub = self.create_publisher(AckermannDriveStamped,
                                               self.drive_topic,
                                               1)
        self.error_pub = self.create_publisher(Float32,
                                               "/traj_error",
                                               10)
        
        #ADDED CODE IN INIT DIFFERING FROM TEMPLATE 
        self.odom_sub = self.create_subscription(Odometry, 
                                                 self.odom_topic, 
                                                 self.pose_callback, 
                                                 1)
        self.traj_sub = self.create_subscription(PoseArray,
                                                 "/trajectory/current",
                                                 self.trajectory_callback,
                                                 1)
        
        self.initialized_traj = False
        self.initialized_pose = False

    def timer_callback(self):
        #ENSURES WE HAVE A PATH
        if not self.initialized_traj or not self.initialized_pose:
            return

        #RETRIEVE POINTS ALONG PATH INTO NP
        path_pts = np.array(self.trajectory.points)

        #CALCULATE THE CLOSEST POINT TO THE CAR, WORKS IN MOST CASES
        dists = np.linalg.norm(path_pts - np.array([self.x_car, self.y_car]), axis=1)
        closest_idx = np.argmin(dists)

        traj_error = np.min(dists)
        #PUBLISH TRAJECTORY ERROR
        error_msg = Float32()
        error_msg.data = np.min(self.find_segment_distances(path_pts,[self.x_car,self.y_car]))
        self.error_pub.publish(error_msg)

        #USING THE NEAREST POINT AS A STARTING POINT, CALCULATE WHERE ALONG THE PATH THE LOOKAHEAD RADIUS LIES (lookahead point)
        for i in range(closest_idx, len(path_pts) - 1):
            p1 = path_pts[i]
            p2 = path_pts[i + 1]

            Q = np.array([self.x_car, self.y_car])
            r = self.lookahead
            V = p2 - p1           
            F = p1 - Q

            a = np.dot(V, V)
            b = 2 * np.dot(V, F)
            c = np.dot(F, F) - r**2

            discriminant = b**2 - 4 * a * c
            if discriminant < 0: 
                continue #line does not intersect circle

            sqrt_disc = np.sqrt(discriminant)
            t1 = (-b + sqrt_disc) / (2 * a)
            t2 = (-b - sqrt_disc) / (2 * a)

            if 0 <= t1 <= 1:
                lookahead_pt = p1 + t1 * V
                self.get_logger().info("Reaches t1 checkpoint")
                break
            elif 0 <= t2 <= 1:
                lookahead_pt = p1 + t2 * V
                self.get_logger().info("Reaches t2 checkpoint")
                break
            else:
                continue #line segment does not intersect circle (it could if it was extended)
        else:
            return #no lookahead point

        #CALUCLATE LOOKAHEAD POINT IN CAR'S FRAME
        dx = lookahead_pt[0] - self.x_car
        dy = lookahead_pt[1] - self.y_car
        local_x = np.cos(-self.yaw) * dx - np.sin(-self.yaw) * dy
        local_y = np.sin(-self.yaw) * dx + np.cos(-self.yaw) * dy

        # AVOIDS DIVIDING BY 0
        if local_x == 0:
            return 
        
        #CALCULATE STEERING ANGLE BASED ON PURE PURSUIT
        curvature = (2 * local_y) / (self.lookahead**2)
        steering_angle = np.arctan(self.wheelbase_length * curvature)

        #PUBLISH DRIVE
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = self.speed
        drive_msg.drive.steering_angle = steering_angle
        self.drive_pub.publish(drive_msg)
    
    def pose_callback(self, odometry_msg):
        #RETRIEVING CAR POSE FROM PARTICLE FILTER
        car_pose = odometry_msg.pose.pose.position
        self.x_car, self.y_car = car_pose.x, car_pose.y

        #CALCULATE ROTATION AROUND Z AXIS (yaw) FROM ODOM IN WORLD FRAME
        orientation = odometry_msg.pose.pose.orientation
        sin = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cos = 1 - 2 * (orientation.y**2 + orientation.z**2)
        self.yaw = np.arctan2(sin, cos)
        self.initialized_pose = True

    def trajectory_callback(self, msg):
        self.get_logger().info(f"Receiving new trajectory {len(msg.poses)} points")

        self.trajectory.clear()
        self.trajectory.fromPoseArray(msg)
        self.trajectory.publish_viz(duration=0.0)

        self.initialized_traj = True
    
    def find_segment_distances(self, traj_points, loc):
        to_locs = loc - traj_points[:-1]
        segments = traj_points[1:] - traj_points[:-1]
        L2s = np.sum(np.square(segments),axis=1)
        proj_percs = np.zeros_like(L2s) #defaults to 0 projection for case when endpoints of segment are same
        non_zero = L2s > 1e-4
        proj_percs[non_zero] = np.clip(np.sum(to_locs[non_zero]*segments[non_zero],axis=1)/L2s[non_zero],0,1) #0, 1, or percentage
        point_to_lines=traj_points[:-1] + proj_percs[:,None]*segments - loc
        dists = np.linalg.norm(point_to_lines,axis=1)
        return dists


def main(args=None):
    rclpy.init(args=args)
    follower = PurePursuit()
    rclpy.spin(follower)
    rclpy.shutdown()