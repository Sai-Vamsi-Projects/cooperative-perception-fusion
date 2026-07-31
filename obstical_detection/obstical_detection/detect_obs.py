import rclpy
from rclpy.node import Node
import math
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy


class ScanROIFilter(Node):
    def __init__(self):
        super().__init__('scan_roi_filter')

        # BEST_EFFORT QoS for LiDAR
        scan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            scan_qos
        )

        self.pub = self.create_publisher(
            LaserScan,
            '/scan_roi',
            10
        )

        # ROI limits (CHANGE AS YOU WANT)
        self.roi_start_deg = -90
        self.roi_end_deg   = 90
        self.roi_start = math.radians(self.roi_start_deg)
        self.roi_end   = math.radians(self.roi_end_deg)
        self.get_logger().info("ROI Scan Filter Started")

    def scan_callback(self, msg: LaserScan):

        # Compute index bounds for ROI
        start_index = int((self.roi_start - msg.angle_min) / msg.angle_increment)
        end_index = int((self.roi_end - msg.angle_min) / msg.angle_increment)

        start_index = max(0, start_index)
        end_index = min(len(msg.ranges), end_index)

        # Create new LaserScan message
        roi_msg = LaserScan()
        roi_msg.header = msg.header
        roi_msg.angle_min = msg.angle_min + start_index * msg.angle_increment
        roi_msg.angle_max = msg.angle_min + end_index * msg.angle_increment
        roi_msg.angle_increment = msg.angle_increment
        roi_msg.time_increment = msg.time_increment
        roi_msg.scan_time = msg.scan_time
        roi_msg.range_min = msg.range_min
        roi_msg.range_max = msg.range_max

        # Keep only ROI ranges
        roi_msg.ranges = msg.ranges[start_index:end_index]

        # Publish ROI
        self.pub.publish(roi_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ScanROIFilter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
