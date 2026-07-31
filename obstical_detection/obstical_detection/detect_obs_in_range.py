import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import math
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy

class ObstacleDetector(Node):
    def __init__(self):
        super().__init__('obstacle_detector')
        scan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1)

        self.scan_sub = self.create_subscription(LaserScan,'/scan',self.scan_callback,scan_qos)
        self.roi_start_deg = -147.5
        self.roi_end_deg   = -82.5

        # Convert to radians
        self.roi_start = math.radians(self.roi_start_deg)
        self.roi_end   = math.radians(self.roi_end_deg)
        self.get_logger().info(f"{self.roi_start, self.roi_end}")
        self.stop_pub = self.create_publisher(Bool, '/stop_obstacle', 10)
        self.pub = self.create_publisher(LaserScan,'/scan_roi1',10)
        self.get_logger().info("Obstacle Detector Node Started")

    def scan_callback(self, msg: LaserScan):
        start_index = int((self.roi_start - msg.angle_min) / msg.angle_increment)
        end_index = int((self.roi_end - msg.angle_min) / msg.angle_increment)
        start_index = max(0, start_index)
        end_index = min(len(msg.ranges), end_index)
        roi_ranges= msg.ranges[start_index:end_index]
        sortranges=sorted(set(roi_ranges))
        m=sortranges[1]
        obsticle_etection = False

        if m <= 0.8 and m >= 0.7:
            roi_start_deg = -130
            roi_end_deg   = -105
        
            roi_ranges = self.process(roi_start_deg,roi_end_deg,msg)
            sortranges1=sorted(set(roi_ranges))
            n=sortranges1[1]
            if n <= 0.8:
                self.get_logger().info("true 0.8")
                self.get_logger().info(f"{n}")
                obsticle_etection = True
        elif m <= 0.7 and m >= 0.6:
            roi_start_deg = -135
            roi_end_deg   = -100
        
            roi_ranges = self.process(roi_start_deg,roi_end_deg,msg)
            sortranges1=sorted(set(roi_ranges))
            n=sortranges1[1]
            if n <= 0.7:
                self.get_logger().info("true 0.7")
                self.get_logger().info(f"{n}")
                obsticle_etection = True
        elif m <=0.6 and m>=0.5:
            roi_start_deg = -140
            roi_end_deg   = -95
        
            roi_ranges = self.process(roi_start_deg,roi_end_deg,msg)
            sortranges1=sorted(set(roi_ranges))
            n=sortranges1[1]
            if n <= 0.6:
                self.get_logger().info("true 0.6")
                self.get_logger().info(f"{n}")
                obsticle_etection = True
        elif m <= 0.5 and m >=0.4:
            roi_start_deg = -145
            roi_end_deg   = -90
        
            roi_ranges = self.process(roi_start_deg,roi_end_deg,msg)
            sortranges1=sorted(set(roi_ranges))
            n=sortranges1[1]
            if n <= 0.5:
                self.get_logger().info("true 0.5")
                self.get_logger().info(f"{n}")
                obsticle_etection = True
        elif m <= 0.4:
            self.get_logger().info("true 0.4")
            self.get_logger().info(f"{m}")
            obsticle_etection = True
        else:
            obsticle_etection = False
        self.publish_stop(obsticle_etection)

    def process(self, start_deg, end_deg, msg: LaserScan):
        roi_start = math.radians(start_deg)
        roi_end   = math.radians(end_deg)
        self.get_logger().info(f"{roi_start, roi_end}")
        start_index = int((roi_start - msg.angle_min) / msg.angle_increment)
        end_index = int((roi_end - msg.angle_min) / msg.angle_increment)
        start_index = max(0, start_index)
        end_index = min(len(msg.ranges), end_index)
        roi_range= msg.ranges[start_index:end_index]
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
        self.pub.publish(roi_msg)
        return roi_range

    def publish_stop(self, value: bool):
        msg = Bool()
        msg.data = value
        self.stop_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()