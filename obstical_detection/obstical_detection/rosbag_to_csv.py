#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import csv
from vx_custom_msgs.msg import FusionResult

CSV_FILE = "fusion_2.csv"

class FusionResultLogger(Node):
    def __init__(self):
        super().__init__('fusion_result_logger')
        self.subscription = self.create_subscription(
            FusionResult,
            '/fusion_result',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning

        # Open CSV file and write headers
        self.csv_file = open(CSV_FILE, 'w', newline='')
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=[
            'confidence', 
            'map_x', 'map_y', 'map_z',
            'width_m', 'height_m', 'depth_m',
            'e_x', 'e_y', 'e_z'
        ])
        self.csv_writer.writeheader()
        self.get_logger().info(f"Logging /fusion_result to {CSV_FILE}")

    def listener_callback(self, msg):
        row = {
            'confidence': msg.confidence,
            'map_x': msg.map_position.x,
            'map_y': msg.map_position.y,
            'map_z': msg.map_position.z,
            'width_m': msg.width,
            'height_m': msg.height,
            'depth_m': msg.depth,
            'e_x': msg.error.x,
            'e_y': msg.error.y,
            'e_z': msg.error.z,
        }
        self.csv_writer.writerow(row)
        self.csv_file.flush()  # ensure it's written immediately
        self.get_logger().info(f"Logged object with confidence: {msg.confidence:.2f}")

    def destroy_node(self):
        # Close the CSV file when shutting down
        self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = FusionResultLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
