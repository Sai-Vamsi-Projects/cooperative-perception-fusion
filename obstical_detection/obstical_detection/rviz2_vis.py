#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray
from vx_custom_msgs.msg import CPM   # <-- replace with your package name

class CPMVisualizer(Node):

    def __init__(self):
        super().__init__('cpm_visualizer')

        self.sub = self.create_subscription(
            CPM,
            '/cpm',
            self.cpm_callback,
            10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/cpm_markers',
            10
        )

        self.get_logger().info("CPM visualizer started")

    def cpm_callback(self, msg: CPM):
        marker_array = MarkerArray()

        for obj in msg.objects:
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()

            # Use unique marker ID per sender + object
            marker.id = msg.sender_id * 1000 + obj.object_id

            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = obj.x
            marker.pose.position.y = obj.y
            marker.pose.position.z = obj.z

            marker.pose.orientation.w = 1.0

            # Box size (approx for 2D LiDAR objects)
            marker.scale.x = 0.8
            marker.scale.y = 0.8
            marker.scale.z = 1.5

            # Color based on sender
            marker.color.a = 0.8
            marker.color.r = (msg.sender_id % 3 == 0)
            marker.color.g = (msg.sender_id % 3 == 1)
            marker.color.b = (msg.sender_id % 3 == 2)

            marker.lifetime.sec = 1

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)

def main():
    rclpy.init()
    node = CPMVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
