import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header

from etsi_its_cpm_ts_msgs.msg import (
    CollectivePerceptionMessage)


class CPMToMarkerNode(Node):

    def __init__(self):
        super().__init__("cpm_to_marker_node")

        self.sub = self.create_subscription(
            CollectivePerceptionMessage,
            "/cpm_1",
            self.cpm_callback,
            10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/markers",
            10
        )

        self.get_logger().info("CPM → MarkerArray node started")

    def cpm_callback(self, msg: CollectivePerceptionMessage):
        marker_array = MarkerArray()

        # ✅ CORRECT ACCESS
        containers = msg.payload.cpm_containers.value.array

        for container in containers:

            # Perceived Object Container ID = 5
            if container.container_id.value != 5:
                continue

            poc = container.container_data_perceived_object_container

            objects = poc.perceived_objects.array

            for obj in objects:

                obj_id = obj.object_id.value

                # ETSI CPM: centimeters → meters
                x = obj.position.x_coordinate.value.value * 0.01
                y = obj.position.y_coordinate.value.value * 0.01
                z = 0.0

                length = obj.object_dimension_x.value.value * 0.1
                width  = obj.object_dimension_y.value.value * 0.1
                height = obj.object_dimension_z.value.value * 0.1

                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = self.get_clock().now().to_msg()

                marker.ns = "cpm_objects"
                marker.id = obj_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD

                marker.pose.position.x = x
                marker.pose.position.y = y
                marker.pose.position.z = height / 2.0
                marker.pose.orientation.w = 1.0

                marker.scale.x = float(length)
                marker.scale.y = float(width)

                marker.color.r = 0.0
                marker.color.g = 0.6
                marker.color.b = 1.0
                marker.color.a = 0.8

                marker.lifetime.sec = 1

                marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)


def main():
    rclpy.init()
    node = CPMToMarkerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
