#!/usr/bin/env python3
import socket
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message

from etsi_its_cpm_ts_msgs.msg import CollectivePerceptionMessage


class CPMUdpReceiver(Node):
    def __init__(self):
        super().__init__("cpm_udp_receiver")

        self.pub = self.create_publisher(
            CollectivePerceptionMessage,
            "/cpm",
            10
        )

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", 5000))
        self.sock.setblocking(False)

        self.timer = self.create_timer(0.01, self.receive)

        self.get_logger().info("Listening for CPM UDP packets on port 5000")

    def receive(self):
        try:
            data, addr = self.sock.recvfrom(4096)

            msg = deserialize_message(
                data,
                CollectivePerceptionMessage
            )

            self.pub.publish(msg)

        except BlockingIOError:
            pass
        except Exception as e:
            self.get_logger().error(f"CPM decode failed: {e}")


def main():
    rclpy.init()
    node = CPMUdpReceiver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
