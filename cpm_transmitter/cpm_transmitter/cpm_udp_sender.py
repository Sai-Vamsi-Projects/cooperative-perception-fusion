#!/usr/bin/env python3
import socket
import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message

from etsi_its_cpm_ts_msgs.msg import CollectivePerceptionMessage


class CPMUdpSender(Node):
    def __init__(self):
        super().__init__("cpm_udp_sender")

        # Subscribe to CPM
        self.create_subscription(
            CollectivePerceptionMessage,
            "/cpm_1",
            self.cpm_callback,
            10
        )

        # UDP broadcast config
        self.udp_ip = "255.255.255.255"   # broadcast
        self.udp_port = 5000

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        self.get_logger().info(
            f"Broadcasting CPM via UDP {self.udp_ip}:{self.udp_port}"
        )

    def cpm_callback(self, msg: CollectivePerceptionMessage):
        try:
            raw_bytes = serialize_message(msg)
            self.sock.sendto(raw_bytes, (self.udp_ip, self.udp_port))
        except Exception as e:
            self.get_logger().error(f"UDP send failed: {e}")


def main():
    rclpy.init()
    node = CPMUdpSender()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
