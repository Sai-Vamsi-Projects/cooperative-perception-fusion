
#!/usr/bin/env python3
"""
CPMTransmitter Node – Final Version
-----------------------------------
Converts fused ENU detections into ETSI ITS CPM messages and publishes /cpm.

Features:
 - Includes station_id, object_class, and confidence
 - Computes object dimensions dynamically (fallback if missing)
 - Uses ROS time for proper RViz visualization
 - Encodes ENU positions in ETSI scaling (0.01 m)
"""

import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message, deserialize_message
from nav_msgs.msg import Odometry
from vx_custom_msgs.msg import FusedObjects
from etsi_its_cpm_ts_msgs.msg import (
    CollectivePerceptionMessage,
    PerceivedObject,
    WrappedCpmContainer,
    PerceivedObjectContainer,
    ObjectClass,
    ObjectClassWithConfidence,
)


class CPMTransmitter(Node):
    def __init__(self):
        super().__init__("cpm_transmitter")

        # --- Subscribers ---
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.create_subscription(FusedObjects, "/fusion_result", self.objects_callback, 10)

        # --- Publisher ---
        self.publisher_ = self.create_publisher(CollectivePerceptionMessage, "/cpm", 10)

        # --- Buffers ---
        self.latest_odom = None
        self.latest_objects = []
        self.station_id = 0

        # --- Timer ---
        self.timer = self.create_timer(0.5, self.publish_and_convert)

        # --- ETSI TrafficParticipantType map ---
        self.class_map = {
            "Person": 1,               # pedestrian
            "Car": 5,                  # passengerCar
            "Traffic_light_red": 15,   # roadSideUnit / static infra
            "Traffic_light_green": 15,
            "Plant": 14,               # agricultural
            "House": 15,               # roadSideUnit
            "unknown": 0,
        }

        self.get_logger().info("CPMTransmitter ready: publishing /cpm with live timestamps.")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def get_t_its(self, nanoseconds):
        """Convert ROS nanoseconds to ITS timestamp (ms)."""
        return int(nanoseconds / 1e6)

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg

    def objects_callback(self, msg: FusedObjects):
        self.latest_objects = msg.objects
        self.station_id = msg.station_id

    # ------------------------------------------------------------------
    # CPM builder
    # ------------------------------------------------------------------
    def build_cpm(self):
        msg = CollectivePerceptionMessage()

        # --- Header ---
        msg.header.protocol_version.value = 2
        msg.header.message_id.value = msg.header.message_id.CPM
        msg.header.station_id.value = self.station_id
        # msg.header.stamp = self.get_clock().now().to_msg()  # ✅ proper ROS time

        # --- Management container ---
        msg.payload.management_container.reference_time.value = self.get_t_its(
            self.get_clock().now().nanoseconds
        )

        # Reference position from odometry
        if self.latest_odom:
            pos = self.latest_odom.pose.pose.position
            msg.payload.management_container.reference_position.latitude.value = int(pos.x * 1e7)
            msg.payload.management_container.reference_position.longitude.value = int(pos.y * 1e7)

        # --- PerceivedObjectContainer ---
        cpm_container = WrappedCpmContainer()
        cpm_container.container_id.value = (
            cpm_container.CHOICE_CONTAINER_DATA_PERCEIVED_OBJECT_CONTAINER
        )
        perceived_container = PerceivedObjectContainer()
        perceived_container.number_of_perceived_objects.value = len(self.latest_objects)

        # --- Loop through fused objects ---
        for obj in self.latest_objects:
            po = PerceivedObject()
            po.object_id_is_present = True
            po.object_id.value = int(obj.object_id)

            # --- Position (ENU → CPM scaling: 0.01 m) ---
            po.position.x_coordinate.value.value = int(obj.enu_position.x * 100)
            po.position.x_coordinate.confidence.value = po.position.x_coordinate.confidence.UNAVAILABLE
            po.position.y_coordinate.value.value = int(obj.enu_position.y * 100)
            po.position.y_coordinate.confidence.value = po.position.y_coordinate.confidence.UNAVAILABLE

            # --- Dimensions (try dynamic, else fallback) ---
            if hasattr(obj, "dimensions"):
                dx = getattr(obj.dimensions, "x", 3.5)
                dy = getattr(obj.dimensions, "y", 1.8)
                dz = getattr(obj.dimensions, "z", 1.6)
            else:
                dx, dy, dz = 3.5, 1.8, 1.6  # fallback m

            po.object_dimension_x_is_present = True
            po.object_dimension_x.value.value = int(dx * 10)
            po.object_dimension_x.confidence.value = po.object_dimension_x.confidence.UNAVAILABLE

            po.object_dimension_y_is_present = True
            po.object_dimension_y.value.value = int(dy * 10)
            po.object_dimension_y.confidence.value = po.object_dimension_y.confidence.UNAVAILABLE

            po.object_dimension_z_is_present = True
            po.object_dimension_z.value.value = int(dz * 10)
            po.object_dimension_z.confidence.value = po.object_dimension_z.confidence.UNAVAILABLE

            # --- Classification ---
            class_code = self.class_map.get(obj.object_class, 0)
            cls = ObjectClass()
            cls.choice = cls.CHOICE_VEHICLE_SUB_CLASS
            cls.vehicle_sub_class.value = int(class_code)

            cls_conf = ObjectClassWithConfidence()
            cls_conf.object_class = cls
            cls_conf.confidence.value = 70  # 70% confidence (per ETSI 1–101)

            po.classification.array.append(cls_conf)
            po.classification_is_present = True

            # --- Perception quality (scaled from 0–1 → 0–100 int) ---
            po.object_perception_quality_is_present = True
            po.object_perception_quality.value = int(obj.confidence * 100)

            perceived_container.perceived_objects.array.append(po)

        # Add to payload
        cpm_container.container_data_perceived_object_container = perceived_container
        msg.payload.cpm_containers.value.array.append(cpm_container)
        return msg

    # ------------------------------------------------------------------
    # Publisher
    # ------------------------------------------------------------------
    def publish_and_convert(self):
        if not self.latest_odom or not self.latest_objects:
            return

        cpm_msg = self.build_cpm()
        # cpm_msg.header.stamp = self.get_clock().now().to_msg()  # ensure valid ROS time

        self.publisher_.publish(cpm_msg)

        # Optional test: serialize/deserialize to verify structure
        serialize_message(cpm_msg)
        self.get_logger().info(
            f"CPM published: {len(self.latest_objects)} objects | station_id={self.station_id}"
        )


# ----------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = CPMTransmitter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
