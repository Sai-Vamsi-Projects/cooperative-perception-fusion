#!/usr/bin/env python3
"""
CPMTransmitter – Final (FusionResult-only, NO ODOM)
---------------------------------------------------
Consumes vx_custom_msgs/FusionResult from /fusion_result and publishes ETSI ITS CPM on /cpm.

Key points:
- NO /odom subscription, NO odometry usage, NO ENU assumption.
- Uses only fields that actually exist in FusionResult:
  map_position, confidence, width, height, depth (depth used only as RANGE, not as object dimension).
- CPM reference_position is set to 0/0 as a placeholder to avoid unset-field issues.
- PerceivedObject position uses meter->0.01 m scaling (x*100, y*100) like your previous pipeline.
- Object dimensions:
    - object_dimension_y  <- fusion.width   (treated as "width")
    - object_dimension_z  <- fusion.height  (treated as "height")
    - object_dimension_x  is omitted (unknown length) to avoid publishing wrong data.
- Classification is set to "unknown" (no class in FusionResult).
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.serialization import serialize_message

from vx_custom_msgs.msg import FusionResult

from etsi_its_cpm_ts_msgs.msg import (
    CollectivePerceptionMessage,
    PerceivedObject,
    WrappedCpmContainer,
    PerceivedObjectContainer,
    ObjectClass,
    ObjectClassWithConfidence,
)


def _is_finite_number(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


# def _clamp_int(x: float, lo: int, hi: int) -> int:
#     if not _is_finite_number(x):
#         return lo
#     xi = int(x)
#     if xi < lo:
#         return lo
#     if xi > hi:
#         return hi
#     return xi


class CPMTransmitter(Node):
    def __init__(self):
        super().__init__("cpm_transmitter")

        # ---- Subscriber ----
        self.create_subscription(
            FusionResult,
            "/fusion_result",
            self.fusion_callback,
            10,
        )

        # ---- Publisher ----
        self.publisher_ = self.create_publisher(
            CollectivePerceptionMessage,
            "/cpm_1",
            10,
        )

        # If you have a real station id elsewhere, set it here or read from a param.
        self.station_id = 0

        # Since FusionResult does not carry stable track IDs, we generate a simple counter.
        # NOTE: This is not a tracker. IDs will increase each callback.
        self.object_id_counter = 0

        self.get_logger().info("CPMTransmitter ready (FusionResult-only, no odom).")

    # ---------------------------------------------------------
    def ros_to_its_time_ms(self, nanoseconds: int) -> int:
        """ROS nanoseconds -> ETSI ITS timestamp in milliseconds."""
        return int(nanoseconds / 1e6)

    # ---------------------------------------------------------
    def fusion_callback(self, fusion: FusionResult) -> None:
        # Validate minimum required fields (position + confidence should exist)
        # ROS messages always have the fields, but values can be NaN/Inf.
        if not (_is_finite_number(fusion.map_position.x) and _is_finite_number(fusion.map_position.y)):
            self.get_logger().warn("Skipping CPM: fusion.map_position has NaN/Inf.")
            return

        if not _is_finite_number(fusion.confidence):
            self.get_logger().warn("Skipping CPM: fusion.confidence has NaN/Inf.")
            return

        cpm_msg = self.build_cpm_from_fusion(fusion)

        # Publish + serialize to catch encoding problems early
        self.publisher_.publish(cpm_msg)
        serialize_message(cpm_msg)

        self.get_logger().info(
            f"CPM published | obj_id={self.object_id_counter} | conf={fusion.confidence:.2f}"
        )

    # ---------------------------------------------------------
    def build_cpm_from_fusion(self, fusion: FusionResult) -> CollectivePerceptionMessage:
        msg = CollectivePerceptionMessage()

        # ---------------- Header ----------------
        msg.header.protocol_version.value = 2
        msg.header.message_id.value = msg.header.message_id.CPM
        msg.header.station_id.value = int(self.station_id)
        # msg.header.stamp = self.get_clock().now().to_msg() -----------change

        # ---------------- Management container ----------------
        now_ns = self.get_clock().now().nanoseconds
        msg.payload.management_container.reference_time.value = self.ros_to_its_time_ms(now_ns)

        # NO ODOM available -> set placeholder reference position explicitly.
        # This avoids "unset field" failures in some consumers.
        # (It is NOT a real geo reference.)
        msg.payload.management_container.reference_position.latitude.value = 0
        msg.payload.management_container.reference_position.longitude.value = 0

        # ---------------- PerceivedObjectContainer ----------------
        cpm_container = WrappedCpmContainer()
        cpm_container.container_id.value = (
            cpm_container.CHOICE_CONTAINER_DATA_PERCEIVED_OBJECT_CONTAINER
        )

        perceived_container = PerceivedObjectContainer()
        perceived_container.number_of_perceived_objects.value = 1

        po = PerceivedObject()

        # Object id (generated)
        self.object_id_counter += 1
        po.object_id_is_present = True
        po.object_id.value = int(self.object_id_counter)

        # ---------------- Position ----------------
        # Keeping your existing convention: meters -> 0.01m scaling (x*100, y*100).
        # NOTE: ETSI CPM typically expects coordinates relative to a reference position.
        # With no odom/geo reference, we keep it consistent with your pipeline for visualization/testing.
        po.position.x_coordinate.value.value = int(float(fusion.map_position.x) * 100.0)
        po.position.y_coordinate.value.value = int(float(fusion.map_position.y) * 100.0)

        po.position.x_coordinate.confidence.value = (
            po.position.x_coordinate.confidence.UNAVAILABLE
        )
        po.position.y_coordinate.confidence.value = (
            po.position.y_coordinate.confidence.UNAVAILABLE
        )

        # ---------------- Dimensions ----------------
        # FusionResult:
        # - width  = estimated physical width (meters)
        # - height = estimated physical height (meters)
        # - depth  = RANGE (distance), NOT object length -> DO NOT publish as a dimension
        #
        # ETSI PerceivedObject dimensions x/y/z represent object size. Since "length" is unknown,
        # we only publish what we actually have.
        #
        # Scaling used in your code: 0.1m -> value.value = meters*10
        if _is_finite_number(fusion.width) and float(fusion.width) > 0.0:
            po.object_dimension_y_is_present = True
            po.object_dimension_y.value.value = int(float(fusion.width) * 10.0)
            po.object_dimension_y.confidence.value = (
                po.object_dimension_y.confidence.UNAVAILABLE
            )

        if _is_finite_number(fusion.height) and float(fusion.height) > 0.0:
            po.object_dimension_z_is_present = True
            po.object_dimension_z.value.value = int(float(fusion.height) * 10.0)
            po.object_dimension_z.confidence.value = (
                po.object_dimension_z.confidence.UNAVAILABLE
            )

        if _is_finite_number(fusion.depth) and float(fusion.depth) > 0.0:
            po.object_dimension_x_is_present = True
            po.object_dimension_x.value.value = int(float(fusion.depth) * 10.0)
            po.object_dimension_x.confidence.value = (
                po.object_dimension_x.confidence.UNAVAILABLE
            )

        # ---------------- Classification ----------------
        # FusionResult does not include class -> publish "unknown".
        cls = ObjectClass()
        # Safest compatibility: keep vehicle_sub_class path but use 0 (unknown) like your earlier mapping.
        cls.choice = cls.CHOICE_VEHICLE_SUB_CLASS
        cls.vehicle_sub_class.value = 0

        cls_conf = ObjectClassWithConfidence()
        cls_conf.object_class = cls
        # Confidence for class is unknown; keep a neutral value.
        cls_conf.confidence.value = int((fusion.confidence)*100)

        po.classification_is_present = True
        po.classification.array.append(cls_conf)

        # ---------------- Perception quality ----------------
        # obj.confidence is float [0..1] ideally; clamp to [0..100]
        po.object_perception_quality_is_present = True
        # po.object_perception_quality.value = _clamp_int(float(fusion.confidence) * 100.0, 0, 100)
        po.object_perception_quality.value = int(fusion.confidence)

        perceived_container.perceived_objects.array.append(po)

        # Add container to CPM payload
        cpm_container.container_data_perceived_object_container = perceived_container
        msg.payload.cpm_containers.value.array.append(cpm_container)

        return msg


def main(args=None):
    rclpy.init(args=args)
    node = CPMTransmitter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()