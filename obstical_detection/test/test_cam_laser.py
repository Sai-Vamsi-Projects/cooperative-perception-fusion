import pytest
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, CameraInfo
from vision_msgs.msg import Detection2D, Detection2DArray
from vx_custom_msgs.msg import CamLaser
import math
import numpy as np

from obstical_detection.cam_and_lidar_pointCluster import LidarCameraFusion


@pytest.fixture(scope="module")
def rclpy_init():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def fusion_node(rclpy_init):
    node = LidarCameraFusion()
    yield node
    node.destroy_node()


def publish_camera_info(node):
    msg = CameraInfo()
    msg.header.frame_id = "camera_link"
    # fx, fy, cx, cy
    msg.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
    node.camera_info_callback(msg)


def publish_scan(node, distances, angles):
    scan = LaserScan()
    scan.header.frame_id = "lidar_link"
    scan.angle_min = angles[0]
    scan.angle_max = angles[-1]
    scan.angle_increment = angles[1] - angles[0] if len(angles) > 1 else 0.1
    scan.ranges = distances
    node.scan_callback(scan)


def publish_detections(node, bbox_center=(320.0, 240.0), size=(50.0, 50.0), class_id="\x05"):
    det = Detection2D()
    det.bbox.center.x = bbox_center[0]
    det.bbox.center.y = bbox_center[1]
    det.bbox.size_x = size[0]
    det.bbox.size_y = size[1]
    det.results.append(type('res', (), {})())  # dummy object
    det.results[0].hypothesis = type('hyp', (), {})()
    det.results[0].hypothesis.class_id = class_id
    det_arr = Detection2DArray()
    det_arr.detections.append(det)
    node.detection_callback(det_arr)


def test_fusion_output(fusion_node):
    # Publish camera info
    publish_camera_info(fusion_node)

    # Publish LiDAR scan (distances and angles)
    distances = [1.0, 0.8, 0.5, 0.6, 0.9]  # meters
    angles = [math.radians(-10), math.radians(-5), 0.0, math.radians(5), math.radians(10)]
    publish_scan(fusion_node, distances, angles)

    # Publish detection bounding box
    publish_detections(fusion_node)

    # Check that fused messages are published
    # Since CamLaser publisher is internal, we'll override it to capture messages
    received_msgs = []

    def fake_publish(msg):
        received_msgs.append(msg)

    fusion_node.pub.publish = fake_publish

    # Run fusion manually
    fusion_node.try_fusion()

    # Wait a moment for processing
    rclpy.spin_once(fusion_node, timeout_sec=0.1)

    # Assertions based on acceptance criteria
    assert len(received_msgs) > 0, "No fused messages published"

    for msg in received_msgs:
        # A.C.2.1.5: depth should be > 0 and class_id should be a valid label
        assert msg.distance > 0.0
        assert msg.class_id in fusion_node.label_map.values() or "class_" in msg.class_id

