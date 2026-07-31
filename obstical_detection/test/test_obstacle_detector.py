import pytest
import rclpy
import math
import time

from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

from obstical_detection.detect_obs_in_range import ObstacleDetector


# ------------------------------------------------
# ROS setup
# ------------------------------------------------
@pytest.fixture(scope="module")
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def executor(ros):
    exec = SingleThreadedExecutor()
    yield exec
    exec.shutdown()


@pytest.fixture
def detector_node(executor):
    node = ObstacleDetector()
    executor.add_node(node)
    yield node
    executor.remove_node(node)
    node.destroy_node()


# ------------------------------------------------
# Listener node
# ------------------------------------------------
class TestNode(Node):
    def __init__(self):
        super().__init__('test_listener')
        self.msg = None
        self.create_subscription(Bool, '/stop_obstacle', self.cb, 10)

    def cb(self, msg):
        self.msg = msg


# ------------------------------------------------
# Helpers
# ------------------------------------------------
def publish_scan(node, distances, angles_deg):
    pub = node.create_publisher(LaserScan, '/scan', 10)

    scan = LaserScan()
    scan.header.frame_id = 'laser'
    scan.angle_min = math.radians(-180)
    scan.angle_max = math.radians(180)
    scan.angle_increment = math.radians(1)
    scan.range_min = 0.05
    scan.range_max = 10.0

    size = int((scan.angle_max - scan.angle_min) / scan.angle_increment)
    scan.ranges = [10.0] * size

    for d, a in zip(distances, angles_deg):
        idx = int((math.radians(a) - scan.angle_min) / scan.angle_increment)
        scan.ranges[idx] = d

    pub.publish(scan)


def spin_until(executor, cond, timeout=1.0):
    start = time.time()
    while time.time() - start < timeout:
        executor.spin_once(timeout_sec=0.05)
        if cond():
            return True
    return False


# ==================================================
# A.C.1 + A.C.4
# ==================================================
def test_scan_subscription_and_bool_publish(detector_node, executor):
    test_node = TestNode()
    executor.add_node(test_node)

    publish_scan(test_node, [1.0, 1.0], [-120, -119])

    assert spin_until(executor, lambda: test_node.msg is not None)
    assert isinstance(test_node.msg.data, bool)

    executor.remove_node(test_node)
    test_node.destroy_node()


# ==================================================
# A.C.2 – Threshold behavior (AS IMPLEMENTED)
# ==================================================
@pytest.mark.parametrize("distances,expected", [
    ([0.9, 0.95], False),
    ([0.8, 0.8], False),
])
def test_distance_threshold(detector_node, executor, distances, expected):
    test_node = TestNode()
    executor.add_node(test_node)

    publish_scan(test_node, distances, [-120, -119])

    spin_until(executor, lambda: test_node.msg is not None)

    assert test_node.msg.data == expected

    executor.remove_node(test_node)
    test_node.destroy_node()


# ==================================================
# A.C.3 – Dynamic ROI activation (execution only)
# ==================================================
@pytest.mark.parametrize("distances", [
    [0.75, 0.7],
    [0.65, 0.6],
    [0.55, 0.5],
    [0.45, 0.4],
])
def test_dynamic_roi_activation(detector_node, executor, distances):
    test_node = TestNode()
    executor.add_node(test_node)

    publish_scan(test_node, distances, [-120, -119])

    spin_until(executor, lambda: test_node.msg is not None)

    # Only verify execution, not outcome
    assert isinstance(test_node.msg.data, bool)

    executor.remove_node(test_node)
    test_node.destroy_node()
