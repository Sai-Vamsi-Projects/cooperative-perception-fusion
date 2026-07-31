#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import numpy as np
from sklearn.cluster import DBSCAN

from sensor_msgs.msg import LaserScan, CameraInfo
from vision_msgs.msg import Detection2DArray
from geometry_msgs.msg import PointStamped

import tf2_ros
import tf2_geometry_msgs

from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
)


class LidarCameraFusion(Node):
    def __init__(self):
        super().__init__("lidar_camera_fusion_node")

        scan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # ---------------- SUBSCRIPTIONS ----------------
        self.create_subscription(LaserScan, "/scan", self.scan_cb, scan_qos)
        self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.caminfo_cb,
            10,
        )
        self.create_subscription(
            Detection2DArray,
            "/detectnet/detections",
            self.det_cb,
            10,
        )

        # ---------------- TF ----------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------------- DATA ----------------
        self.scan = None
        self.caminfo = None
        self.detections = None

        self.get_logger().info("LiDAR–Camera Fusion with Dimension Estimation started")

    # ================= CALLBACKS =================

    def scan_cb(self, msg):
        self.scan = msg
        self.process()

    def caminfo_cb(self, msg):
        self.caminfo = msg
        self.process()

    def det_cb(self, msg):
        self.detections = msg
        self.process()

    # ================= MAIN PIPELINE =================

    def process(self):
        if self.scan is None or self.caminfo is None or self.detections is None:
            return

        # 1) LaserScan → points
        points_lidar = self.laserscan_to_points(self.scan)
        if len(points_lidar) < 5:
            return

        # 2) Cluster LiDAR points
        clusters = self.dbscan_clusters(points_lidar)
        if not clusters:
            return

        # 3) Cluster centroids
        centroids_lidar = [np.mean(c, axis=0) for c in clusters]

        # 4) Transform centroids to camera frame
        centroids_cam = self.transform_points(
            centroids_lidar,
            source=self.scan.header.frame_id,
            target=self.caminfo.header.frame_id,
        )

        if len(centroids_cam) == 0:
            return

        # 5) Project to image
        uv_points = self.project_to_image(centroids_cam)

        # 6) Match + estimate dimensions
        self.match_with_detections(
            uv_points, centroids_lidar, centroids_cam
        )

    # ================= CORE FUNCTIONS =================

    def laserscan_to_points(self, scan):
        angles = np.arange(
            scan.angle_min, scan.angle_max, scan.angle_increment
        )
        ranges = np.array(scan.ranges)

        mask = np.isfinite(ranges)
        angles = angles[mask]
        ranges = ranges[mask]

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        z = np.zeros_like(x)

        return np.vstack((x, y, z)).T

    def dbscan_clusters(self, points):
        db = DBSCAN(eps=0.1, min_samples=5).fit(points)
        labels = db.labels_

        clusters = []
        for label in set(labels):
            if label == -1:
                continue
            clusters.append(points[labels == label])

        return clusters

    def transform_points(self, points, source, target):
        out = []

        try:
            tf = self.tf_buffer.lookup_transform(
                target, source, rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return np.array([])

        for p in points:
            ps = PointStamped()
            ps.header.frame_id = source
            ps.point.x, ps.point.y, ps.point.z = map(float, p)

            pc = tf2_geometry_msgs.do_transform_point(ps, tf)
            out.append([pc.point.x, pc.point.y, pc.point.z])

        return np.array(out)

    def transform_point(self, point, source, target):
        try:
            tf = self.tf_buffer.lookup_transform(
                target, source, rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return point

        ps = PointStamped()
        ps.header.frame_id = source
        ps.point.x, ps.point.y, ps.point.z = map(float, point)

        pc = tf2_geometry_msgs.do_transform_point(ps, tf)
        return np.array([pc.point.x, pc.point.y, pc.point.z])

    def project_to_image(self, points):
        fx = self.caminfo.k[0]
        fy = self.caminfo.k[4]
        cx = self.caminfo.k[2]
        cy = self.caminfo.k[5]

        uv = []
        for x, y, z in points:
            if z <= 0.0:
                continue
            u = fx * x / z + cx
            v = fy * y / z + cy
            uv.append((u, v))

        return uv

    # ================= DIMENSION ESTIMATION =================

    def estimate_object_dimensions(self, det, depth_z):
        if depth_z <= 0.0:
            return None, None

        fx = self.caminfo.k[0]
        fy = self.caminfo.k[4]

        pixel_width = det.bbox.size_x
        pixel_height = det.bbox.size_y

        width_m = (pixel_width * depth_z) / fx
        height_m = (pixel_height * depth_z) / fy

        return width_m, height_m

    # ================= MATCHING =================

    def match_with_detections(self, uv_points, centroids_lidar, centroids_cam):
        if not uv_points:
            return

        for det in self.detections.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y

            xmin, xmax = cx - w / 2.0, cx + w / 2.0
            ymin, ymax = cy - h / 2.0, cy + h / 2.0

            for (u, v), centroid_lidar, centroid_cam in zip(
                uv_points, centroids_lidar, centroids_cam
            ):
                if xmin <= u <= xmax and ymin <= v <= ymax:

                    depth_z = centroid_cam[2]

                    width_m, height_m = self.estimate_object_dimensions(
                        det, depth_z
                    )

                    object_map = self.transform_point(
                        centroid_lidar,
                        source=self.scan.header.frame_id,
                        target="map",
                    )

                    self.get_logger().info(
                        f"Class: {det.results[0].hypothesis.class_id}, "
                        f"Conf: {det.results[0].hypothesis.score:.2f}, "
                        f"Map Pos: [{object_map[0]:.2f}, {object_map[1]:.2f}], "
                        f"Width: {width_m:.2f} m, "
                        f"Height: {height_m:.2f} m"
                    )


# ================= MAIN =================

def main():
    rclpy.init()
    node = LidarCameraFusion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()