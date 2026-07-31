import rclpy
from rclpy.node import Node

from sensor_msgs.msg import LaserScan, CameraInfo
from vision_msgs.msg import Detection2DArray

import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
import numpy as np
from sklearn.cluster import DBSCAN
from vx_custom_msgs.msg import CamLaser
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy


class LidarCameraFusion(Node):

    def __init__(self):
        super().__init__('lidar_camera_fusion_node')
        scan_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.create_subscription(LaserScan, '/scan', self.scan_callback, scan_qos)
        self.create_subscription(Detection2DArray, '/detectnet/detections',
                                 self.detection_callback, 10)
        self.create_subscription(CameraInfo,
                                 '/camera/camera/color/camera_info',
                                 self.camera_info_callback, 10)
        self.pub = self.create_publisher(CamLaser,'/fused_data',10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.camera_info = None
        self.latest_scan = None
        self.latest_detections = None

        self.get_logger().info("Fusion Node Started")

        self.label_map = {
            "\x01": "Plant",
            "\x02": "Traffic_light_red",
            "\x03": "Traffic_light_green",
            "\x04": "Person",
            "\x05": "Car",
            "\x06": "House"
        }


    # ---------------------------------------------------------------------
    def camera_info_callback(self, msg):
        self.camera_info = msg
        self.try_fusion()

    def scan_callback(self, msg):
        self.latest_scan = msg
        self.try_fusion()

    def detection_callback(self, msg):
        self.latest_detections = msg
        self.try_fusion()

    # ---------------------------------------------------------------------
    def try_fusion(self):
        if self.camera_info is None or self.latest_scan is None or self.latest_detections is None:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame=self.camera_info.header.frame_id,
                source_frame=self.latest_scan.header.frame_id,
                time=rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        points_cam = self.lidar_to_camera_points(self.latest_scan, transform)
        uv_points = self.project_points(points_cam)

        self.fuse_and_print(self.latest_detections, uv_points, points_cam)

    # ---------------------------------------------------------------------
    def lidar_to_camera_points(self, scan, transform):
        angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)
        ranges = np.array(scan.ranges)

        valid = np.isfinite(ranges)
        ranges = ranges[valid]
        angles = angles[valid]

        xs = ranges * np.cos(angles)
        ys = ranges * np.sin(angles)
        zs = np.zeros_like(xs)

        points_cam = []

        for x, y, z in zip(xs, ys, zs):
            p = PointStamped()
            p.header.frame_id = scan.header.frame_id
            p.point.x = float(x)
            p.point.y = float(y)
            p.point.z = float(z)

            try:
                p_cam = tf2_geometry_msgs.do_transform_point(p, transform)
                points_cam.append([p_cam.point.x, p_cam.point.y, p_cam.point.z])
            except:
                pass

        return np.array(points_cam)

    # ---------------------------------------------------------------------
    def project_points(self, pts):
        if pts.shape[0] == 0:
            return []

        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        zs = pts[:, 2]
        invalid = zs <= 0
        zs[invalid] = 1e-6

        u = (pts[:, 0] * fx / zs) + cx
        v = (pts[:, 1] * fy / zs) + cy

        return np.vstack((u, v)).T

    # ---------------------------------------------------------------------
    def fuse_and_print(self, dets, uv_points, pts_3d):
        if len(uv_points) == 0:
            return

        fused_results = []

        for det in dets.detections:

            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y

            xmin = cx - w / 2
            xmax = cx + w / 2
            ymin = cy - h / 2
            ymax = cy + h / 2

           
            mask = (uv_points[:, 0] >= xmin) & (uv_points[:, 0] <= xmax) & \
                   (uv_points[:, 1] >= ymin) & (uv_points[:, 1] <= ymax)

            matched_pts = pts_3d[mask]

            if len(matched_pts) > 0:

                if len(matched_pts) >= 3:  # DBSCAN needs at least 3 points
                    clustering = DBSCAN(eps=0.15, min_samples=3).fit(matched_pts)

                    labels = clustering.labels_

                    # Ignore noise cluster (-1)
                    unique_labels = [l for l in set(labels) if l != -1]

                    if len(unique_labels) > 0:
                        # Find cluster with **closest mean Z** (closest to camera)
                        cluster_means = []
                        for lbl in unique_labels:
                            cluster_pts = matched_pts[labels == lbl]
                            mean_depth = np.mean(cluster_pts[:, 2])
                            cluster_means.append((mean_depth, cluster_pts))

                        cluster_means.sort(key=lambda x: x[0])
                        best_cluster = cluster_means[0][1]

                        depth = float(np.mean(best_cluster[:, 2]))

                    else:
                        depth = float(np.mean(matched_pts[:, 2]))

                else:
                    depth = float(np.mean(matched_pts[:, 2]))

                fused_results.append({
                    "label": det.results[0].hypothesis.class_id,
                    "bbox": [xmin, ymin, xmax, ymax],
                    "depth": depth
                })
        self.Puslisher_data(fused_results)

        for obj in fused_results:
            self.get_logger().info(
                f"[Fusion]  {obj['label']}: depth={obj['depth']:.2f} m BBox={obj['bbox']}"
            )
    def Puslisher_data(self, fused_results):
        for obj in fused_results:
            msg = CamLaser()
            obj_class = self.label_map.get(str(obj['label']), f"class_{obj['label']}")
            msg.class_id = obj_class
            msg.distance = float(obj["depth"])

            self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraFusion()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
