import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, CameraInfo
from vision_msgs.msg import Detection2DArray
from vx_custom_msgs.msg import CamLaser
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
import numpy as np
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

        # --- Subscriptions ---
        self.create_subscription(LaserScan, '/scan', self.scan_callback,scan_qos)
        self.create_subscription(Detection2DArray,
                                 '/detectnet/detections',
                                 self.detection_callback,
                                 10)
        self.create_subscription(CameraInfo,
                                 '/camera/camera/color/camera_info',
                                 self.camera_info_callback,
                                 10)
        

        # --- TF2 Setup ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Camera intrinsics
        self.camera_info = None

        # Cache latest messages
        self.latest_scan = None
        self.latest_detections = None

        self.get_logger().info("Fusion Node Started")

    # -----------------------------
    #    Camera info callback
    # -----------------------------
    def camera_info_callback(self, msg):
        self.camera_info = msg
        self.try_fusion()

    # -----------------------------
    #        LiDAR callback
    # -----------------------------
    def scan_callback(self, msg):
        self.latest_scan = msg
        self.try_fusion()

    # -----------------------------
    #    Detections callback
    # -----------------------------
    def detection_callback(self, msg):
        self.latest_detections = msg
        self.try_fusion()

    # -----------------------------
    #        FUSION LOGIC
    # -----------------------------
    def try_fusion(self):
        if self.camera_info is None:
            return
        if self.latest_scan is None:
            return
        if self.latest_detections is None:
            return

        try:
            # Look up transform LIDAR → Camera
            transform = self.tf_buffer.lookup_transform(
                target_frame=self.camera_info.header.frame_id,        # <-- adjust to your camera frame
                source_frame=self.latest_scan.header.frame_id,
                time=rclpy.time.Time()
            )

        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        # --- Convert scan to 3D points ---
        points_cam_frame = self.lidar_to_camera_points(
            self.latest_scan, transform
        )

        # --- Project to image plane ---
        uv_points = self.project_points(points_cam_frame)

        # --- Fuse with detections ---
        self.fuse_and_print(self.latest_detections, uv_points, points_cam_frame)

    # -----------------------------------------------------
    #   Convert 2D LiDAR ranges → 3D points in camera frame
    # -----------------------------------------------------
    def lidar_to_camera_points(self, scan, transform):
        angles = np.arange(scan.angle_min, scan.angle_max, scan.angle_increment)
        ranges = np.array(scan.ranges)

        valid = np.isfinite(ranges)
        ranges = ranges[valid]
        angles = angles[valid]

        # LiDAR points in LiDAR frame (2D lidar -> z = 0)
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

    # -----------------------------------------------------
    #          Camera projection (3D → image)
    # -----------------------------------------------------
    def project_points(self, pts):
        if pts.shape[0] == 0:
            return []

        fx = self.camera_info.k[0]
        fy = self.camera_info.k[4]
        cx = self.camera_info.k[2]
        cy = self.camera_info.k[5]

        # Pinhole model projection
        zs = pts[:, 2]
        invalid = zs <= 0
        zs[invalid] = 1e-6

        u = (pts[:, 0] * fx / zs) + cx
        v = (pts[:, 1] * fy / zs) + cy

        return np.vstack((u, v)).T

    # -----------------------------------------------------
    #       Fusion: Match LiDAR pts with BBoxes
    # -----------------------------------------------------
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

            # Points inside the bbox
            mask = (uv_points[:, 0] >= xmin) & (uv_points[:, 0] <= xmax) & \
                   (uv_points[:, 1] >= ymin) & (uv_points[:, 1] <= ymax)

            matched_pts = pts_3d[mask]

            if len(matched_pts) > 0:
                depth = np.mean(matched_pts[:, 2])
                fused_results.append({
                    "label": det.results[0].hypothesis.class_id.encode(),
                    "bbox": [xmin, ymin, xmax, ymax],
                    "depth": float(depth)
                })
        self.Puslisher_data(fused_results)
        # Print fusion output
        for obj in fused_results:
            self.get_logger().info(
                f"[Fusion] {obj['label']}: depth={obj['depth']:.2f} m "
                f"BBox={obj['bbox']}"
            )
    def Puslisher_data(self, fused_results):
        for obj in fused_results:
            msg = CamLaser()

            msg.class_id = str(obj["label"])
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
