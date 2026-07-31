#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped
from etsi_its_cpm_ts_msgs.msg import CollectivePerceptionMessage

import numpy as np
from scipy.spatial.distance import mahalanobis

# =====================================================
# EKF TRACK
# =====================================================
class EKFTrack:
    def __init__(self, track_id, initial_pos, R):
        self.id = track_id
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0])  # [x, y, vx, vy]
        self.P = np.eye(4) * 0.5
        self.R = R
        self.last_update = None
        self.hits = 1

    def predict(self, dt):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])
        Q = np.eye(4) * 0.05
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z):
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        y = np.array(z) - (H @ self.x)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ H) @ self.P
        self.hits += 1


# =====================================================
# COOPERATIVE FUSION NODE
# =====================================================
class CooperativeFusionNode(Node):

    def __init__(self):
        super().__init__("cooperative_fusion_node")

        # ---- SUBSCRIPTIONS ----
        self.create_subscription(
            CollectivePerceptionMessage,
            "/cpm",
            self.cpm_callback,
            10
        )

        self.create_subscription(
            PointStamped,
            "/svr_prediction",
            self.svr_callback,
            10
        )

        # ---- PUBLISHER ----
        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/fused_tracks",
            10
        )

        # ---- TRACK STORAGE ----
        self.tracks = {}
        self.next_track_id = 0

        # ---- PARAMETERS ----
        self.assoc_threshold = 3.0
        self.track_timeout = 1.0
        self.min_hits = 3

        # ---- SVR OUTPUT (DEFAULT) ----
        self.ex = 0.3
        self.ey = 0.3

        # ---- EGO POSITION (MAP FRAME) ----
        self.ego_x = 0.0
        self.ego_y = 0.0

        self.create_timer(0.1, self.publish_markers)
        self.get_logger().info("✅ Cooperative CPM Fusion Node (corrected) started")

    # -------------------------------------------------
    def svr_callback(self, msg: PointStamped):
        self.ex = max(msg.point.x, 0.05)
        self.ey = max(msg.point.y, 0.05)

    # -------------------------------------------------
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # -------------------------------------------------
    def cpm_callback(self, msg: CollectivePerceptionMessage):
        current_time = self.now()

        # 1️⃣ Predict all tracks
        for track in self.tracks.values():
            if track.last_update is not None:
                dt = current_time - track.last_update
                track.predict(max(dt, 0.01))

        # 2️⃣ Process CPM objects
        for container in msg.payload.cpm_containers.value.array:
            if container.container_id.value != container.CHOICE_CONTAINER_DATA_PERCEIVED_OBJECT_CONTAINER:
                continue

            poc = container.container_data_perceived_object_container

            for po in poc.perceived_objects.array:
                x = po.position.x_coordinate.value.value * 0.01
                y = po.position.y_coordinate.value.value * 0.01
                z = [x, y]

                # ---- Ego filtering ----
                if np.hypot(x - self.ego_x, y - self.ego_y) < 1.5:
                    continue

                # ---- Measurement covariance from SVR + confidence ----
                if po.object_perception_quality_is_present:
                    conf = np.clip(po.object_perception_quality.value / 100.0, 0.1, 1.0)
                else:
                    conf = 0.5

                R = np.array([
                    [self.ex * (1.0 - conf + 0.1), 0],
                    [0, self.ey * (1.0 - conf + 0.1)]
                ])

                matched = None
                min_dist = float("inf")

                # 3️⃣ Data association (Mahalanobis)
                for track in self.tracks.values():
                    try:
                        S_inv = np.linalg.inv(track.P[:2, :2])
                        d = mahalanobis(np.array(z), track.x[:2], S_inv)
                    except np.linalg.LinAlgError:
                        d = np.linalg.norm(np.array(z) - track.x[:2])

                    if d < self.assoc_threshold and d < min_dist:
                        min_dist = d
                        matched = track

                # 4️⃣ Update or create
                if matched:
                    matched.R = R
                    matched.update(z)
                    matched.last_update = current_time
                else:
                    trk = EKFTrack(self.next_track_id, z, R)
                    trk.last_update = current_time
                    self.tracks[self.next_track_id] = trk
                    self.next_track_id += 1

    # -------------------------------------------------
    def publish_markers(self):
        current_time = self.now()
        marker_array = MarkerArray()

        # ---- Remove stale tracks ----
        dead = [
            tid for tid, trk in self.tracks.items()
            if (current_time - trk.last_update) > self.track_timeout
        ]
        for tid in dead:
            del self.tracks[tid]

        # ---- Publish confirmed tracks ----
        for tid, track in self.tracks.items():
            if track.hits < self.min_hits:
                continue

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.id = tid
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            marker.pose.position.x = track.x[0]
            marker.pose.position.y = track.x[1]
            marker.pose.position.z = 0.5

            marker.scale.x = 0.6
            marker.scale.y = 0.6
            marker.scale.z = 1.2

            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.9

            marker_array.markers.append(marker)

        self.marker_pub.publish(marker_array)


# =====================================================
# MAIN
# =====================================================
def main(args=None):
    rclpy.init(args=args)
    node = CooperativeFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()