#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from etsi_its_cpm_ts_msgs.msg import CollectivePerceptionMessage

import numpy as np
import time
from scipy.spatial.distance import mahalanobis


# =====================================================
# EKF TRACK
# =====================================================
class EKFTrack:
    def __init__(self, track_id, initial_pos, R, cpm_object_id=None):
        self.id = track_id
        self.x = np.array([initial_pos[0], initial_pos[1], 0.0, 0.0])  # [x, y, vx, vy]
        self.P = np.eye(4) * 0.5
        self.R = R
        self.last_update = time.time()
        self.hits = 1
        self.cpm_object_id = cpm_object_id

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
        self.last_update = time.time()
        self.hits += 1


# =====================================================
# COOPERATIVE FUSION NODE
# =====================================================
class CooperativeFusionNode(Node):

    def __init__(self):
        super().__init__("cooperative_fusion_node")

        self.create_subscription(
            CollectivePerceptionMessage,
            "/cpm",
            self.cpm_callback,
            10
        )

        self.marker_pub = self.create_publisher(
            MarkerArray,
            "/fused_tracks",
            10
        )

        self.tracks = {}
        self.next_track_id = 0

        # ---- PARAMETERS ----
        self.assoc_threshold = 3.0
        self.track_timeout = 1.0
        self.min_hits = 3

        self.default_R = np.array([[0.3, 0],
                                   [0, 0.3]])

        self.create_timer(0.1, self.publish_markers)

        self.get_logger().info("✅ Cooperative CPM Fusion Node started")

    # -------------------------------------------------
    def cpm_callback(self, msg: CollectivePerceptionMessage):
        now = time.time()

        # 1️⃣ Predict all tracks FIRST
        for track in self.tracks.values():
            dt = now - track.last_update
            track.predict(dt)

        # 2️⃣ Process CPM objects
        for container in msg.payload.cpm_containers.value.array:
            if container.container_id.value != container.CHOICE_CONTAINER_DATA_PERCEIVED_OBJECT_CONTAINER:
                continue

            poc = container.container_data_perceived_object_container

            for po in poc.perceived_objects.array:
                x = po.position.x_coordinate.value.value * 0.01
                y = po.position.y_coordinate.value.value * 0.01
                z = [x, y]

                cpm_id = po.object_id.value if po.object_id_is_present else None

                # Measurement covariance from confidence
                if po.object_perception_quality_is_present:
                    conf = po.object_perception_quality.value / 100.0
                    R = np.array([[0.3 * (1 - conf + 0.1), 0],
                                  [0, 0.3 * (1 - conf + 0.1)]])
                else:
                    R = self.default_R

                matched = None
                min_dist = float("inf")

                # 3️⃣ Association
                for track in self.tracks.values():

                    # Strong hint: same CPM object ID
                    if cpm_id is not None and track.cpm_object_id == cpm_id:
                        matched = track
                        break

                    try:
                        d = mahalanobis(
                            np.array(z),
                            track.x[:2],
                            np.linalg.inv(track.P[:2, :2])
                        )
                    except np.linalg.LinAlgError:
                        d = np.linalg.norm(np.array(z) - track.x[:2])

                    if d < self.assoc_threshold and d < min_dist:
                        min_dist = d
                        matched = track

                # 4️⃣ Update or create
                if matched:
                    matched.update(z)
                else:
                    self.tracks[self.next_track_id] = EKFTrack(
                        self.next_track_id,
                        z,
                        R,
                        cpm_object_id=cpm_id
                    )
                    self.next_track_id += 1

    # -------------------------------------------------
    def publish_markers(self):
        now = time.time()
        marker_array = MarkerArray()

        # 5️⃣ Track timeout
        dead = [
            tid for tid, trk in self.tracks.items()
            if (now - trk.last_update) > self.track_timeout
        ]
        for tid in dead:
            del self.tracks[tid]

        # 6️⃣ Publish confirmed tracks
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
            marker.scale.z = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 0.85

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