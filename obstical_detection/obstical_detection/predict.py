import rclpy
from rclpy.node import Node
import numpy as np
import joblib

# Message types
from vx_custom_msgs.msg import FusionResult             # <-- change package name
from geometry_msgs.msg import PointStamped


class SVRInferenceNode(Node):

    def __init__(self):
        super().__init__('svr_inference_node')

        # ---------------- Load model & scalers ----------------
        self.model = joblib.load("src/obstical_detection/model/svr_model.pkl")

        self.conf_scaler = joblib.load("src/obstical_detection/model/conf_scaler.pkl")
        self.pos_scaler  = joblib.load("src/obstical_detection/model/pos_scaler.pkl")
        self.size_scaler = joblib.load("src/obstical_detection/model/size_scaler.pkl")
        self.out_scaler  = joblib.load("src/obstical_detection/model/out_scaler.pkl")

        self.get_logger().info("SVR model and scalers loaded ✔")

        # ---------------- Subscriber ----------------
        self.sub = self.create_subscription(
            FusionResult,
            '/fusion_result',
            self.fusion_callback,
            10
        )

        # ---------------- Publisher ----------------
        self.pub = self.create_publisher(
            PointStamped,
            '/svr_prediction',
            10
        )

    # ---------------------------------------------------------
    def fusion_callback(self, msg):
        """
        Called whenever /fusion message arrives
        """

        # 1️⃣ Build input vector (MATCHES TRAINING ORDER)
        X_new = np.array([[
            msg.confidence,
            msg.map_position.x,
            msg.map_position.y,
            msg.map_position.z,
            msg.width,
            msg.height,
            msg.depth
        ]])

        # 2️⃣ Split features
        X_conf = X_new[:, 0:1]   # confidence
        X_pos  = X_new[:, 1:4]   # map x,y,z
        X_size = X_new[:, 4:7]   # width,height,depth

        # 3️⃣ Scale using TRAINED scalers
        X_scaled = np.hstack((
            self.conf_scaler.transform(X_conf),
            self.pos_scaler.transform(X_pos),
            self.size_scaler.transform(X_size)
        ))

        # 4️⃣ Predict
        y_pred_scaled = self.model.predict(X_scaled)

        # 5️⃣ Inverse scale output
        y_pred = self.out_scaler.inverse_transform(y_pred_scaled)

        e_x, e_y, e_z = y_pred[0]

        # Log prediction
        self.get_logger().info(
            f"Predicted GT → x={e_x:.3f}, y={e_y:.3f}, z={e_z:.3f}"
        )

        # 6️⃣ Publish prediction
        out_msg = PointStamped()
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = "map"

        out_msg.point.x = float(e_x)
        out_msg.point.y = float(e_y)
        out_msg.point.z = float(e_z)

        self.pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SVRInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
