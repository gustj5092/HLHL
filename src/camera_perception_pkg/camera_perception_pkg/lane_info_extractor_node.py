import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy

from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from interfaces_pkg.msg import TargetPoint, LaneInfo, DetectionArray, BoundingBox2D, Detection
from geometry_msgs.msg import Point
from .lib import camera_perception_func_lib as CPFL

#---------------Variable Setting---------------
# Subscribe할 토픽 이름
SUB_TOPIC_NAME = "detections"

# Publish할 토픽 이름
PUB_TOPIC_NAME = "yolov8_lane_info"
ROI_IMAGE_TOPIC_NAME = "roi_image"  # 추가: ROI 이미지 퍼블리시 토픽

# 화면에 이미지를 처리하는 과정을 띄울것인지 여부: True, 또는 False 중 택1하여 입력
SHOW_IMAGE = True
#----------------------------------------------


class Yolov8InfoExtractor(Node):
    def __init__(self):
        super().__init__('lane_info_extractor_node')

        self.sub_topic = self.declare_parameter('sub_detection_topic', SUB_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.show_image = self.declare_parameter('show_image', SHOW_IMAGE).value
        # Add a parameter for the camera number
        self.cam_num = self.declare_parameter('cam_num', 0).value

        self.cv_bridge = CvBridge()

        # QoS settings
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        
        self.subscriber = self.create_subscription(DetectionArray, self.sub_topic, self.yolov8_detections_callback, self.qos_profile)
        self.publisher = self.create_publisher(LaneInfo, self.pub_topic, self.qos_profile)

        # ROI 이미지 퍼블리셔 추가
        self.roi_image_publisher = self.create_publisher(Image, ROI_IMAGE_TOPIC_NAME, self.qos_profile)

    def yolov8_detections_callback(self, detection_msg: DetectionArray):
        if len(detection_msg.detections) == 0:
            return
        
        lane_edge_image = CPFL.draw_edges(detection_msg, cls_name='lane', color=255)

        (h, w) = (lane_edge_image.shape[0], lane_edge_image.shape[1]) #(480, 640)
        dst_mat = [[round(w * 0.3), round(h * 0.0)], [round(w * 0.7), round(h * 0.0)], [round(w * 0.7), h], [round(w * 0.3), h]]
        #src_mat = [[241, 225], [342, 219], [637, 348], [27, 341]]
        src_mat = [[270, 184], [364, 184], [635, 414], [2, 352]]
        lane_bird_image = CPFL.bird_convert(lane_edge_image, srcmat=src_mat, dstmat=dst_mat)
        roi_image = CPFL.roi_rectangle_below(lane_bird_image, cutting_idx=200)

        if self.show_image:
            #cv2.imshow('lane_edge_image', lane_edge_image)
            #cv2.imshow('lane_bird_img', lane_bird_image)
            # Create a unique window name for each camera's ROI image
            cv2.imshow(f'roi_img_{self.cam_num}', roi_image)
            cv2.waitKey(1)

        # roi_image를 uint8 형식으로 변환
        roi_image = cv2.convertScaleAbs(roi_image)  # 64FC1 -> uint8로 변환

        # roi_image를 ROS Image 메시지로 변환
        try:
            roi_image_msg = self.cv_bridge.cv2_to_imgmsg(roi_image, encoding="mono8")
            # ROI 이미지를 퍼블리시
            self.roi_image_publisher.publish(roi_image_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to convert and publish ROI image: {e}")
        
        grad = CPFL.dominant_gradient(roi_image, theta_limit=70)
                
        target_points = []
        for target_point_y in range(5, 155, 50):  # 예시로 5에서 155까지 50씩 증가
            target_point_x = CPFL.get_lane_center(roi_image, detection_height=target_point_y, 
                                                detection_thickness=10, road_gradient=grad, lane_width=300)
            
            target_point = TargetPoint()
            target_point.target_x = round(target_point_x)
            target_point.target_y = round(target_point_y)
            target_points.append(target_point)

        left_xy, right_xy = self._extract_left_right_points(roi_image)
        lane = LaneInfo()
        lane.slope = grad
        lane.target_points = target_points
        for (x, y) in left_xy:
            p = Point(); p.x = float(x); p.y = float(y); p.z = 0.0
            lane.left_lane_points.append(p)
        for (x, y) in right_xy:
            p = Point(); p.x = float(x); p.y = float(y); p.z = 0.0
            lane.right_lane_points.append(p)
        self.publisher.publish(lane)

    def _extract_left_right_points(self, roi_image):
        """
        단순/견고: 외곽 컨투어 2개를 좌/우로 간주하고,
        고정된 y 샘플들에서 x를 보간해 포인트 리스트 반환.
        """
        sample_ys = [5, 55, 105, 155]  # 기존 center 샘플 y와 동일하게 맞춤  :contentReference[oaicite:6]{index=6}
        # 이진화 (이미 mono8이므로 임계값만)
        _, bin_img = cv2.threshold(roi_image, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [c for c in contours if cv2.arcLength(c, False) > 50]

        if not contours:
            return [], []

        # 평균 x로 정렬 → 좌/우
        contours = sorted(contours, key=lambda c: float(c[:,0,0].mean()))
        if len(contours) == 1:
            # 하나만 있을 때는 좌/우 판단만 해서 한쪽만 리턴
            mid_x = roi_image.shape[1] / 2.0
            cx = float(contours[0][:,0,0].mean())
            pts = self._points_on_y(contours[0], sample_ys)
            return (pts, []) if cx < mid_x else ([], pts)

        c_left, c_right = contours[0], contours[1]
        left_pts  = self._points_on_y(c_left,  sample_ys)
        right_pts = self._points_on_y(c_right, sample_ys)
        return left_pts, right_pts

    def _points_on_y(self, contour, sample_ys):
        pts = contour[:,0,:].astype(float)  # (N,2) x,y
        # y로 정렬
        pts = pts[pts[:,1].argsort()]
        xs, ys = pts[:,0], pts[:,1]
        out = []
        for yq in sample_ys:
            idx = ys.searchsorted(yq)
            if 0 < idx < len(ys):
                y0, y1 = ys[idx-1], ys[idx]
                x0, x1 = xs[idx-1], xs[idx]
                if y1 != y0:
                    t = (yq - y0) / (y1 - y0)
                    xq = x0 + t*(x1 - x0)
                    out.append((float(xq), float(yq)))
        return out

def main(args=None):
    rclpy.init(args=args)
    node = Yolov8InfoExtractor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()
  
if __name__ == '__main__':
    main()