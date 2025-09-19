import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from interfaces_pkg.msg import LaneInfo, PathPlanningResult
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

#---------------Variable Setting---------------
SUB_LANE_TOPIC_NAME = "yolov8_lane_info"  # lane_info_extractor 노드에서 퍼블리시하는 타겟 지점 토픽
PUB_TOPIC_NAME = "path_planning_result"   # 경로 계획 결과 퍼블리시 토픽
CAR_CENTER_POINT = (269, 440) # 이미지 상에서 차량 앞 범퍼의 중심이 위치한 픽셀 좌표
LANE_WIDTH_PX = 80.0   # 한쪽만 보일 때 중앙 근사를 위한 픽셀 오프셋
EMA_ALPHA     = 0.35     # 0~1, 작을수록 더 부드럽고 반응은 느려짐
#----------------------------------------------
class PathPlannerNode(Node):
    def __init__(self):
        super().__init__('path_planner_node')

        # 파라미터 선언
        self.sub_lane_topic = self.declare_parameter('sub_lane_topic', SUB_LANE_TOPIC_NAME).value
        self.pub_topic = self.declare_parameter('pub_topic', PUB_TOPIC_NAME).value
        self.car_center_point = self.declare_parameter('car_center_point', CAR_CENTER_POINT).value
        
        # QoS 설정
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )

        # 변수 초기화
        self.target_points = []  # 차선의 타겟 지점들 (차선 중앙)
        self.prev_x = None

        # 서브스크라이버 설정 (타겟 지점 구독)
        self.lane_sub = self.create_subscription(LaneInfo, self.sub_lane_topic, self.lane_callback, self.qos_profile)

        # 퍼블리셔 설정 (경로 계획 결과 퍼블리시)
        self.publisher = self.create_publisher(PathPlanningResult, self.pub_topic, self.qos_profile)

    def _build_center_from_left_right(self, L, R):
        if not (L or R):
            return None, None

        # y 축 그리드 만들기 (좌/우의 y 모음)
        ys = sorted(set([y for _,y in L] + [y for _,y in R]))
        if len(ys) < 2:
            return None, None

        Ld = {y:x for x,y in L}
        Rd = {y:x for x,y in R}
        xs = []
        out_ys = []
        for y in ys:
            if y in Ld and y in Rd:
                xc = 0.5*(Ld[y] + Rd[y])
            elif y in Ld:
                xc = Ld[y] + 0.5*LANE_WIDTH_PX
            elif y in Rd:
                xc = Rd[y] - 0.5*LANE_WIDTH_PX
            else:
                continue
            xs.append(float(xc))
            out_ys.append(float(y))

        if len(xs) < 2:
            return None, None
        return np.array(xs, dtype=float), np.array(out_ys, dtype=float)

    # [NEW] 지수이동평균 스무딩
    def _ema(self, xs: np.ndarray) -> np.ndarray:
        if self.prev_x is None or len(self.prev_x) != len(xs):
            self.prev_x = xs.copy()
            return xs
        y = EMA_ALPHA * xs + (1.0 - EMA_ALPHA) * self.prev_x
        self.prev_x = y
        return y
    
    def lane_callback(self, msg: LaneInfo):
        L = sorted([(p.x, p.y) for p in getattr(msg, 'left_lane_points', [])],  key=lambda t: t[1])
        R = sorted([(p.x, p.y) for p in getattr(msg, 'right_lane_points', [])], key=lambda t: t[1])

        if L or R:
            xs, ys = self._build_center_from_left_right(L, R)
            if xs is not None:
                xs = self._ema(xs)
                self._publish(xs, ys)
                return     
        # 타겟 지점 받아오기
        self.target_points = msg.target_points
        
        # 타겟 지점이 3개 이상 모이면 경로 계획 시작
        if len(self.target_points) >= 2:
            self.plan_path()

    def plan_path(self):
        # self.target_points가 TargetPoint 객체들의 리스트라고 가정
        if not self.target_points:
            self.get_logger().warn("No target points available")
            return
        
        # TargetPoint 객체에서 x, y 값 추출
        x_points, y_points = zip(*[(tp.target_x, tp.target_y) for tp in self.target_points])

        #차량 앞 범퍼의 중심이 위치한 픽셀 좌표 추가
        y_points_list, x_points_list = list(y_points), list(x_points) 
        y_points_list.append(self.car_center_point[1])
        x_points_list.append(self.car_center_point[0])
        y_points, x_points = tuple(y_points_list), tuple(x_points_list)
        
        # y 값을 기준으로 정렬 (y가 증가하는 순서로 정렬)
        sorted_points = sorted(zip(y_points, x_points), key=lambda point: point[0])

        # 정렬된 y, x 값을 다시 분리
        y_points, x_points = zip(*sorted_points)
        
        # 몇개의 점으로 경로 계획을 하는지 확인
        #self.get_logger().info(f"Planning path with {len(y_points)} points")

        # 스플라인 보간법을 사용하여 경로 생성
        #cs = CubicSpline(y_points, x_points, bc_type='natural')

        # 생성된 경로 점들 (추가적인 점들을 생성하여 부드러운 경로를 얻음)
        y_new = np.linspace(min(y_points), max(y_points), 100)
        #x_new = cs(y_new)
        x_new = np.interp(y_new, y_points, x_points)
        self._publish(np.array(x_new, dtype=float), np.array(y_new, dtype=float))
        # 경로를 따라가는 정보 (PathPlanningResult 메시지로 발행)
        #path_msg = PathPlanningResult()
        #path_msg.x_points = list(x_new)
        #path_msg.y_points = list(y_new)
        # 타겟 지점 초기화 (다음 경로 계산을 위해)
        self.target_points.clear()

        # 경로 퍼블리시
        #self.publisher.publish(path_msg)
    def _publish(self, xs: np.ndarray, ys: np.ndarray):
        path_msg = PathPlanningResult()
        path_msg.x_points = list(xs)
        path_msg.y_points = list(ys)
        self.publisher.publish(path_msg)



def main(args=None):
    rclpy.init(args=args)
    node = PathPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n\nshutdown\n\n")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
