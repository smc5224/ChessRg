import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import time

# 전역 변수
initial_button_image = None  # 초기 버튼 이미지 (비교 기준)
button_coordinates = None  # 버튼 영역 좌표 (x, y, width, height)
perspective_matrix = None  # 평면 보정을 위한 변환 행렬


def setup_board_and_button(video):
    """
    체스판과 버튼 영역을 설정하는 함수.
    사용자가 마우스로 점 4개를 클릭해 평면 보정 후 버튼 영역을 지정함.
    """
    global initial_button_image, button_coordinates, perspective_matrix  # 전역 변수 선언

    # 평면 보정을 위한 변수 초기화
    selected_point_count = 0  # 선택된 점의 개수 (최대 4개)
    selected_points = np.zeros((4, 2), dtype=np.float32)  # 선택된 점의 좌표 저장
    perspective_matrix = None  # 평면 보정을 위한 변환 행렬 초기화

    # 버튼 영역 초기값
    button_rect = [50, 50, 200, 200]  # 버튼 사각형의 초기 좌표와 크기 (x, y, width, height)
    is_dragging = False  # 버튼을 드래그 중인지 여부
    is_resizing = False  # 버튼 크기를 조정 중인지 여부
    start_drag_point = None  # 드래그 시작 지점
    corner_detection_size = 15  # 버튼 크기 조정용 모서리 영역 크기

    def draw_rectangle(image, rect):
        """
        사각형과 모서리를 그리는 함수.
        """
        x, y, w, h = rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 버튼 사각형
        cv2.rectangle(image, (x + w - corner_detection_size, y + h - corner_detection_size),
                      (x + w, y + h), (0, 0, 255), -1)  # 모서리 크기 조정 핸들

    def is_in_corner(x, y, rect):
        """
        좌표 (x, y)가 사각형의 크기 조정용 모서리 영역에 있는지 확인.
        """
        rect_x, rect_y, rect_width, rect_height = rect
        return (rect_x + rect_width - corner_detection_size <= x <= rect_x + rect_width and
                rect_y + rect_height - corner_detection_size <= y <= rect_y + rect_height)

    def on_mouse(event, x, y, flags, param):
        """
        마우스 이벤트 처리 콜백 함수.
        """
        nonlocal is_dragging, is_resizing, start_drag_point, button_rect, selected_point_count, selected_points
        global perspective_matrix

        if perspective_matrix is None and selected_point_count < 4:  # 평면 보정을 위한 점 선택
            if event == cv2.EVENT_LBUTTONDOWN:
                selected_points[selected_point_count] = [x, y]
                selected_point_count += 1
                if selected_point_count == 4:  # 점 4개가 선택되면 평면 보정 수행
                    destination_points = np.array([
                        [0, 0],
                        [300, 0],
                        [300 , 300],
                        [0, 300]
                    ], dtype=np.float32)
                    perspective_matrix = cv2.getPerspectiveTransform(selected_points, destination_points)

        elif perspective_matrix is not None:  # 버튼 영역 조정 (평면 보정 후)
            if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 클릭
                if is_in_corner(x, y, button_rect):  # 모서리를 클릭한 경우
                    is_resizing = True
                    start_drag_point = (x, y)
                elif button_rect[0] <= x <= button_rect[0] + button_rect[2] and button_rect[1] <= y <= button_rect[1] + button_rect[3]:  # 버튼 내부 클릭
                    is_dragging = True
                    start_drag_point = (x, y)

            elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 이동
                if is_dragging:  # 버튼 이동
                    dx, dy = x - start_drag_point[0], y - start_drag_point[1]
                    button_rect[0] += dx
                    button_rect[1] += dy
                    start_drag_point = (x, y)
                elif is_resizing:  # 버튼 크기 조정
                    dx, dy = x - start_drag_point[0], y - start_drag_point[1]
                    button_rect[2] += dx
                    button_rect[3] += dy
                    button_rect[2] = max(20, button_rect[2])  # 최소 너비 제한
                    button_rect[3] = max(20, button_rect[3])  # 최소 높이 제한
                    start_drag_point = (x, y)

            elif event == cv2.EVENT_LBUTTONUP:  # 마우스 버튼 해제
                is_dragging = False
                is_resizing = False

    # 마우스 콜백 설정
    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', on_mouse)

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # 점 찍기 표시
        for i in range(selected_point_count):
            cv2.circle(frame, (int(selected_points[i, 0]), int(selected_points[i, 1])), 5, (0, 0, 255), -1)

        if perspective_matrix is not None:  # 평면 보정 후
            warped_frame = cv2.warpPerspective(frame, perspective_matrix, (300, 300))           
            cv2.imshow('Board View', warped_frame)  # 보정된 체스판 출력
            draw_rectangle(frame, button_rect)  # 버튼 영역 표시

        cv2.imshow('Video', frame)  # 원본 영상 출력
        key = cv2.waitKey(1)
        if key == 13 and perspective_matrix is not None:  # Enter 키로 버튼 영역 설정 완료
            x, y, w, h = button_rect
            initial_button_image = frame[y:y + h, x:x + w]  # 버튼 초기 이미지 저장
            button_coordinates = button_rect
            break
        elif key == 27:  # ESC 키로 종료
            break

    
    cv2.destroyWindow("Video")

    

def detect_turn_change(video):
    """
    버튼 상태 변화(턴 변경)를 감지하는 함수.
    상태 전이 로직을 활용하여 턴 변경을 감지.
    """
    global initial_button_image, button_coordinates

    if initial_button_image is None or button_coordinates is None:
        print("버튼 초기 상태가 설정되지 않았습니다.")
        return False

    # 초기 상태 설정
    previous_state = "VISIBLE"  # 처음에는 버튼이 보이는 상태
    current_state = "VISIBLE"

    consecutive_hidden_count = 0  # 버튼이 가려진 상태가 연속적으로 몇 프레임 유지되는지 카운트
    consecutive_visible_count = 0  # 버튼이 보이는 상태가 연속적으로 몇 프레임 유지되는지 카운트
    threshold_count = 10  # 상태 전이를 결정하는데 필요한 연속 프레임 수
    cooldown_period = 2  # 버튼 상태가 변경된 후 유지되는 시간 (초)

    last_state_change_time = time.time()  # 마지막으로 상태가 변경된 시간 기록

    while True:
        ret, frame = video.read()
        if not ret:
            break

        # 버튼 영역 추출
        x, y, w, h = button_coordinates
        current_button_region = frame[y:y + h, x:x + w]

        # 버튼 상태 비교 (SSIM 이용)
        current_gray = cv2.cvtColor(current_button_region, cv2.COLOR_BGR2GRAY)
        initial_gray = cv2.cvtColor(initial_button_image, cv2.COLOR_BGR2GRAY)
        similarity, _ = compare_ssim(initial_gray, current_gray, full=True)

        # 상태 판단 (임계값 기준)
        if similarity < 0.5:
            current_state = "HIDDEN"
            consecutive_hidden_count += 1
        else:
            current_state = "VISIBLE"
            consecutive_visible_count += 1

        # 상태 전이 감지 (연속적인 상태 변화 확인)
        current_time = time.time()
        if previous_state == "VISIBLE" and consecutive_hidden_count >= threshold_count:
            if current_time - last_state_change_time > cooldown_period:
                print("턴 변화 감지: 버튼이 가려졌습니다.")
                initial_button_image = current_button_region  # 새로운 상태를 초기 상태로 갱신
                previous_state = "HIDDEN"
                last_state_change_time = current_time
                consecutive_hidden_count = 0

        elif previous_state == "HIDDEN" and consecutive_visible_count >= threshold_count:
            if current_time - last_state_change_time > cooldown_period:
                print("버튼이 다시 보입니다.")
                previous_state = "VISIBLE"
                last_state_change_time = current_time
                consecutive_visible_count = 0

        # 버튼 영역 표시
        cv2.imshow('Button View', current_button_region)
        if cv2.waitKey(1) == 27:  # ESC 키로 종료
            break

    video.release()
    cv2.destroyAllWindows()
    return False


def turn_based_game(video):
    """
    턴 기반 게임 흐름을 처리하는 함수.
    """
    setup_board_and_button(video)  # 체스판과 버튼 영역 설정
    print("초기 설정 완료. 버튼 변화를 감지합니다.")
    return detect_turn_change(video)  # 상태 전이 기반 턴 변화 감지


# 프로그램 실행
cap = cv2.VideoCapture(0)
result = turn_based_game(cap)





