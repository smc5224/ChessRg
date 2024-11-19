import cv2
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim

# 전역 변수
initial_button_state = None  # 버튼 초기 상태 저장
button_rect = None  # 버튼 영역 좌표 저장
pers_mat = None  # 평면 보정 행렬 초기화


def setup_board_and_button(video_source=0, output_size=300):
    """
    마우스로 점 4개를 찍어 평면 보정을 수행한 후,
    드래그 및 크기 조정을 통해 버튼 영역을 설정.

    Parameters:
    - video_source: 동영상 소스 (파일 경로 또는 웹캠).

    Updates:
    - initial_button_state: 초기 버튼 상태 저장.
    - button_rect: 버튼 영역 좌표 저장.
    """
    global initial_button_state, button_rect, pers_mat  # 전역 변수 선언

    cnt = 0
    src_pts = np.zeros((4, 2), dtype=np.float32)
    pers_mat = None  # 명시적으로 초기화

    rect = [50, 50, 200, 200]  # 초기 사각형
    dragging = False
    resizing = False
    start_point = None
    corner_size = 15  # 모서리 감지 영역 크기

    def draw_rectangle(image, rect):
        x, y, w, h = rect
        # 사각형 그리기
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # 오른쪽 아래 코너 표시
        cv2.rectangle(image, (x + w - corner_size, y + h - corner_size),
                      (x + w, y + h), (0, 0, 255), -1)

    def is_in_corner(x, y, rect):
        """
        (x, y)가 사각형의 오른쪽 아래 모서리 영역 안에 있는지 확인.
        """
        rx, ry, rw, rh = rect
        return (rx + rw - corner_size <= x <= rx + rw and
                ry + rh - corner_size <= y <= ry + rh)

    def on_mouse(event, x, y, flags, param):
        nonlocal dragging, resizing, start_point, rect, cnt, src_pts
        global pers_mat  # 함수 내부에서 전역 변수 사용 선언

        if pers_mat is None and cnt < 4:  # 점 찍기
            if event == cv2.EVENT_LBUTTONDOWN:
                src_pts[cnt] = [x, y]
                cnt += 1
                if cnt == 4:  # 점 4개 선택 완료
                    dst_pts = np.array([
                        [0, 0],
                        [output_size - 1, 0],
                        [output_size - 1, output_size - 1],
                        [0, output_size - 1]
                    ], dtype=np.float32)
                    pers_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)

        elif pers_mat is not None:  # 사각형 조정
            if event == cv2.EVENT_LBUTTONDOWN:  # 클릭 시작
                if is_in_corner(x, y, rect):  # 모서리 클릭
                    resizing = True
                    start_point = (x, y)
                elif rect[0] <= x <= rect[0] + rect[2] and rect[1] <= y <= rect[1] + rect[3]:  # 사각형 내부 클릭
                    dragging = True
                    start_point = (x, y)

            elif event == cv2.EVENT_MOUSEMOVE:  # 드래그 중
                if dragging:  # 사각형 이동
                    dx, dy = x - start_point[0], y - start_point[1]
                    rect[0] += dx
                    rect[1] += dy
                    start_point = (x, y)
                elif resizing:  # 크기 조정
                    dx, dy = x - start_point[0], y - start_point[1]
                    rect[2] += dx
                    rect[3] += dy
                    rect[2] = max(20, rect[2])  # 최소 너비 제한
                    rect[3] = max(20, rect[3])  # 최소 높이 제한
                    start_point = (x, y)

            elif event == cv2.EVENT_LBUTTONUP:  # 드래그 종료
                dragging = False
                resizing = False

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("동영상을 열 수 없습니다.")
        return

    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', on_mouse)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 점 표시
        for i in range(cnt):
            cv2.circle(frame, (int(src_pts[i, 0]), int(src_pts[i, 1])), 5, (0, 0, 255), -1)

        # 평면 보정 결과 표시
        if pers_mat is not None:
            warped_frame = cv2.warpPerspective(frame, pers_mat, (output_size, output_size))
            cv2.imshow('Board View', warped_frame)

            # 초기 버튼 상태 저장
            if initial_button_state is None:
                x, y, w, h = rect
                initial_button_state = frame[y:y + h, x:x + w]  # 초기 버튼 상태 저장

            # 사각형 표시
            draw_rectangle(frame, rect)

        cv2.imshow('Video', frame)
        key = cv2.waitKey(1)
        if key == 13 and pers_mat is not None:  # Enter 키로 좌표 저장
            button_rect = rect
            break
        elif key == 27:  # ESC 키로 종료
            break

    cap.release()
    cv2.destroyAllWindows()



def detect_turn_change(video_source, threshold=0.9):
    """
    버튼 영역의 변화를 감지하여 턴이 넘어갔는지 판단.

    Parameters:
    - video_source: 동영상 파일 경로 또는 웹캠 (0은 기본 웹캠).
    - threshold: SSIM 유사도 임계값.

    Returns:
    - bool: 변화 감지 여부.
    """
    global initial_button_state, button_rect, pers_mat

    if initial_button_state is None or button_rect is None:
        print("버튼 초기 상태가 설정되지 않았습니다.")
        return False

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("동영상을 열 수 없습니다.")
        return False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 현재 버튼 영역 추출
        x, y, w, h = button_rect
        button_region = frame[y:y + h, x:x + w]

        # SSIM 계산
        now_button_state = cv2.cvtColor(button_region, cv2.COLOR_BGR2GRAY)
        initial_gray = cv2.cvtColor(initial_button_state, cv2.COLOR_BGR2GRAY)
        similarity, _ = compare_ssim(initial_gray, now_button_state, full=True)
        if similarity < threshold:
            cap.release()
            return True

        # 버튼 영역 계속 출력
        cv2.imshow('Button View', button_region)

        # ESC 키로 종료
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return False


def turn_based_game(video_source=0, output_size=300, threshold=0.9):
    """
    동영상 입력에서 보드와 버튼 설정 후 턴 감지. 
    보드와 버튼의 실시간 상태를 계속 출력.

    Parameters:
    - video_source: 동영상 소스.
    - output_size: 보드 크기.
    - threshold: SSIM 유사도 임계값.

    Returns:
    - bool: 턴이 넘어갔는지 여부.
    """
    global initial_button_state, button_rect, pers_mat

    setup_board_and_button(video_source, output_size)
    return detect_turn_change(video_source, threshold)


result = turn_based_game(video_source=0, output_size=300, threshold=0.9)
if result:
    print("턴이 넘어갔습니다!")
else:
    print("턴 변화가 감지되지 않았습니다.")