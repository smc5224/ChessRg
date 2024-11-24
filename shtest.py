import cv2
import numpy as np
import copy
from skimage.metrics import structural_similarity as ssim

is_move_Castling = [[0,0,0],[0,0,0]]
turn_count = 1  # 턴을 관리할 변수
turn_count_button = 1
board_state = [
    ["BR", "BN", "BB", "BQ", "BK", "BB", "BN", "BR"],  # 백 기물의 첫 번째 줄
    ["BP", "BP", "BP", "BP", "BP", "BP", "BP", "BP"],  # 백 폰의 줄
    [None, None, None, None, None, None, None, None],  # 빈 칸
    [None, None, None, None, None, None, None, None],  # 빈 칸
    [None, None, None, None, None, None, None, None],  # 빈 칸
    [None, None, None, None, None, None, None, None],  # 빈 칸
    ["WP", "WP", "WP", "WP", "WP", "WP", "WP", "WP"],  # 흑 폰의 줄
    ["WR", "WN", "WB", "WQ", "WK", "WB", "WN", "WR"]   # 흑 기물의 첫 번째 줄
]
initial_button_image = None  # 초기 버튼 이미지 (비교 기준)
change_button_image = None
button_coordinates = None  # 버튼 영역 좌표 (x, y, width, height)
perspective_matrix = None  # 평면 보정을 위한 변환 행렬
progress_frame_past = None #프레임 저장할 전역변수(전)
progress_frame_now = None #프레임 저장할 전역변수(후)

mock_data = {
    'board status': [["BR", "BN", "BB", "BQ", "BK", "BB", "BN", "BR"],
                    ["BP", "BP", "BP", "BP", "BP", "BP", "BP", "BP"],
                    [None, None, None, None, None, None, None, None], 
                    [None, None, None, None, None, None, None, None],
                    [None, None, None, None, None, None, None, None],
                    [None, None, None, None, None, None, None, None],
                    ["WP", "WP", "WP", "WP", "WP", "WP", "WP", "WP"],
                    ["WR", "WN", "WB", "WQ", "WK", "WB", "WN", "WR"]],
    'notation': 'Initial Position',
}
                

data = [{
    'board status': [["BR", "BN", "BB", "BQ", "BK", "BB", "BN", "BR"],
                    ["BP", "BP", "BP", "BP", "BP", "BP", "BP", "BP"],
                    [None, None, None, None, None, None, None, None], 
                    [None, None, None, None, None, None, None, None],
                    [None, None, None, None, None, None, None, None],
                    [None, None, None, None, None, None, None, None],
                    ["WP", "WP", "WP", "WP", "WP", "WP", "WP", "WP"],
                    ["WR", "WN", "WB", "WQ", "WK", "WB", "WN", "WR"]],
    'notation': 'Initial Position',
}]

def setup_board_and_button(video):
    """
    체스판과 버튼 영역을 설정하는 함수.
    사용자가 마우스로 점 4개를 클릭해 평면 보정 후 버튼 영역을 지정함.
    """
    global initial_button_image, button_coordinates, perspective_matrix, progress_frame_past, progress_frame_now

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
                        [300, 300],
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

    ret, frame = video.read()
    if not ret:
        return

    original_frame = frame.copy()

    # 축소 비율 설정
    scale_factor = 0.5  # 화면 크기를 50%로 줄임
    frame_width = int(frame.shape[1] * scale_factor)
    frame_height = int(frame.shape[0] * scale_factor)
    resized_size = (frame_width, frame_height)

    def on_mouse(event, x, y, flags, param):
        """
        마우스 이벤트 처리 콜백 함수.
        """
        nonlocal is_dragging, is_resizing, start_drag_point, button_rect, selected_point_count, selected_points
        global perspective_matrix

        # 마우스 좌표를 원본 크기 기준으로 변환
        scaled_x = x / scale_factor
        scaled_y = y / scale_factor

        if perspective_matrix is None and selected_point_count < 4:  # 평면 보정을 위한 점 선택
            if event == cv2.EVENT_LBUTTONDOWN:
                # 변환된 좌표를 사용하여 점을 저장
                selected_points[selected_point_count] = [scaled_x, scaled_y]
                selected_point_count += 1
                if selected_point_count == 4:  # 점 4개가 선택되면 평면 보정 수행
                    destination_points = np.array([
                        [0, 0],
                        [300, 0],
                        [300, 300],
                        [0, 300]
                    ], dtype=np.float32)
                    perspective_matrix = cv2.getPerspectiveTransform(selected_points, destination_points)

        elif perspective_matrix is not None:  # 버튼 영역 조정 (평면 보정 후)
            if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 클릭
                if is_in_corner(int(scaled_x), int(scaled_y), button_rect):  # 모서리를 클릭한 경우
                    is_resizing = True
                    start_drag_point = (scaled_x, scaled_y)
                elif (button_rect[0] <= scaled_x <= button_rect[0] + button_rect[2] and
                    button_rect[1] <= scaled_y <= button_rect[1] + button_rect[3]):  # 버튼 내부 클릭
                    is_dragging = True
                    start_drag_point = (scaled_x, scaled_y)

            elif event == cv2.EVENT_MOUSEMOVE:  # 마우스 이동
                if is_dragging:  # 버튼 이동
                    dx, dy = scaled_x - start_drag_point[0], scaled_y - start_drag_point[1]
                    button_rect[0] += dx
                    button_rect[1] += dy
                    start_drag_point = (scaled_x, scaled_y)
                elif is_resizing:  # 버튼 크기 조정
                    dx, dy = scaled_x - start_drag_point[0], scaled_y - start_drag_point[1]
                    button_rect[2] += dx
                    button_rect[3] += dy
                    button_rect[2] = max(20, button_rect[2])  # 최소 너비 제한
                    button_rect[3] = max(20, button_rect[3])  # 최소 높이 제한
                    start_drag_point = (scaled_x, scaled_y)

            elif event == cv2.EVENT_LBUTTONUP:  # 마우스 버튼 해제
                is_dragging = False
                is_resizing = False

    # 마우스 콜백 설정
    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', on_mouse)

    while True:
        if not ret:
            break

        # 이전 프레임을 복원 (사각형 그리기 전)
        frame = cv2.resize(original_frame, resized_size)

        # 점 찍기 표시 (좌표를 축소 비율에 맞게 변환)
        for i in range(selected_point_count):
            scaled_x = int(selected_points[i, 0] * scale_factor)
            scaled_y = int(selected_points[i, 1] * scale_factor)
            cv2.circle(frame, (scaled_x, scaled_y), 5, (0, 0, 255), -1)

        if perspective_matrix is not None:  # 평면 보정 후
            warped_frame = cv2.warpPerspective(original_frame, perspective_matrix, (300, 300))
            progress_frame_past = warped_frame
            cv2.imshow('Board View', warped_frame)  # 보정된 체스판 출력
            draw_rectangle(frame, [int(coord * scale_factor) for coord in button_rect])

        # 원본 영상 출력
        cv2.imshow('Video', frame)

        key = cv2.waitKey(1)
        if key == 13 and perspective_matrix is not None:  # Enter 키로 버튼 영역 설정 완료
            x, y, w, h = button_rect
            initial_button_image = original_frame[int(y):int(y + h), int(x):int(x + w)]  # 버튼 초기 이미지 저장
            button_coordinates = button_rect
            break
        elif key == 27:  # ESC 키로 종료
            break

    cv2.destroyWindow("Video")


def detect_turn_change(video):

    """
    버튼 상태 변화(턴 변경)를 감지하는 함수 (프레임 스킵 적용).
    """
    global initial_button_image, button_coordinates, turn_count_button

    if initial_button_image is None or button_coordinates is None:
        print("버튼 초기 상태가 설정되지 않았습니다.")
        return False, None

    # 초기 상태 설정
    hidden_detected = False  # 숨김 상태 감지 플래그
    consecutive_hidden_count = 0
    consecutive_visible_count = 0
    threshold_count = 5  # 연속 변화 임계값 조정
    similarity_threshold = 0.5  # 유사도 임계값 조정
    frame_skip_interval = 1  # 프레임 스킵 간격
    frame_counter = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break

        # 프레임 스킵: frame_counter가 interval에 맞지 않을 경우 건너뜀
        frame_counter += 1
        if frame_counter % frame_skip_interval != 0:
            continue

        # 버튼 영역 추출
        x, y, w, h = button_coordinates
        x, y, w, h = int(x), int(y), int(w), int(h)
        current_button_region = frame[y:y + h, x:x + w]

        # 그레이스케일 변환 및 블러 적용
        current_gray = cv2.cvtColor(current_button_region, cv2.COLOR_BGR2GRAY)
        #current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)

        # 비교 대상 이미지 처리
        reference_gray = cv2.cvtColor(initial_button_image, cv2.COLOR_BGR2GRAY)
        #reference_gray = cv2.GaussianBlur(reference_gray, (5, 5), 0)

        # SSIM 계산
        similarity = ssim(reference_gray, current_gray)

        # 상태 판단
        if similarity < similarity_threshold:
            consecutive_hidden_count += 1
            consecutive_visible_count = 0
        else:
            consecutive_visible_count += 1
            consecutive_hidden_count = 0

        # 상태 전이 로직
        if not hidden_detected and consecutive_hidden_count >= threshold_count:
            # 숨김 상태로 전이
            hidden_detected = True
            consecutive_hidden_count = 0

        if hidden_detected and consecutive_visible_count >= threshold_count:
            # VISIBLE 상태로 복귀, 턴 변경 감지
            turn_count_button += 1
            hidden_detected = False
            consecutive_visible_count = 0
            initial_button_image = current_button_region.copy()
            return True, frame

        # 디버깅 정보 출력 (필요에 따라 주석 처리)
        #print(f"Frame: {frame_counter}, Similarity: {similarity:.2f}, Hidden Count: {consecutive_hidden_count}, Visible Count: {consecutive_visible_count}")

        # 버튼 영역 시각화
        cv2.imshow('Button View', current_button_region)
        if cv2.waitKey(1) == 27:  # ESC 키로 종료
            break

    return False, None



def process_frame(frame):
    """
    턴이 변경될 때마다 현재 체스판 상태를 처리하는 함수.
    """
    # 체스판을 분할하거나, 현재 상태를 분석하는 등의 작업을 수행
    global progress_frame_past, progress_frame_now, perspective_matrix
    if progress_frame_now is not None:
        progress_frame_past = progress_frame_now
        progress_frame_now = cv2.warpPerspective(frame, perspective_matrix, (300, 300))
    else :

        progress_frame_now = cv2.warpPerspective(frame, perspective_matrix, (300, 300))
    cv2.imshow('now',progress_frame_now)   
    


def split_chessboard(image):    # 체스판을 8x8로 자르고 리스트로 만들기 ( (i, j) i는 세로축, j는 가로축 / 왼쪽 위부터 0 )
    board_size = 8  # 체스판은 8x8
    height, width = image.shape[:2]
    cell_width = width // board_size
    cell_height = height // board_size
    cells = []

    for i in range(board_size):
        row = []
        for j in range(board_size):
            x_start = j * cell_width
            y_start = i * cell_height
            cell = image[y_start:y_start + cell_height, x_start:x_start + cell_width]
            row.append(cell)
        cells.append(row)

    return cells 

#simm썻을때 compare_cells
"""
def compare_cells(cell1, cell2):
    # Grayscale로 변환
    cell1_gray = cv2.cvtColor(cell1, cv2.COLOR_BGR2GRAY)
    cell2_gray = cv2.cvtColor(cell2, cv2.COLOR_BGR2GRAY)
    # 구조적 유사도 계산
    score, _ = ssim(cell1_gray, cell2_gray, full=True)
    return score
"""


def compare_cells_hsv(cell1, cell2):
    """
    두 셀의 HSV 색상 기반 히스토그램 비교
    """
    # HSV 변환
    cell1_hsv = cv2.cvtColor(cell1, cv2.COLOR_BGR2HSV)
    cell2_hsv = cv2.cvtColor(cell2, cv2.COLOR_BGR2HSV)

    # Hue와 Saturation만 비교
    hist1 = cv2.calcHist([cell1_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist2 = cv2.calcHist([cell2_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])

    # 정규화
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # 히스토그램 비교 (상관계수 방식)
    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return similarity


def detect_moves_hsv(initial_image, new_image):
    """
    체스판에서 HSV 기반 셀 비교로 말 이동 감지
    """
    # 체스판을 8x8 셀로 나누기
    initial_board = split_chessboard(initial_image)  # 8x8 배열
    new_board = split_chessboard(new_image)  # 8x8 배열

    moves = []  # 이동된 셀 저장

    for i in range(8):
        for j in range(8):
            # 두 셀 비교
            similarity = compare_cells_hsv(initial_board[i][j], new_board[i][j])
            if similarity < 0.85:  # 임계값 설정 (0.85 이하이면 변화 감지)
                moves.append((i, j))  # 이동된 셀 좌표 저장

    return moves



def detect_moves(initial_image, new_image):

    global board_state  # board_state를 전역 변수로 선언
    global turn_count   # turn_count를 전역 변수로 선언
    global mock_data, data

    initial_board = split_chessboard(initial_image)
    new_board = split_chessboard(new_image)
    moves = []
    minM = []
    for i in range(8):
        for j in range(8):
            similarity = compare_cells_hsv(initial_board[i][j], new_board[i][j])
            minM.append(similarity)
            # SSIM이 0.9 이하이면 말의 이동이 있다고 간주
            if similarity < 0.7:
                moves.append((i, j))  # 이동된 위치 저장
    #print(min(minM))

    # 이동이 발생한 칸을 분석하여 어느 칸에서 어느 칸으로 이동했는지 판단
    if len(moves) == 2:
        Acell, Bcell = moves
        piece_A = board_state[Acell[0]][Acell[1]]
        piece_B = board_state[Bcell[0]][Bcell[1]]

        if piece_A is not None and piece_B is None:
            if is_valid_move(piece_A, Acell, Bcell) == True:
                if ((piece_A=='WP' and Bcell[0] == 0) or (piece_A=='BP'and Bcell[0] == 7)): #프로모션 조건 확인
                    # A에 기물이 있고 B는 비어 있는 경우
                    perform_promotion(piece_A, Acell, Bcell)
                    
                else:       
                    # A에 기물이 있고 B는 비어 있는 경우
                    
                    print(f"{piece_A}가 {Acell}에서 {Bcell}로 이동했습니다.")
                    board_state[Acell[0]][Acell[1]] = None  # A 위치를 빈칸으로
                    board_state[Bcell[0]][Bcell[1]] = piece_A  # B 위치로 이동
                    mock_data['board status'] = copy.deepcopy(board_state)
                    mock_data['notation'] = notation_transformation(piece_A, Acell, Bcell)
                    data.append(copy.copy(mock_data))

            else : 
                print(f"규칙오류 : {piece_A}가 {Acell}에서 {Bcell}로 이동할 수 없습니다.")
                turn_count-=1

        elif piece_A is None and piece_B is not None:
            if is_valid_move(piece_B, Bcell, Acell) == True:
                if ((piece_B=='WP' and Acell[0] == 0) or (piece_B=='BP'and Acell[0] == 7)): #프로모션 조건 확인
                    # B에 기물이 있고 A는 비어 있는 경우
                    perform_promotion(piece_B, Bcell, Acell)
                else:
                # B에 기물이 있고 A는 비어 있는 경우
                    print(f"{piece_B}가 {Bcell}에서 {Acell}로 이동했습니다.")
                    board_state[Bcell[0]][Bcell[1]] = None  # B 위치를 빈칸으로
                    board_state[Acell[0]][Acell[1]] = piece_B  # A 위치로 이동
                    mock_data['board status'] = copy.deepcopy(board_state)
                    mock_data['notation'] = notation_transformation(piece_B, Bcell, Acell)
                    data.append(copy.copy(mock_data))
            else : 
                print(f"규칙오류 : {piece_B}가 {Bcell}에서 {Acell}로 이동할 수 없습니다.")
                turn_count-=1
        else:
            if turn_count % 2 != 0:
                if board_state[Acell[0]][Acell[1]].startswith('W'):
                    startCell = Acell
                    start_piece = piece_A
                    endCell = Bcell
                    end_piece = piece_B
                else:
                    startCell = Bcell
                    start_piece = piece_B
                    endCell = Acell
                    end_piece = piece_A
            else:
                if board_state[Acell[0]][Acell[1]].startswith('W'):
                    startCell = Bcell
                    start_piece = piece_B
                    endCell = Acell
                    end_piece = piece_A
                else:
                    startCell = Acell
                    start_piece = piece_A
                    endCell = Bcell
                    end_piece = piece_B

            if is_valid_move(start_piece, startCell, endCell) == True:
                # 두 위치 모두 기물이 있는 경우 (기물 잡기 상황)
                if ((start_piece=='WP' and endCell[0] == 0) or (start_piece=='BP'and endCell[0] == 7)): #프로모션 조건 확인
                    perform_promotion(start_piece, startCell, endCell) 

                elif end_piece.endswith('K'):       #게임 끝낫는지 확인
                    if turn_count % 2 != 0:  # 홀수 턴이면 백이 흑을 잡음
                        print(f"백의 {start_piece}가 {endCell}에서 흑의 {end_piece}를 잡으며 백이 승리합니다.")
                        board_state[endCell[0]][endCell[1]] = start_piece  # B 위치로 A의 기물 이동
                        board_state[startCell[0]][startCell[1]] = None  # A 위치를 빈칸으로
                        turn_count += 1
                        return 'W'
                    else:  # 짝수 턴이면 흑이 백을 잡음
                        print(f"흑의 {start_piece}가 {endCell}에서 백의 {end_piece}를 잡으며 흑이 승리합니다.")
                        board_state[endCell[0]][endCell[1]] = start_piece  # B 위치로 A의 기물 이동
                        board_state[startCell[0]][startCell[1]] = None  # A 위치를 빈칸으로
                        turn_count += 1
                        return 'B'
                    
                    

                else:  
                    board_state[endCell[0]][endCell[1]] = start_piece  # B 위치로 A의 기물 이동
                    board_state[startCell[0]][startCell[1]] = None  # A 위치를 빈칸으로  
                    if turn_count % 2 != 0:  # 홀수 턴이면 백이 흑을 잡음
                        print(f"백의 {start_piece}가 {endCell}에서 흑의 {end_piece}를 잡았습니다.")
                        mock_data['board status'] = copy.deepcopy(board_state)
                        mock_data['notation'] = notation_transformation(start_piece, startCell, endCell,is_move_kill=True, captured_piece=end_piece)
                        data.append(copy.copy(mock_data))
                    else:  # 짝수 턴이면 흑이 백을 잡음
                        print(f"흑의 {start_piece}가 {endCell}에서 백의 {end_piece}를 잡았습니다.")
                        mock_data['board status'] = copy.deepcopy(board_state)
                        mock_data['notation'] = notation_transformation(start_piece, startCell, endCell,is_move_kill=True, captured_piece=end_piece)
                        data.append(copy.copy(mock_data))
                    

            else :
                    if turn_count % 2 != 0:  # 홀수 턴이면 백이 흑을 잡음
                        print(f"규칙오류 : 백의 {piece_A}가 {Bcell}에서 흑의 {piece_B}를 잡을 수 없습니다.")
                        turn_count-=1
                    else:  # 짝수 턴이면 흑이 백을 잡음
                        print(f"규칙오류 : 흑의 {piece_A}가 {Bcell}에서 백의 {piece_B}를 잡을 수 없습니다.")
                        turn_count-=1
                        
        # 턴 수 증가
        turn_count += 1

    elif len(moves) == 4:

        white_king_side = {(7, 4), (7, 6), (7, 7), (7, 5)}
        white_queen_side = {(7, 4), (7, 2), (7, 0), (7, 3)}
        black_king_side = {(0, 4), (0, 6), (0, 7), (0, 5)}
        black_queen_side = {(0, 4), (0, 2), (0, 0), (0, 3)}

        move_set = set(moves)
        if move_set == white_king_side or move_set == black_king_side:
            print("킹 사이드 캐슬링이 감지되었습니다.")
            perform_castling("king", move_set)
        elif move_set == white_queen_side or move_set == black_queen_side:
            print("퀸 사이드 캐슬링이 감지되었습니다.")
            perform_castling("queen", move_set)
        
    elif len(moves) == 3:
        for i in range(3):
            for j in range(i + 1, 3):
                pos1, pos2 = moves[i], moves[j]
                diffX = pos1[0] - pos2[0]
                diffY = pos1[1] - pos2[1]
                # 대각선 관계 확인
                if abs(diffX) == 1 and abs(diffY) == 1:
                    # 대각선 관계가 있는 경우 나머지 좌표를 target으로 설정
                    target = moves[3 - i - j]  # 남은 하나의 인덱스
                    if target.endswith("P"):
                        if turn_count % 2 != 0:
                            if (diffX == -1 and diffY == 1):
                                En_type = 'angLW'
                            elif (diffX == -1 and diffY == -1):
                                En_type = 'angRW'
                        else:
                            if (diffX == -1 and diffY == 1):
                                En_type = 'angRB'
                            elif (diffX == -1 and diffY == -1):
                                En_type = 'angLB'
                    else:
                        print("규칙오류 : 앙파상은 폰끼리만 가능합니다")
                else:
                    print("인식오류: 앙파상이 아닌 3칸인식")
        perform_En_passant(En_type, target)

    else:
        print("이동을 감지하지 못했습니다. 또는 복수의 이동이 감지되었습니다.")
        print(len(moves))



def perform_castling(castling_type, move_set):
    global board_state
    global turn_count, is_move_Castling,mock_data,data
    print(board_state)
    print(move_set)
    if castling_type == "queen":
        if (7,2) in move_set :  # 화이트 퀸 사이드
            if is_valid_move("WK", (7,4), (7,2)):
                board_state[7][4], board_state[7][2] = None, "WK"
                board_state[7][0], board_state[7][3] = None, "WR"
                mock_data['board status'] = copy.deepcopy(board_state)
                mock_data['notation'] = notation_transformation('WK', (7,4), (7,2), is_move_Castling==True)
                data.append(copy.copy(mock_data))
                turn_count += 1  # 캐슬링 후 턴 증가
            else: 
                print("규칙오류 : 캐슬링이 불가능합니다.")
                turn_count-=1
        elif (0, 2) in move_set:  # 블랙 퀸 사이드
            if is_valid_move("BK", (0,4), (0,2)):
                board_state[0][4], board_state[0][2] = None, "BK"
                board_state[0][0], board_state[0][3] = None, "BR"
                mock_data['board status'] = copy.deepcopy(board_state)
                mock_data['notation'] = notation_transformation('BK', (0,4), (0,2), is_move_Castling==True)
                data.append(copy.copy(mock_data))
                turn_count += 1  # 캐슬링 후 턴 증가
            else: 
                print("규칙오류 : 캐슬링이 불가능합니다.")
                turn_count-=1
    elif castling_type == "king":
        if (7, 6) in move_set:  # 화이트 킹 사이드
            if is_valid_move("WK", (7,4), (7,6)):
                board_state[7][4], board_state[7][6] = None, "WK"
                board_state[7][7], board_state[7][5] = None, "WR"
                mock_data['board status'] = copy.deepcopy(board_state)
                mock_data['notation'] = notation_transformation('WK', (7,4), (7,6), is_move_Castling==True)
                data.append(copy.copy(mock_data))
                turn_count += 1  # 캐슬링 후 턴 증가
            else: 
                print("규칙오류 : 캐슬링이 불가능합니다.")
                turn_count-=1
        elif (0, 6) in move_set:  # 블랙 킹 사이드
            if is_valid_move("BK", (0,4), (0,6)):
                board_state[0][4], board_state[0][6] = None, "BK"
                board_state[0][7], board_state[0][5] = None, "BR"
                mock_data['board status'] = copy.deepcopy(board_state)
                mock_data['notation'] = notation_transformation('BK', (0,4), (0,6), is_move_Castling==True)
                data.append(copy.copy(mock_data))
                turn_count += 1  # 캐슬링 후 턴 증가
            else: 
                print("규칙오류 : 캐슬링이 불가능합니다.")
                turn_count-=1

    
    # 체스 기물 이동 규칙을 확인하는 함수


def perform_En_passant(En_passant_type, En_passant_target):
    """
    앙파상을 수행하는 함수.
    :param En_passant_type: 앙파상 타입 (angLW, angRW, angLB, angRB)
    - angLW: 백 폰이 오른쪽 대각선으로 이동하여 흑 폰을 잡음
    - angRW: 백 폰이 왼쪽 대각선으로 이동하여 흑 폰을 잡음
    - angLB: 흑 폰이 오른쪽 대각선으로 이동하여 백 폰을 잡음
    - angRB: 흑 폰이 왼쪽 대각선으로 이동하여 백 폰을 잡음
    :param En_passant_target: 잡히는 상대 기물의 좌표 (row, col)
    """
    global board_state
    global turn_count,mock_data,data

    target_row, target_col = En_passant_target

    if En_passant_type == "angLW":  # 백 폰이 오른쪽 대각선으로 이동
        # 백 폰이 앙파상으로 이동하는 위치는 잡히는 폰의 위 칸
        move_row, move_col = target_row - 1, target_col
        if is_valid_move("WP", (target_row, target_col - 1), (move_row, move_col)):
            # 백 폰 이동
            board_state[target_row][target_col - 1] = None
            board_state[move_row][move_col] = "WP"
            # 흑 폰 제거
            board_state[target_row][target_col] = None
            mock_data['board status'] = copy.deepcopy(board_state)
            start_cell = (target_row,target_col - 1)
            end_cell = (move_row,move_col)
            mock_data['notation'] = notation_transformation("WP", start_cell, end_cell,is_move_kill=True, is_move_en_passant=True, captured_piece='BP'  )
            data.append(copy.copy(mock_data))
            

    elif En_passant_type == "angRW":  # 백 폰이 왼쪽 대각선으로 이동
        # 백 폰이 앙파상으로 이동하는 위치는 잡히는 폰의 위 칸
        move_row, move_col = target_row - 1, target_col
        if is_valid_move("WP", (target_row, target_col + 1), (move_row, move_col)):
            # 백 폰 이동
            board_state[target_row][target_col + 1] = None
            board_state[move_row][move_col] = "WP"
            # 흑 폰 제거
            board_state[target_row][target_col] = None
            start_cell = (target_row,target_col + 1)
            end_cell = (move_row,move_col)
            mock_data['notation'] = notation_transformation("WP", start_cell, end_cell,is_move_kill=True, is_move_en_passant=True, captured_piece='BP')
            data.append(copy.copy(mock_data))

    elif En_passant_type == "angLB":  # 흑 폰이 오른쪽 대각선으로 이동
        # 흑 폰이 앙파상으로 이동하는 위치는 잡히는 폰의 아래 칸
        move_row, move_col = target_row + 1, target_col
        if is_valid_move("BP", (target_row, target_col - 1), (move_row, move_col)):
            # 흑 폰 이동
            board_state[target_row][target_col - 1] = None
            board_state[move_row][move_col] = "BP"
            # 백 폰 제거
            board_state[target_row][target_col] = None
            start_cell = (target_row,target_col - 1)
            end_cell = (move_row,move_col)
            mock_data['notation'] = notation_transformation("BP", start_cell, end_cell,is_move_kill=True, is_move_en_passant=True, captured_piece='WP')
            data.append(copy.copy(mock_data))
    elif En_passant_type == "angRB":  # 흑 폰이 왼쪽 대각선으로 이동
        # 흑 폰이 앙파상으로 이동하는 위치는 잡히는 폰의 아래 칸
        move_row, move_col = target_row + 1, target_col
        if is_valid_move("BP", (target_row, target_col + 1), (move_row, move_col)):
            # 흑 폰 이동
            board_state[target_row][target_col + 1] = None
            board_state[move_row][move_col] = "BP"
            # 백 폰 제거
            board_state[target_row][target_col] = None
            start_cell = (target_row,target_col + 1)
            end_cell = (move_row,move_col)
            mock_data['notation'] = notation_transformation("BP", start_cell, end_cell,is_move_kill=True, is_move_en_passant=True, captured_piece='WP')
            data.append(copy.copy(mock_data))
    else:
        print("Invalid En Passant type!")

    turn_count += 1  # 앙파상 후 턴 증가


def perform_promotion(piece, start, end):
    global board_state
    global turn_count

    # 프로모션 가능한 기물 확인 (퀸, 룩, 나이트, 비숍 중 선택 가능)
    promotion_options = ['Q', 'R', 'N', 'B']

    if piece == 'WP' :
        print("백 폰 프로모션: 퀸(Q), 룩(R), 나이트(N), 비숍(B) 중 선택하십시오.")
    elif piece == 'BP':
        print("흑 폰 프로모션: 퀸(Q), 룩(R), 나이트(N), 비숍(B) 중 선택하십시오.")
    else:
        print("프로모션 가능한 폰이 아닙니다.")
        return

    # 사용자 입력 받기 (기물을 선택)
    while True:
        new_piece = input("선택할 기물을 입력하세요 (Q, R, N, B): ").upper()
        if new_piece in promotion_options:
            break
        print("잘못된 선택입니다. 다시 입력하세요.")

    # 백 폰과 흑 폰에 따른 기물 설정
    if piece == 'WP':
        promoted_piece = 'W' + new_piece  # 백 기물
    elif piece == 'BP':
        promoted_piece = 'B' + new_piece  # 흑 기물

    # 프로모션 수행
    board_state[start[0]][start[1]] = None  # 기존 폰 제거
    board_state[end[0]][end[1]] = promoted_piece  # 새 기물 배치

    print(f"{piece}가 {start}에서 {end}로 이동한 후 {promoted_piece}(으)로 프로모션되었습니다.")


def is_valid_move(piece, start, end):
    """
    체스 기물의 이동 규칙을 검사합니다. 캐슬링 조건 포함.
    :param piece: 이동하는 기물 (예: 'WP', 'BP', 'WR', 등)
    :param start: 시작 위치 (행, 열)
    :param end: 도착 위치 (행, 열)
    :return: 이동이 규칙에 맞는지 여부 (True / False)
    """
    global is_move_Castling

    start_row, start_col = start
    end_row, end_col = end

    # 이동 차이 계산
    row_diff = end_row - start_row
    col_diff = end_col - start_col

    # 백 기물 (대문자로 시작)
    if piece.startswith('W'):
        # 폰
        if piece == 'WP':
            if start_row == 6:  # 초기 위치에서
                return ((row_diff == -2 and col_diff == 0 and board_state[end_row][end_col] is None) or
                        (row_diff == -1 and col_diff == 0 and board_state[end_row][end_col] is None) or
                        (row_diff == -1 and abs(col_diff) == 1 and board_state[end_row][end_col] is not None))
            else:  # 일반 이동
                return ((row_diff == -1 and col_diff == 0 and board_state[end_row][end_col] is None) or
                        (row_diff == -1 and abs(col_diff) == 1 and board_state[end_row][end_col] is not None) or
                        (row_diff == -1 and abs(col_diff) == 1 and board_state[end_row+1][end_col] is not None)) #앙파상 판별

        # 룩
        elif piece == 'WR':
            if start[0]==7 and start[1]==0:
                is_move_Castling[0][0]+=1
            if start[0]==7 and start[1]==7:
                is_move_Castling[0][2]+=1
            return (row_diff == 0 and col_diff != 0) or (row_diff != 0 and col_diff == 0)
        # 나이트
        elif piece == 'WN':
            return abs(row_diff) == 2 and abs(col_diff) == 1 or abs(row_diff) == 1 and abs(col_diff) == 2
        # 비숍
        elif piece == 'WB':
            return abs(row_diff) == abs(col_diff)
        # 퀸
        elif piece == 'WQ':
            return abs(row_diff) == abs(col_diff) or row_diff == 0 or col_diff == 0
        # 킹
        elif piece == 'WK':
            # 캐슬링 체크 추가
            if row_diff == 0 and abs(col_diff) == 2:
                # 킹사이드 캐슬링
                if col_diff == 2:
                    return (is_move_Castling[0][1] == 0 and  # 킹이 움직이지 않음
                            is_move_Castling[0][2] == 0 and  # 오른쪽 룩이 움직이지 않음
                            board_state[start_row][7] == 'WR' and  # 룩이 제자리에 있음
                            board_state[start_row][5] is None and
                            board_state[start_row][6] is None)
                # 퀸사이드 캐슬링
                elif col_diff == -2:
                    return (is_move_Castling[0][1] == 0 and  # 킹이 움직이지 않음
                            is_move_Castling[0][0] == 0 and  # 왼쪽 룩이 움직이지 않음
                            board_state[start_row][0] == 'WR' and  # 룩이 제자리에 있음
                            board_state[start_row][1] is None and
                            board_state[start_row][2] is None and
                            board_state[start_row][3] is None)
            # 일반 킹 이동
            is_move_Castling[0][1]+=1
            return abs(row_diff) <= 1 and abs(col_diff) <= 1

    # 흑 기물 (소문자로 시작)
    elif piece.startswith('B'):
        # 폰
        if piece == 'BP':
            if start_row == 1:  # 초기 위치에서
                return ((row_diff == 2 and col_diff == 0 and board_state[end_row][end_col] is None) or
                        (row_diff == 1 and col_diff == 0 and board_state[end_row][end_col] is None) or
                        (row_diff == 1 and abs(col_diff) == 1 and board_state[end_row][end_col] is not None))
            else:  # 일반 이동
                return ((row_diff == 1 and col_diff == 0 and board_state[end_row][end_col] is None) or
                        (row_diff == 1 and abs(col_diff) == 1 and board_state[end_row][end_col] is not None) or
                        (row_diff == 1 and abs(col_diff) == 1 and board_state[end_row-1][end_col] is not None)) #마지막줄 - 앙파상 판별
        # 룩
        elif piece == 'BR':
            if start[0]==0 and start[1]==0:
                is_move_Castling[1][0]+=1
            if start[0]==0 and start[1]==7:
                is_move_Castling[1][2]+=1
            return (row_diff == 0 and col_diff != 0) or (row_diff != 0 and col_diff == 0)
        # 나이트
        elif piece == 'BN':
            return abs(row_diff) == 2 and abs(col_diff) == 1 or abs(row_diff) == 1 and abs(col_diff) == 2
        # 비숍
        elif piece == 'BB':
            return abs(row_diff) == abs(col_diff)
        # 퀸
        elif piece == 'BQ':
            return abs(row_diff) == abs(col_diff) or row_diff == 0 or col_diff == 0
        # 킹
        elif piece == 'BK':
            # 캐슬링 체크 추가
            if row_diff == 0 and abs(col_diff) == 2:
                # 킹사이드 캐슬링
                if col_diff == 2:
                    return (is_move_Castling[1][1] == 0 and  # 킹이 움직이지 않음
                            is_move_Castling[1][2] == 0 and  # 오른쪽 룩이 움직이지 않음
                            board_state[start_row][7] == 'BR' and  # 룩이 제자리에 있음
                            board_state[start_row][5] is None and
                            board_state[start_row][6] is None)
                # 퀸사이드 캐슬링
                elif col_diff == -2:
                    return (is_move_Castling[1][1] == 0 and  # 킹이 움직이지 않음
                            is_move_Castling[1][0] == 0 and  # 왼쪽 룩이 움직이지 않음
                            board_state[start_row][0] == 'BR' and  # 룩이 제자리에 있음
                            board_state[start_row][1] is None and
                            board_state[start_row][2] is None and
                            board_state[start_row][3] is None)
            # 일반 킹 이동
            is_move_Castling[1][1]+=1 
            return abs(row_diff) <= 1 and abs(col_diff) <= 1

    # 정의되지 않은 기물인 경우
    return False


def video_play(video):

    global initial_button_image, button_coordinates, turn_count_button, progress_frame_past, progress_frame_now, data
    
    setup_board_and_button(video)
    a = 0
    while True:
        turnchange, frame = detect_turn_change(video)
        a+=1
        if turnchange:
            process_frame(frame)
            is_game_end = detect_moves(progress_frame_past,progress_frame_now)
            if is_game_end =='W':
                break
            elif is_game_end =='B':
                break
    return data


def notation_transformation(
    piece,
    startcell,
    endcell,
    is_move_kill=False,
    is_move_castling=False,
    is_move_en_passant=False,
    is_move_promotion=False,
    promotion_piece=None,
    captured_piece=None
):
    """
    체스 이동을 커스텀 기보법으로 변환하는 함수.

    Parameters:
    - color (str): 'W' 또는 'B' (백 또는 흑)
    - piece (str): 이동하는 체스 말의 약자 ('K', 'Q', 'R', 'B', 'N', 'P')
    - startcell (tuple): 시작 위치 (row, col) 0-7 인덱스 (0이 상단, 7이 하단)
    - endcell (tuple): 도착 위치 (row, col) 0-7 인덱스
    - is_move_kill (bool): 잡기 여부
    - is_move_castling (bool): 캐슬링 여부
    - is_move_en_passant (bool): 앙파상 여부
    - is_move_promotion (bool): 승격 여부
    - promotion_piece (str): 승격 시 선택한 말의 약자 (예: 'Q', 'R', 'B', 'N')
    - captured_piece (str): 잡힌 상대 말의 약자 (예: 'BN')

    Returns:
    - str: 변환된 기보 문자열
    """
    color = piece[:1]
    piece = piece[1:]
    # 체스판 좌표 변환 (0-7 인덱스를 'a1' - 'h8'으로 변환)
    def cell_to_notation(cell):
        row, col = cell
        return chr(ord('a') + col) + str(8 - row)

    start_notation = cell_to_notation(startcell)
    end_notation = cell_to_notation(endcell)

    if is_move_castling:
        if end_notation in ['g1', 'g8']:
            return f"{color} kingside castling"
        elif end_notation in ['c1', 'c8']:
            return f"{color} queenside castling"
    move_notation = ""

    if is_move_en_passant:
        # 앙파상: e.p. + 이동 정보 + kill + 상대 말
        move_notation += "e.p. "

    # 색상과 말의 약자
    move_notation += f"{color}{piece} {start_notation} {end_notation}"

    if is_move_kill:
        move_notation += f" kill {captured_piece}"

    if is_move_promotion and promotion_piece:
        move_notation += f" {piece} to {color}{promotion_piece}"

    return move_notation

cap = cv2.VideoCapture('video\\tes3.mp4')
video_play(cap)
cap.release()



