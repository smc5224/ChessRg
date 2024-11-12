import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# 사진을 체스판으로 자르기
def detect_and_crop_chessboard(image_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)
    if image is None:
        print("이미지를 불러올 수 없습니다. 경로를 확인하세요.")
        return None

    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 블러를 적용해 노이즈 줄이기
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 에지 검출 (Canny Edge Detection)
    edges = cv2.Canny(blurred, 50, 150)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 가장 큰 사각형 윤곽선을 찾기 (체스판을 가정)
    max_contour = max(contours, key=cv2.contourArea)

    # 체스판 윤곽선을 사각형으로 근사화
    epsilon = 0.02 * cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, epsilon, True)

    if len(approx) == 4:  # 사각형인지 확인
        # 사각형의 네 점을 얻어서 정렬
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")

        # 점을 위쪽 좌우, 아래쪽 좌우 순으로 정렬
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        # 체스판을 정사각형으로 변환 (워핑)
        (tl, tr, br, bl) = rect
        maxWidth = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
        maxHeight = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        cropped_chessboard = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        # 체스판만 잘라낸 이미지 반환
        return cropped_chessboard
    else:
        print("체스판을 인식하지 못했습니다.")
        return None
# 체스판을 8x8로 자르고 리스트로 만들기 ( (i, j) i는 세로축, j는 가로축 / 왼쪽 위부터 0 )
def split_chessboard(image):
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
# 
def compare_cells(cell1, cell2):
    # Grayscale로 변환
    cell1_gray = cv2.cvtColor(cell1, cv2.COLOR_BGR2GRAY)
    cell2_gray = cv2.cvtColor(cell2, cv2.COLOR_BGR2GRAY)

    # 구조적 유사도 계산
    score, _ = ssim(cell1_gray, cell2_gray, full=True)
    return score

turn_count = 1  # 턴을 관리할 변수
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

def detect_moves(initial_image, new_image):

    global board_state  # board_state를 전역 변수로 선언
    global turn_count   # turn_count를 전역 변수로 선언

    initial_board = split_chessboard(initial_image)
    new_board = split_chessboard(new_image)
    moves = []

    for i in range(8):
        for j in range(8):
            similarity = compare_cells(initial_board[i][j], new_board[i][j])

            # SSIM이 0.9 이하이면 말의 이동이 있다고 간주
            if similarity < 0.9:
                moves.append((i, j))  # 이동된 위치 저장

    # 이동이 발생한 칸을 분석하여 어느 칸에서 어느 칸으로 이동했는지 판단
    if len(moves) == 2:
        Acell, Bcell = moves
        piece_A = board_state[Acell[0]][Acell[1]]
        piece_B = board_state[Bcell[0]][Bcell[1]]

        if piece_A is not None and piece_B is None:
            # A에 기물이 있고 B는 비어 있는 경우
            print(f"{piece_A}가 {Acell}에서 {Bcell}로 이동했습니다.")
            board_state[Acell[0]][Acell[1]] = None  # A 위치를 빈칸으로
            board_state[Bcell[0]][Bcell[1]] = piece_A  # B 위치로 이동

        elif piece_A is None and piece_B is not None:
            # B에 기물이 있고 A는 비어 있는 경우
            print(f"{piece_B}가 {Bcell}에서 {Acell}로 이동했습니다.")
            board_state[Bcell[0]][Bcell[1]] = None  # B 위치를 빈칸으로
            board_state[Acell[0]][Acell[1]] = piece_B  # A 위치로 이동

        else:
            # 두 위치 모두 기물이 있는 경우 (기물 잡기 상황)
            if turn_count % 2 != 0:  # 홀수 턴이면 백이 흑을 잡음
                print(f"백의 {piece_A}가 {Bcell}에서 흑의 {piece_B}를 잡았습니다.")
            else:  # 짝수 턴이면 흑이 백을 잡음
                print(f"흑의 {piece_A}가 {Bcell}에서 백의 {piece_B}를 잡았습니다.")
            
            board_state[Bcell[0]][Bcell[1]] = piece_A  # B 위치로 A의 기물 이동
            board_state[Acell[0]][Acell[1]] = None  # A 위치를 빈칸으로

        # 턴 수 증가
        turn_count += 1
    else:
        print("이동을 감지하지 못했습니다. 또는 복수의 이동이 감지되었습니다.")

# 이미지 경로 설정
image1 = 'ChessRg\c1.PNG'  # 업로드한 체스판 이미지 경로 사용
image1 = detect_and_crop_chessboard(image1)
image2 = 'ChessRg\c2.PNG'          # 두 번째 체스판 이미지 경로
image2 = detect_and_crop_chessboard(image2)
image3 = 'ChessRg\c3.PNG'  # 업로드한 체스판 이미지 경로 사용
image3 = detect_and_crop_chessboard(image3)
image4 = 'ChessRg\c4.PNG'          # 두 번째 체스판 이미지 경로
image4 = detect_and_crop_chessboard(image4)
image5 = 'ChessRg\c5.PNG'  # 업로드한 체스판 이미지 경로 사용
image5 = detect_and_crop_chessboard(image5)

# 이동 감지
detect_moves(image1, image2)
detect_moves(image2, image3)
detect_moves(image3, image4)
detect_moves(image4, image5)
