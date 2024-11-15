import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim




is_move_Castling = [[0,0,0],[0,0,0]]
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
            if is_valid_move(piece_A, Acell, Bcell) == True:
                # A에 기물이 있고 B는 비어 있는 경우
                print(f"{piece_A}가 {Acell}에서 {Bcell}로 이동했습니다.")
                board_state[Acell[0]][Acell[1]] = None  # A 위치를 빈칸으로
                board_state[Bcell[0]][Bcell[1]] = piece_A  # B 위치로 이동
            else : 
                print(f"규칙오류 : {piece_A}가 {Acell}에서 {Bcell}로 이동할 수 없습니다.")
                turn_count-=1

        elif piece_A is None and piece_B is not None:
            if is_valid_move(piece_B, Bcell, Acell) == True:
                # B에 기물이 있고 A는 비어 있는 경우
                print(f"{piece_B}가 {Bcell}에서 {Acell}로 이동했습니다.")
                board_state[Bcell[0]][Bcell[1]] = None  # B 위치를 빈칸으로
                board_state[Acell[0]][Acell[1]] = piece_B  # A 위치로 이동
            else : 
                print(f"규칙오류 : {piece_B}가 {Bcell}에서 {Acell}로 이동할 수 없습니다.")
                turn_count-=1
        else:
            if turn_count % 2 != 0:
                if board_state[Acell[0]][Acell[1]].startswith() == 'W':
                    startCell = Acell
                    start_piece = piece_A
                    endCell = Bcell
                    end_piece = piece_B
                else:
                    startCell = Bcell
                    start_piece = piece_B
                    endCell = Acell
            else:
                if board_state[Acell[0]][Acell[1]].startswith() == 'W':
                    startCell = Bcell
                    start_piece = piece_B
                    endCell = Acell
                    end_piece = piece_A
                else:
                    startCell = Acell
                    start_piece = piece_A
                    endCell = Bcell
                    end_piece = piece_B

            if is_valid_move(piece_A, startCell, endCell) == True:
                # 두 위치 모두 기물이 있는 경우 (기물 잡기 상황)
                if turn_count % 2 != 0:  # 홀수 턴이면 백이 흑을 잡음
                    print(f"백의 {start_piece}가 {endCell}에서 흑의 {end_piece}를 잡았습니다.")
                else:  # 짝수 턴이면 흑이 백을 잡음
                    print(f"흑의 {start_piece}가 {endCell}에서 백의 {end_piece}를 잡았습니다.")
                
                board_state[endCell[0]][endCell[1]] = start_piece  # B 위치로 A의 기물 이동
                board_state[startCell[0]][startCell[1]] = None  # A 위치를 빈칸으로
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
        black_king_side = {(0, 4), (0, 2), (0, 0), (0, 3)}
        black_queen_side = {(0, 4), (0, 6), (0, 7), (0, 5)}

        move_set = set(moves)
        if move_set == white_king_side or move_set == black_king_side:
            print("킹 사이드 캐슬링이 감지되었습니다.")
            perform_castling("king", move_set)
            return
        elif move_set == white_queen_side or move_set == black_queen_side:
            print("퀸 사이드 캐슬링이 감지되었습니다.")
            perform_castling("queen", move_set)
            return

    else:
        print("이동을 감지하지 못했습니다. 또는 복수의 이동이 감지되었습니다.")

def perform_castling(castling_type, move_set):
    global board_state
    global turn_count

    if castling_type == "queen":
        if (7, 4) in move_set:  # 화이트 퀸 사이드
            if is_valid_move("WK", (7,4), (7,2)):
                board_state[7][4], board_state[7][2] = None, "WK"
                board_state[7][0], board_state[7][3] = None, "WR"
        elif (0, 4) in move_set:  # 블랙 퀸 사이드
            if is_valid_move("BK", (0,4), (0,2)):
                board_state[0][4], board_state[0][2] = None, "BK"
                board_state[0][0], board_state[0][3] = None, "BR"
    elif castling_type == "king":
        if (7, 4) in move_set:  # 화이트 킹 사이드
            if is_valid_move("WK", (7,4), (7,6)):
                board_state[7][4], board_state[7][6] = None, "WK"
                board_state[7][7], board_state[7][5] = None, "WR"
        elif (0, 4) in move_set:  # 블랙 킹 사이드
            if is_valid_move("BK", (0,4), (0,6)):
                board_state[0][4], board_state[0][6] = None, "BK"
                board_state[0][7], board_state[0][5] = None, "BR"

    turn_count += 1  # 캐슬링 후 턴 증가
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
    global turn_count

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

    elif En_passant_type == "angRW":  # 백 폰이 왼쪽 대각선으로 이동
        # 백 폰이 앙파상으로 이동하는 위치는 잡히는 폰의 위 칸
        move_row, move_col = target_row - 1, target_col
        if is_valid_move("WP", (target_row, target_col + 1), (move_row, move_col)):
            # 백 폰 이동
            board_state[target_row][target_col + 1] = None
            board_state[move_row][move_col] = "WP"
            # 흑 폰 제거
            board_state[target_row][target_col] = None

    elif En_passant_type == "angLB":  # 흑 폰이 오른쪽 대각선으로 이동
        # 흑 폰이 앙파상으로 이동하는 위치는 잡히는 폰의 아래 칸
        move_row, move_col = target_row + 1, target_col
        if is_valid_move("BP", (target_row, target_col - 1), (move_row, move_col)):
            # 흑 폰 이동
            board_state[target_row][target_col - 1] = None
            board_state[move_row][move_col] = "BP"
            # 백 폰 제거
            board_state[target_row][target_col] = None

    elif En_passant_type == "angRB":  # 흑 폰이 왼쪽 대각선으로 이동
        # 흑 폰이 앙파상으로 이동하는 위치는 잡히는 폰의 아래 칸
        move_row, move_col = target_row + 1, target_col
        if is_valid_move("BP", (target_row, target_col + 1), (move_row, move_col)):
            # 흑 폰 이동
            board_state[target_row][target_col + 1] = None
            board_state[move_row][move_col] = "BP"
            # 백 폰 제거
            board_state[target_row][target_col] = None

    else:
        print("Invalid En Passant type!")

    turn_count += 1  # 앙파상 후 턴 증가


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
                        (row_diff == -1 and abs(col_diff) == 1 and board_state[end_row][end_col] is not None)
                        (row_diff == -1 and abs(col_diff) == 1 and board_state[end_row+1][end_col] is not None)) #앙파상 판별

        # 룩
        elif piece == 'WR':
            if start[0]==0 and start[1]==0:
                is_move_Castling[0][0]+=1
            if start[0]==0 and start[1]==7:
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
                        (row_diff == 1 and abs(col_diff) == 1 and board_state[end_row][end_col] is not None)
                        (row_diff == 1 and abs(col_diff) == 1 and board_state[end_row-1][end_col] is not None)) #마지막줄 - 앙파상 판별
        # 룩
        elif piece == 'BR':
            if start[0]==7 and start[1]==0:
                is_move_Castling[1][0]+=1
            if start[0]==7 and start[1]==7:
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

# 이미지 경로 설정
for i in range(9):
    imageA = f'ChessRg\CLtest\{i+1}.PNG'  # 업로드한 체스판 이미지 경로 사용
    imageA = detect_and_crop_chessboard(imageA)
    imageB = f'ChessRg\CLtest\{i+2}.PNG'          # 두 번째 체스판 이미지 경로
    imageB = detect_and_crop_chessboard(imageB)
    detect_moves(imageA, imageB)
    print(board_state)
