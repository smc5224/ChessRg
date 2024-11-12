import cv2
import numpy as np

def split_chessboard(image):
    """
    체스판 이미지를 받아 각 칸을 나누고, 각 칸의 이미지를 리스트로 반환합니다.
    """
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

def compare_boards(initial_board, new_board):
    """
    초기 체스판과 새로운 체스판을 비교하여 이동이 발생한 칸을 찾고 기록합니다.
    """
    board_size = 8
    moves = []

    for i in range(board_size):
        for j in range(board_size):
            # 각 칸을 비교하여 차이점이 있으면 이동으로 간주
            initial_cell = initial_board[i][j]
            new_cell = new_board[i][j]

            # 이미지 차이 계산 (예: 구조적 유사도 또는 절대 차이 이용)
            diff = cv2.absdiff(initial_cell, new_cell)
            _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            non_zero_count = cv2.countNonZero(cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY))

            if non_zero_count > 0:  # 차이가 있는 경우
                moves.append(((i, j), non_zero_count))

    return moves



# 이미지 경로 설정
image = 'ChessRg\chessbord.jpg'
process_chessboard_image(image_path)
