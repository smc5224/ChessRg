import cv2
import numpy as np

def detect_chessboard(image_path, board_size=(7, 7)):
    # 이미지 로드
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 체스판의 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)

    if ret:
        # 체스판의 코너가 성공적으로 인식되었을 때
        cv2.drawChessboardCorners(image, board_size, corners, ret)
        print("체스판 코너가 인식되었습니다.")
    else:
        print("체스판 코너를 찾을 수 없습니다.")
    
    # 결과 출력xdxxxxdxx
    cv2.imshow('Chessboard Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 사용 예시
detect_chessboard('chess.jfif')
