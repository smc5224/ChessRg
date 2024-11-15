import cv2
import numpy as np

# 체스판 크기 설정 (7x7 내부 코너)
CHESSBOARD_SIZE = (7, 7)

# 이미지 읽기
image = cv2.imread('KakaoTalk_20241113_004647539.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 체스판 코너 검출
ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

if ret:
    # 코너 위치를 서브픽셀 수준으로 보정
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

    # 코너 좌표를 numpy 배열로 변환
    corners_refined = corners_refined.squeeze()

    # 한 칸의 길이 계산 (평균 가로 및 세로 거리)
    grid_widths = np.diff(corners_refined[:, 0].reshape(CHESSBOARD_SIZE[1], CHESSBOARD_SIZE[0]), axis=1).mean()
    grid_heights = np.diff(corners_refined[:, 1].reshape(CHESSBOARD_SIZE[1], CHESSBOARD_SIZE[0]), axis=0).mean()

    # 새로운 코너 생성 (테두리 확장)
    expanded_corners = []
    for row in range(-1, CHESSBOARD_SIZE[1] + 1):
        for col in range(-1, CHESSBOARD_SIZE[0] + 1):
            x = corners_refined[0][0] + col * grid_widths
            y = corners_refined[0][1] + row * grid_heights
            expanded_corners.append((x, y))

    expanded_corners = np.array(expanded_corners, dtype=np.float32).reshape((CHESSBOARD_SIZE[1] + 2, CHESSBOARD_SIZE[0] + 2, 2))

    # 격자선 그리기
    for row in range(expanded_corners.shape[0]):
        for col in range(expanded_corners.shape[1] - 1):
            pt1 = tuple(expanded_corners[row, col])
            pt2 = tuple(expanded_corners[row, col + 1])
            cv2.line(image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)

    for row in range(expanded_corners.shape[0] - 1):
        for col in range(expanded_corners.shape[1]):
            pt1 = tuple(expanded_corners[row, col])
            pt2 = tuple(expanded_corners[row + 1, col])
            cv2.line(image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (255, 0, 0), 2)

    # 코너를 표시
    for corner in expanded_corners.reshape(-1, 2):
        cv2.circle(image, (int(corner[0]), int(corner[1])), 5, (0, 0, 255), -1)

    # 이미지 크기를 절반으로 축소
    height, width = image.shape[:2]
    resized_image = cv2.resize(image, (width // 2, height // 2))

    # 결과 출력
    cv2.imshow('Expanded Chessboard with Grid Lines (Resized)', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("체스판 코너를 찾을 수 없습니다.")
