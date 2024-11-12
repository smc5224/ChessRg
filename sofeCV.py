import cv2
import numpy as np

# 이미지 로드 및 그레이스케일 변환
image = cv2.imread('KakaoTalk_20241113_004647539.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 체스보드 크기 (8x8 체스판 내부 코너의 수는 7x7)
pattern_size = (8, 8)

# 체스판 코너 감지
ret, corners = cv2.findChessboardCorners(gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)

if ret:
    # 코너 점들을 더 정밀하게 조정
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), 
                               criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    # 8x8 격자의 바깥쪽 네 점 선택
    top_left = tuple(corners[0][0])  # 좌상단
    top_right = tuple(corners[6][0])  # 우상단
    bottom_left = tuple(corners[-7][0])  # 좌하단
    bottom_right = tuple(corners[-1][0])  # 우하단

    # 네 꼭짓점을 파란색 원으로 표시
    for point in [top_left, top_right, bottom_left, bottom_right]:
        cv2.circle(image, (int(point[0]), int(point[1])), 10, (255, 0, 0), -1)  # 파란색으로 표시

    # 8x8 격자 그리기
    for i in range(7):
        for j in range(7):
            if j < 6:
                pt1 = tuple(corners[i * 7 + j][0])
                pt2 = tuple(corners[i * 7 + (j + 1)][0])
                cv2.line(image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)
            if i < 6:
                pt1 = tuple(corners[i * 7 + j][0])
                pt2 = tuple(corners[(i + 1) * 7 + j][0])
                cv2.line(image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), (0, 255, 0), 2)

    # 결과 이미지 출력
    scale_percent = 50  # 50%로 축소
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    # 크기 조절
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('Chessboard with Outer Points', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("체스보드 코너를 찾을 수 없습니다.")
