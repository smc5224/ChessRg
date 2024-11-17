import cv2
import numpy as np

# 이미지 읽기
image_path = 'ChessRg\\test1.jpg'  # 체스보드 이미지 파일 경로
img = cv2.imread(image_path)

if img is None:
    print(f"이미지를 로드할 수 없습니다: {image_path}")
    exit()

# 이미지 크기 조정 (절반 크기로)
img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

# 그레이스케일 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# CLAHE 적용
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

# 대비 조정
gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=-30)

# 코너 감지
max_corners = 128
quality_level = 0.05
min_distance = 10
corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)

# 감지된 코너 시각화
if corners is not None:
    corners = np.int0(corners)  # 정수로 변환
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)  # 초록색 원으로 코너 표시
        print(f"Corner detected at: ({x}, {y})")
else:
    print("코너를 감지할 수 없습니다.")

# 결과 이미지 출력
cv2.imshow('Detected Corners', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
