import cv2
import numpy as np

# 이미지 로드 및 그레이스케일 변환
image = cv2.imread('images1.jfif')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 50, 150)

# 4. Adaptive Threshold - 조명 변화가 있는 경우에도 균일한 이진화를 위해 적용
thresholded = cv2.adaptiveThreshold(edges, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)

# 코너 검출
corners = cv2.goodFeaturesToTrack(gray, maxCorners=64, qualityLevel=0.02, minDistance=15, blockSize=7, useHarrisDetector=True, k=0.04)

# 코너가 검출되었을 경우 시각화
if corners is not None:
    corners = np.int0(corners)  # 좌표를 정수로 변환
    for i in corners:
        x, y = i.ravel()
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)  # 각 코너에 빨간색 원 그리기

    cv2.imshow('Corners', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("코너를 찾을 수 없습니다.")
