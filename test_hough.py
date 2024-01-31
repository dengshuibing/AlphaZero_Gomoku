import cv2
import numpy as np

# 读取图片
image = cv2.imread('res/temp_board.jpg', cv2.IMREAD_COLOR)

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 进行边缘检测
edges = cv2.Canny(gray, 50, 150)

# 使用霍夫变换检测圆形
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=30)

# 如果找到圆形，则绘制出来
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # 画出外圆
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # 画出圆心
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

# 显示结果
cv2.imshow('Detected Circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
