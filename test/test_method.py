import cv2
import numpy as np

# 读取棋盘格图像
image = cv2.imread("./temp/im_bgr.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 棋盘格尺寸
patternSize = (14, 14)

# 查找棋盘格角点
retval, corners = cv2.findChessboardCorners(gray, patternSize)

if retval:
    # 在图像上标记角点
    cv2.drawChessboardCorners(image, patternSize, corners, retval)
    cv2.imshow('Chessboard with Corners', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Unable to find chessboard corners.")
