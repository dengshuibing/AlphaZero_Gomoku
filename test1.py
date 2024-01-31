import cv2
import numpy as np

#识别棋盘上的 圆和叉

# 读取图像
img = cv2.imread('./res/temp.jpg', cv2.IMREAD_GRAYSCALE)

# 检测圆
# circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=20, minRadius=5, maxRadius=30)

# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     for i in circles[0, :]:
#         # 画圆
#         cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)

# 边缘检测
edges = cv2.Canny(img, 50, 150)

# 寻找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


# 在图像上绘制所有轮廓
# result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# cv2.drawContours(result, contours, -1, (0, 0, 255), 2)

count = 0

distance_threshold = 50
# 轮廓比较阈值
similarity_threshold = 0.1
contours_filtered = []

# 过滤相似的形状
# for contour in contours:
#     i, c = get_max_contours_in_similar(contours,contour,0.5)

#     if i != -1:
#         area1 = cv2.contourArea(contour)
#         area2 = cv2.contourArea(c)
#         area = max(area1,area2)

#         if area == area1:
#             contours_filtered.append(contours[i])
#         else:
#             contours_filtered.append(contour)


# 寻找交叉点
# for contour in contours_filtered:
#     area = cv2.contourArea(contour)
#     if area < 100.0:
#         continue
#     perimeter = cv2.arcLength(contour, False)
#     circularity = 4 * np.pi * (area / (perimeter * perimeter))

#     # 设定一个阈值，根据需要调整
#     circularity_threshold = 0.8

#     if circularity > circularity_threshold:
#         # 这是一个圆形轮廓
#         continue

#     epsilon = 0.02 * cv2.arcLength(contour, True)
#     approx = cv2.approxPolyDP(contour, epsilon, True)

#     if len(approx) == 4:
#         continue
    
#     # 画叉叉
#     cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)
#     print(f"area:{area},count:{count}")
#     count += 1


def get_max_contours_in_similar(contours,similarity_threshold):
    contours_filtered = []
    for i in range(len(contours)-1):
        if cv2.matchShapes(contours[i], contours[i+1], cv2.CONTOURS_MATCH_I1, 0) < similarity_threshold :   # 相似，取面积大的
            area1 = cv2.contourArea(contours[i])
            area2 = cv2.contourArea(contours[i+1])
            area = max(area1,area2)
            if area == area1:
                contours_filtered.append(contours[i])
            else:
                contours_filtered.append(contours[i+1])
    return contours_filtered

# 寻找交叉点
def find_cross(contours):
    cross_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100.0:
            continue

        # 圆形排除
        perimeter = cv2.arcLength(contour, False)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        # 设定一个阈值，根据需要调整
        circularity_threshold = 0.8

        if circularity > circularity_threshold:
            # 这是一个圆形轮廓
            continue
        
        # 矩形排除
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:
            continue
        
        # 叉叉
        cross_contours.append(contour)

    return cross_contours

# 寻找圆圈
def find_circle(contours):
    circle_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100.0:
            continue
        # 矩形排除
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            continue

        perimeter = cv2.arcLength(contour, False)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        # 设定一个阈值，根据需要调整
        circularity_threshold = 0.8

        if circularity > circularity_threshold:
            # 这是一个圆形轮廓
            circle_contours.append(contour)

    return circle_contours


# 打印 轮廓面积
def print_area(contours):
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        print(f'index:{i},area:{area}')


        
if __name__ == "__main__":
    cross_contours = find_cross(contours)
    circle_contours = find_circle(contours)

    # print_area(circle_contours)

    # filter_similar_contours = get_max_contours_in_similar(circle_contours,0.3)

    # print(len(circle_contours))
    # print_area(circle_contours)

    img_bgr = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

    cv2.drawContours(img_bgr, circle_contours, -1, (0, 0, 255), 1)
    cv2.drawContours(img_bgr, cross_contours, -1, (0, 0, 255), 1)

    # for contour in circle_contours:
    #     # 计算轮廓的矩
    #     M = cv2.moments(contour)

    #     # 计算中心点坐标
    #     if M["m00"] != 0:
    #         cX = int(M["m10"] / M["m00"])
    #         cY = int(M["m01"] / M["m00"])
    #     else:
    #         cX, cY = 0, 0

    #     print(f"center point:{cX,cY}")
    #     # 在图像上绘制中心点
    #     cv2.circle(img_bgr, (cX, cY), 5, (0, 255, 0), -1)

    # 显示图像
    cv2.imshow('Detected Circles and Crosses', img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
