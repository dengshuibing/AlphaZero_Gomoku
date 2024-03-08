import cv2
import numpy as np


board_shape = 8
padding = 10

im_bgr = cv2.imread('temp/temp_board.jpg', cv2.IMREAD_COLOR)  # 彩色图像

def _find_chessboard(im_edge):
    contours, hierarchy = cv2.findContours(im_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 提取轮廓
    area = 0 # 找到的最大四边形及其面积
    for item in contours:
        hull = cv2.convexHull(item) # 寻找凸包
        epsilon = 0.1 * cv2.arcLength(hull, True) # 忽略弧长10%的点
        approx = cv2.approxPolyDP(hull, epsilon, True) # 将凸包拟合为多边形

        if len(approx) == 4 and cv2.isContourConvex(approx): # 如果是凸四边形
            ps = np.reshape(approx, (4,2)) # 四个角的坐标
            ps = ps[np.lexsort((ps[:,0],))] # 排序区分左右
            lt, lb = ps[:2][np.lexsort((ps[:2,1],))] # 排序区分上下
            rt, rb = ps[2:][np.lexsort((ps[2:,1],))] # 排序区分上下
            
            a = cv2.contourArea(approx)
            if a > area:
                area = a
                rect = (lt, lb, rt, rb)
    
    if not rect is None:
        return rect


# 四边形顶点排序，[top-left, top-right, bottom-right, bottom-left]
def orderPoints(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


# 透视变换
def warpImage(image, box):
    w, h = pointDistance(box[0], box[1]), \
           pointDistance(box[1], box[2])
    dst_rect = np.array([[0, 0],
                         [w - 1, 0],
                         [w - 1, h - 1],
                         [0, h - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(box, dst_rect)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped

# 计算长宽
def pointDistance(a, b):
    return int(np.sqrt(np.sum(np.square(a - b))))

# 根据棋盘大小 初始化 棋盘
def init_state(board_shape):
    # 0 未下子 1 圆圈 2 叉叉
    return np.zeros((board_shape,board_shape),dtype=int)

# 从轮廓中获取中心点
def get_center_point(contour):
    # 计算轮廓的矩
    M = cv2.moments(contour)

    # 计算中心点坐标
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0

    # print(f"center point:{cX,cY}")
    return (cX,cY)


# 获取棋盘状态 数组
def get_state(shape,board_state,points,type):
    h,w = shape

    for point in points:
        x = point[0]
        y = point[1]
        location = get_location_by_point((x, y),(0,0),(w-1,h-1))
        # print(f"location(x:{location[0]},y:{location[1]})")
        board_state[location[1]][location[0]] = type
    
    return board_state


# 输入一个点 获取 棋盘坐标位置
def get_location_by_point(center_point,start_point,end_point):
    location_x = -1
    location_y = -1

    center_x,center_y = center_point
    min_x, min_y = start_point
    max_x, max_y = end_point

    x_series = np.linspace(min_x, max_x, board_shape+1, dtype=int)
    y_series = np.linspace(min_y, max_y, board_shape+1, dtype=int)
    for i in range(board_shape):
        if x_series[i] < center_x and center_x < x_series[i+1]:
            location_x = i
    for i in range(board_shape):
        if y_series[i] < center_y and center_y < y_series[i+1]:
            location_y = i
    
    return (location_x,location_y)

# 寻找交叉点
def find_cross(shape,contours,states):
    print(len(contours))
    h,w = shape
    cross_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10.0 or area > 1000.0:
            continue

        # 判断 轮廓是否在格子中间
        x,y = get_center_point(contour)
        is_cross = is_cross_in_center((x,y),(0,0),(w-1,h-1))
        if not is_cross:
            continue

        # 圆形排除
        loc_x, loc_y = get_location_by_point((x,y),(0,0),(w-1,h-1))
        if states[loc_y][loc_x] == 1:
            continue
        
        # 叉叉
        cross_contours.append(contour)

    print(len(cross_contours))
    return cross_contours

def find_cross2(shape,contours):
    h,w = shape
    cross_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10.0 or area > 1000.0:
            continue

        # 圆形排除
        perimeter = cv2.arcLength(contour, False)
        if perimeter == 0:
            continue
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

        # 判断 轮廓是否在格子中间
        x,y = get_center_point(contour)
        is_cross = is_cross_in_center((x,y),(0,0),(w-1,h-1))
        if not is_cross:
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


# 判断一个点 是否 居于 格子中间,排除掉 圆所在的格子
def is_cross_in_center(center_point,start_point,end_point):
    is_cross_x = False
    is_cross_y = False

    center_x,center_y = center_point
    min_x, min_y = start_point
    max_x, max_y = end_point

    x_series = np.linspace(min_x, max_x, board_shape+1, dtype=int)
    y_series = np.linspace(min_y, max_y, board_shape+1, dtype=int)
    for i in range(board_shape):
        if x_series[i] + padding < center_x and center_x < x_series[i+1] - padding:
            is_cross_x = True
    for i in range(board_shape):
        if y_series[i] + padding < center_y and center_y < y_series[i+1] - padding:
            is_cross_y = True
    
    return is_cross_x and is_cross_y

# 画 十字点
def draw_point(img,p):
    cv2.line(img, (p[0]-10,p[1]), (p[0]+10,p[1]), (0,0,255), 1)
    cv2.line(img, (p[0],p[1]-10), (p[0],p[1]+10), (0,0,255), 1)

    return img

# 根据边缘检测图 输出圆心坐标
def get_circle_point(edge_img):
    points = []
    # 使用霍夫变换检测圆形
    #                                                               圆心距      canny阈值    投票数      最小半径       最大半径
    circles = cv2.HoughCircles(edge_img, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=50)

    # 如果找到圆形，则绘制出来
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:

            points.append((i[0], i[1]))
    return points

im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY) # 转灰度图像
im_gray = cv2.GaussianBlur(im_gray, (3,3), 0) # 灰度图像滤波降噪
im_edge = cv2.Canny(im_gray, 30, 50) # 边缘检测获得边缘图像


h,w = im_bgr.shape[:2]

# 寻找所有轮廓
contours, _ = cv2.findContours(im_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))

# 画出所有轮廓
img_all = np.copy(im_bgr)
cv2.drawContours(img_all, contours, -1, (0, 0, 255), 2)
cv2.imwrite("res/test_all.jpg",img_all)

# 初始化棋盘状态
states = init_state(board_shape=board_shape)

# 找到圆圆
img_circle = np.copy(im_bgr)
points = get_circle_point(im_edge)
print(points)
for point in points:#画圆心
    img_circle = draw_point(img_circle,point)
cv2.imwrite("res/test_circle.jpg",img_circle)
states = get_state((h,w),states,points,1)

# 找到叉叉
# img_cross = np.copy(im_bgr)
# cross_contours = find_cross((h,w),contours,states)
# points = [get_center_point(contour) for contour in cross_contours ]
# cv2.drawContours(img_cross, cross_contours, -1, (0, 0, 255), 2)#画叉
# for point in points:#画叉的中心
#     img_cross = draw_point(img_cross,point)
# cv2.imwrite("res/test_cross.jpg",img_cross)
# states = get_state((h,w),states,points,2)

