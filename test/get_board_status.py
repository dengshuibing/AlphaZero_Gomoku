import cv2
import numpy as np


im_bgr = cv2.imread('./res/go_screenshot_25.01.2024.png') # 原始的彩色图像文件，BGR模式
im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY) # 转灰度图像
im_gray = cv2.GaussianBlur(im_gray, (3,3), 0) # 灰度图像滤波降噪
im_edge = cv2.Canny(im_gray, 30, 50) # 边缘检测获得边缘图像


# 寻找轮廓
contours, _ = cv2.findContours(im_edge, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)


rect1 = [
[268,  31],
[833,  22],
[844, 592],
[273, 599],
]

# 棋盘大小
board_shape = 5

# 根据棋盘大小 初始化 棋盘
def init_state(board_shape):
    # 0 未下子 1 圆圈 2 叉叉
    return np.zeros((board_shape,board_shape),dtype=int)

def _find_chessboard():
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

def draw_board(img, start_point, end_point):
    min_x, min_y = start_point
    max_x, max_y = end_point

    im = np.copy(img)
    x_series = np.linspace(min_x, max_x, board_shape+1, dtype=int)
    y_series = np.linspace(min_y, max_y, board_shape+1, dtype=int)
    for x,y in zip(x_series,y_series):
        im = cv2.line(im, (min_x, y), (max_x, y), (0,255,0), 1)
        im = cv2.line(im, (x, min_y), (x, max_y), (0,255,0), 1)
    
    return im

def draw_board_rect(img, rect):
    if rect is None:
        print('在图像文件中找不到棋盘！')
    else:
        print('棋盘坐标：')
        print('\t左上角：(%d,%d)'%(rect[0][0],rect[0][1]))
        print('\t左下角：(%d,%d)'%(rect[1][0],rect[1][1]))
        print('\t右上角：(%d,%d)'%(rect[2][0],rect[2][1]))
        print('\t右下角：(%d,%d)'%(rect[3][0],rect[3][1]))
    
    im = np.copy(img)
    for p in rect:
        im = cv2.line(im, (p[0]-10,p[1]), (p[0]+10,p[1]), (0,0,255), 1)
        im = cv2.line(im, (p[0],p[1]-10), (p[0],p[1]+10), (0,0,255), 1)

    return im

# 画 十字点
def draw_point(img,p):
    cv2.line(img, (p[0]-10,p[1]), (p[0]+10,p[1]), (0,0,255), 1)
    cv2.line(img, (p[0],p[1]-10), (p[0],p[1]+10), (0,0,255), 1)

    return img

# 计算长宽
def pointDistance(a, b):
    return int(np.sqrt(np.sum(np.square(a - b))))

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

# 获取棋盘状态 数组
def get_state(board_state,points,type):
    h,w = im_bgr.shape[:2]

    for point in points:
        x = point[0]
        y = point[1]
        location = get_location_by_point((x, y),(0,0),(w-1,h-1))
        # print(f"location(x:{location[0]},y:{location[1]})")
        board_state[location[1]][location[0]] = type
    
    return board_state

if __name__ == '__main__':

    rect = _find_chessboard()

    start_x, start_y = rect[0][0],rect[0][1]
    end_x, end_y = rect[3][0],rect[3][1]

    # for i in range(len(rect1)):
    #     im_bgr = draw_point(im_bgr,rect1[i])
    # im = draw_board_rect(im_bgr,rect)
    # im = draw_board(im_bgr,(start_x,start_y),(end_x,end_y))
    
    # boxes = orderPoints(np.float32(rect1))
    # im = warpImage(im_bgr,boxes)

    h,w = im_bgr.shape[:2]
    # im = draw_board(im_bgr,(0,0),(w-1,h-1))

    location = get_location_by_point((169, 395),(0,0),(w-1,h-1))
    # print(location)

    states = init_state(board_shape=5)

    circle_contours = find_circle(contours)
    points = [get_center_point(contour) for contour in circle_contours ]
    for point in points:
        im_bgr = draw_point(im_bgr,point)
    states = get_state(states,points,1)

    cross_contours = find_cross(contours)
    points = [get_center_point(contour) for contour in cross_contours ]
    for point in points:
        im_bgr = draw_point(im_bgr,point)

    states = get_state(states,points,2)

    print(states)



    cv2.imshow('go', im_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()