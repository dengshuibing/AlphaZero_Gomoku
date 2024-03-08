import cv2
import numpy as np


im_bgr = cv2.imread('./res/camer_board.jpg') # 原始的彩色图像文件，BGR模式
im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY) # 转灰度图像
im_gray = cv2.GaussianBlur(im_gray, (3,3), 0) # 灰度图像滤波降噪
im_edge = cv2.Canny(im_gray, 30, 50) # 边缘检测获得边缘图像



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
    x_series = np.linspace(min_x, max_x, 19, dtype=int)
    y_series = np.linspace(min_y, max_y, 19, dtype=int)
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

if __name__ == '__main__':

    rect = _find_chessboard()

    start_x, start_y = rect[0][0],rect[0][1]
    end_x, end_y = rect[3][0],rect[3][1]

    # im = draw_board_rect(im_edge,rect)
    im = draw_board(im_bgr,(start_x,start_y),(end_x,end_y))

    cv2.imshow('go', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()