import cv2
import numpy as np


im_bgr = cv2.imread('./res/board_18_yellow.png') # 原始的彩色图像文件，BGR模式
im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY) # 转灰度图像
im_gray = cv2.GaussianBlur(im_gray, (3,3), 0) # 灰度图像滤波降噪
im_edge = cv2.Canny(im_gray, 30, 50) # 边缘检测获得边缘图像

# 读取图像
board = cv2.imread('./res/board_18_yellow.png', cv2.IMREAD_UNCHANGED)
black = cv2.imread('./res/black.png', cv2.IMREAD_UNCHANGED)
white = cv2.imread('./res/white.png', cv2.IMREAD_UNCHANGED)

# 棋子边界
padding = 5

# 棋盘大小
board_shape = 18

#棋盘状态
# state = [
#     [2,1,0,0,0,0,0,0],
#     [1,1,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,0,0],
#     [0,0,0,0,0,0,1,2],
#     [0,0,0,0,0,0,2,1]
# ]

# 19
# state = [
#     [0,0,2,1,1,0,1,1,1,2,0,2,0,2,1,0,1,0,0],
#     [0,0,2,1,0,1,1,1,2,0,2,0,2,2,1,1,1,0,0],
#     [0,0,2,1,1,0,0,1,2,2,0,2,0,2,1,0,1,0,0],
#     [0,2,1,0,1,1,0,1,2,0,2,2,2,0,2,1,0,1,0],
#     [0,2,1,1,0,1,1,2,2,2,2,0,0,2,2,1,0,1,0],
#     [0,0,2,1,1,1,1,2,0,2,0,2,0,0,2,1,0,0,0],
#     [0,0,2,2,2,2,1,2,2,0,0,0,0,0,2,1,0,0,0],
#     [2,2,2,0,0,0,2,1,1,2,0,2,0,0,2,1,0,0,0],
#     [1,1,2,0,0,0,2,2,1,2,0,0,0,0,2,1,0,0,0],
#     [1,0,1,2,0,2,1,1,1,1,2,2,2,0,2,1,1,1,1],
#     [0,1,1,2,0,2,1,0,0,0,1,2,0,2,2,1,0,0,1],
#     [1,1,2,2,2,2,2,1,0,0,1,2,2,0,2,1,0,0,0],
#     [2,2,0,2,2,0,2,1,0,0,1,2,0,2,2,2,1,0,0],
#     [0,2,0,0,0,0,2,1,0,1,1,2,2,0,2,1,0,0,0],
#     [0,2,0,0,0,2,1,0,0,1,0,1,1,2,2,1,0,0,0],
#     [0,0,2,0,2,2,1,1,1,1,0,1,0,1,1,0,0,0,0],
#     [0,2,2,0,2,1,0,0,0,0,1,0,0,0,0,1,1,0,0],
#     [0,0,2,0,2,1,0,1,1,0,0,1,0,1,0,1,0,0,0],
#     [0,0,0,2,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0]
# ]

# 18
state = [
    [0,0,2,1,1,0,1,1,1,2,0,2,0,2,1,0,1,0],
    [0,0,2,1,0,1,1,1,2,0,2,0,2,2,1,1,1,0],
    [0,0,2,1,1,0,0,1,2,2,0,2,0,2,1,0,1,0],
    [0,2,1,0,1,1,0,1,2,0,2,2,2,0,2,1,0,1],
    [0,2,1,1,0,1,1,2,2,2,2,0,0,2,2,1,0,1],
    [0,0,2,1,1,1,1,2,0,2,0,2,0,0,2,1,0,0],
    [0,0,2,2,2,2,1,2,2,0,0,0,0,0,2,1,0,0],
    [2,2,2,0,0,0,2,1,1,2,0,2,0,0,2,1,0,0],
    [1,1,2,0,0,0,2,2,1,2,0,0,0,0,2,1,0,0],
    [1,0,1,2,0,2,1,1,1,1,2,2,2,0,2,1,1,1],
    [0,1,1,2,0,2,1,0,0,0,1,2,0,2,2,1,0,0],
    [1,1,2,2,2,2,2,1,0,0,1,2,2,0,2,1,0,0],
    [2,2,0,2,2,0,2,1,0,0,1,2,0,2,2,2,1,0],
    [0,2,0,0,0,0,2,1,0,1,1,2,2,0,2,1,0,0],
    [0,2,0,0,0,2,1,0,0,1,0,1,1,2,2,1,0,0],
    [0,0,2,0,2,2,1,1,1,1,0,1,0,1,1,0,0,0],
    [0,2,2,0,2,1,0,0,0,0,1,0,0,0,0,1,1,0],
    [0,0,2,0,2,1,0,1,1,0,0,1,0,1,0,1,0,0]
]


def add_alpha_channel(img):
    """ 为jpg图像添加alpha通道 """
 
    b_channel, g_channel, r_channel = cv2.split(img) # 剥离jpg图像通道
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255 # 创建Alpha通道
 
    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel)) # 融合通道
    return img_new
 
def merge_img(jpg_img, png_img, y1, y2, x1, x2):
    """ 将png透明图像与jpg图像叠加 
        y1,y2,x1,x2为叠加位置坐标值
    """
    
    # 判断jpg图像是否已经为4通道
    if jpg_img.shape[2] == 3:
        jpg_img = add_alpha_channel(jpg_img)
    
    '''
    当叠加图像时，可能因为叠加位置设置不当，导致png图像的边界超过背景jpg图像，而程序报错
    这里设定一系列叠加位置的限制，可以满足png图像超出jpg图像范围时，依然可以正常叠加
    '''
    yy1 = 0
    yy2 = png_img.shape[0]
    xx1 = 0
    xx2 = png_img.shape[1]
 
    if x1 < 0:
        xx1 = -x1
        x1 = 0
    if y1 < 0:
        yy1 = - y1
        y1 = 0
    if x2 > jpg_img.shape[1]:
        xx2 = png_img.shape[1] - (x2 - jpg_img.shape[1])
        x2 = jpg_img.shape[1]
    if y2 > jpg_img.shape[0]:
        yy2 = png_img.shape[0] - (y2 - jpg_img.shape[0])
        y2 = jpg_img.shape[0]
 
    # 获取要覆盖图像的alpha值，将像素值除以255，使值保持在0-1之间
    alpha_png = png_img[yy1:yy2,xx1:xx2,3] / 255.0
    alpha_jpg = 1 - alpha_png
    
    # 开始叠加
    for c in range(0,3):
        jpg_img[y1:y2, x1:x2, c] = ((alpha_jpg*jpg_img[y1:y2,x1:x2,c]) + (alpha_png*png_img[yy1:yy2,xx1:xx2,c]))
 
    return jpg_img

# 调整图片大小
def resize_image(image,new_size):
    # 获取原始分辨率
    original_resolution = image.shape[:2]

    # 计算调整比例
    ratio = min(new_size[0] / original_resolution[0], new_size[1] / original_resolution[1])

    # 计算调整后的尺寸
    # resized_size = (int(original_resolution[0] * ratio), int(original_resolution[1] * ratio))
    resized_size = (int(new_size[0]), int(new_size[1]))
    resized_image = cv2.resize(image, resized_size)

    return resized_image

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

# 画棋盘线条
def draw_board(img):
    rect = _find_chessboard()

    min_x, min_y = rect[0][0],rect[0][1]
    max_x, max_y = rect[3][0],rect[3][1]

    im = np.copy(img)
    x_series = np.linspace(min_x, max_x, board_shape+1, dtype=int)
    y_series = np.linspace(min_y, max_y, board_shape+1, dtype=int)
    for x,y in zip(x_series,y_series):
        im = cv2.line(im, (min_x, y), (max_x, y), (0,255,0), 1)
        im = cv2.line(im, (x, min_y), (x, max_y), (0,255,0), 1)
    
    return im

#根据状态数组画棋盘
def draw_state():
    # 找到棋盘边界
    rect = _find_chessboard()
    start_x, start_y = rect[0][0],rect[0][1]
    end_x, end_y = rect[3][0],rect[3][1]

    # 单元格 宽高
    unit_w = (end_x-start_x)//board_shape
    unit_h = (end_y-start_y)//board_shape

    # 棋子 宽高
    w = unit_w-padding*2
    h = unit_h-padding*2

    # 黑白棋子 缩小到单元格里面
    black_unit = resize_image(black,(w,h))
    white_unit = resize_image(white,(w,h))

    res_img = None
    
    for i in range(board_shape):
        for j in range(board_shape):
            # 设置叠加位置坐标
            x1 = unit_w*j+unit_w//2-w//2
            y1 = unit_h*i+unit_h//2-h//2

            x2 = x1 + w
            y2 = y1 + h

            # 0 未下子 1 黑子 2 白子
            player = state[i][j]

            if player == 1:
                # 叠加
                res_img = merge_img(board, black_unit, start_y+y1, start_y+y2, start_x+x1, start_x+x2)
            if player == 2:
                # 叠加
                res_img = merge_img(board, white_unit, start_y+y1, start_y+y2, start_x+x1, start_x+x2)
    
    return res_img



def draw_state2():
    # 找到棋盘边界
    rect = _find_chessboard()
    start_x, start_y = rect[0][0],rect[0][1]
    end_x, end_y = rect[3][0],rect[3][1]

    x_series = np.linspace(start_x, end_x, board_shape+1, dtype=int)
    y_series = np.linspace(start_y, end_y, board_shape+1, dtype=int)


    res_img = None
    
    for i in range(board_shape):
        for j in range(board_shape):
            min_x, max_x = x_series[i], x_series[i+1]
            min_y, max_y = y_series[j], y_series[j+1]

            # 画出点
            # res_img = draw_point(board,(min_x,min_y))
            # res_img = draw_point(board,(max_x,max_y))

            # 0 未下子 1 黑子 2 白子
            player = state[i][j]

            if player == 1:
                # 画黑棋子
                res_img = draw_circle(board,((max_x+min_x)//2,(max_y+min_y)//2),min((max_x-min_x),(max_y-min_y))//2-padding)
            if player == 2:
                # 画白棋子
                res_img = draw_circle(board,((max_x+min_x)//2,(max_y+min_y)//2),min((max_x-min_x),(max_y-min_y))//2-padding,(220,220,220))
    
    return res_img


#画棋盘边界
def draw_board_rect(img):
    rect = _find_chessboard()

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

# 画 圆
def draw_circle(img,center_p,r,color=(0, 0, 0)):
    # 画圆
    cv2.circle(img, center_p, r, color, -1)
    return img


if __name__ == '__main__':
   
    # im = draw_board_rect(im_bgr)

    im = draw_state2()
    # im = draw_point(im_bgr,(38,40))

    # im = draw_board(im_bgr)

    # 显示结果图像
    cv2.imshow('result', im)
 
    # 保存结果图像，读者可自行修改文件路径
    # cv2.imwrite('./res/res.jpg', im)
 
    # 定义程序退出方式：鼠标点击显示图像的窗口后，按ESC键即可退出程序
    if cv2.waitKey(0) & 0xFF == 27:
        cv2.destroyAllWindows() 