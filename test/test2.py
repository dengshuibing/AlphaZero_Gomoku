import cv2
import numpy as np

# 生成圆圈

img = cv2.imread('./res/go_screenshot_25.01.2024.png', cv2.IMREAD_UNCHANGED)

board_shape = 5

location = (3,4)


# 棋子边界
padding = 20


def draw_board(img, start_point, end_point):
    min_x, min_y = start_point
    max_x, max_y = end_point

    im = np.copy(img)
    x_series = np.linspace(min_x, max_x, board_shape+1, dtype=int)
    y_series = np.linspace(min_y, max_y, board_shape+1, dtype=int)
    for x,y in zip(x_series,y_series):
        im = cv2.line(im, (min_x, y), (max_x, y), (0,0,255), 1)
        im = cv2.line(im, (x, min_y), (x, max_y), (0,0,255), 1)
    
    return im

def draw_circle(img,start_point, end_point,location):
    min_x, min_y = start_point
    max_x, max_y = end_point

    x_series = np.linspace(min_x, max_x, board_shape+1, dtype=int)
    y_series = np.linspace(min_y, max_y, board_shape+1, dtype=int)

    x1 = x_series[location[0]-1]
    x2 = x_series[location[0]]

    y1 = y_series[location[1]-1]
    y2 = y_series[location[1]]

    center_x = (x1+x2) // 2
    w = x2 - x1

    center_y = (y1+y2) // 2
    h = y2 - y1

    cv2.circle(img, (center_x,center_y), min(w//2-padding,h//2-padding), (0, 0, 0) , 5)

    return img

if __name__ == '__main__':

    height, width = 720,720
    blank_image = 255 * np.ones((height, width, 3), np.uint8)  # 创建一个三通道（BGR）的空白图像

    board_image = draw_board(blank_image,(0,0),(width-1,height-1))
    image = draw_circle(board_image,(0,0),(width-1,height-1),location)

    


    # 显示图片
    cv2.imshow('Blank Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
