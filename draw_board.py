import cv2 as cv
import numpy as np




# 根据二维数组绘制 绘制 棋盘当前状态


# 调整图片大小
def resize_image(image,new_size):
    # 获取原始分辨率
    original_resolution = image.shape[:2]

    # 计算调整比例
    ratio = min(new_size[0] / original_resolution[0], new_size[1] / original_resolution[1])

    # 计算调整后的尺寸
    # resized_size = (int(original_resolution[0] * ratio), int(original_resolution[1] * ratio))
    resized_size = (int(new_size[0]), int(new_size[1]))
    resized_image = cv.resize(image, resized_size)

    return resized_image

# 添加透明度通道信息
def add_alpha(image):
    # 创建具有透明度通道的空白PNG图像
    height, width = image.shape[:2]
    png_image = np.zeros((height, width, 4), dtype=np.uint8)  # 创建带有 Alpha 通道的空白图像
    # 将JPEG图像复制到PNG图像的RGB通道中
    png_image[:, :, :3] = image
    # 将 Alpha 通道设置为完全不透明（值为255）
    png_image[:, :, 3] = 255
    return png_image

# 把小尺寸上的黑色线条拷贝到大尺寸的指定位置
def combine_two_images(back,front,start_point,end_point):
    min_x,min_y = start_point
    max_x,max_y = end_point
    resize_head = resize_image(front,(max_x-min_x,max_y-min_y))

    for i in range(0,resize_head.shape[0]): #访问所有行
        for j in range(0,resize_head.shape[1]): #访问所有列
            if resize_head[i,j,0] < 150 and resize_head[i,j,1] < 150 and resize_head[i,j,2] < 150:
                # resize_head[i,j,3] = 0
                back[min_y+i, min_x+j] = resize_head[i,j]
    
    return back

if __name__ == '__main__':
    # 获取红色矩形坐标
    image = cv.imread("./res/png/monkey_body_rect1.png")
    min_x,min_y,max_x,max_y= 150,150,200,200

    # 合并
    board_path = "./res/board.png"
    board1_path = "./res/board1.png"
    black_path = "./res/black.png"
    white_path = "./res/white.png"
    result_path = "./res/result.png"

    black = cv.imread(black_path,cv.IMREAD_UNCHANGED)
    white = cv.imread(white_path,cv.IMREAD_UNCHANGED)
    board = cv.imread(board_path,cv.IMREAD_UNCHANGED)
    board1 = cv.imread(board1_path,cv.IMREAD_UNCHANGED)

    
    # board1 = add_alpha(board1)

    # 将JPEG图像保存为PNG格式
    # cv.imwrite('./res/board1.png', board1, [cv.IMWRITE_PNG_COMPRESSION, 0])

    all = combine_two_images(board1,white,(min_x,min_y),(max_x,max_y))

    cv.imwrite(result_path,all)



