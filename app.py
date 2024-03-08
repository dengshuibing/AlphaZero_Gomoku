import os
from flask import Flask,request,jsonify
import numpy as np
from io import BytesIO
from game import Board, Game
import logging
import pickle
from mcts_alphaZero import MCTSPlayer
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
from human_play import Human
import json
from config import HOST
from mqtt import send_message_to_topic, DBManager
import cv2
from go import GoPhase

server = Flask(__name__,static_url_path='')

@server.route('/')
def hello():
    res = {
        'success':True,
        'data':"hello world!"
    }
    return jsonify(res)


# 棋盘大小
n = 5
board_shape = 8
width, height = 8, 8
padding = 10
board = Board(width=width, height=height, n_in_row=n)
game = Game(board)

param_theano = pickle.load(open('./model/best_policy_8_8_5.model', 'rb'),encoding='bytes')
best_policy = PolicyValueNet(width, height, param_theano=param_theano)
mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                            c_puct=5,
                            n_playout=400)  # set larger n_playout for better performance
# human player, input your move in the format: 2,3
human = Human()

deviceid = None

conn = DBManager()

# 机器 画叉，人类 画圈
@server.route('/start', methods=['POST'])
def start():
    global deviceid
    # deviceid = request.form.get('deviceid')
    deviceid = request.headers.get('deviceid')
    if not deviceid:
        return jsonify({
            "msg":'params error'
        })
    
    topic = conn.select_topic(deviceid)
    logging.info(f"topic: {topic}")
    
    start_player = request.form.get('start_player',2)
    data = game.start_play_api(human, mcts_player, deviceid, topic, int(start_player))
   
    res = {
        'success':True,
        'data':{
            "states" : map_to_json(data),
            "state_shapes" : board.state_shapes.tolist()
        }
    }
    return jsonify(res)

#接口下棋
@server.route('/humanPlay', methods=['POST'])
def humanPlay():
    # position = request.form.get('position',None)
    position = request.headers.get('position')

    if not position:
        return jsonify({
            "msg":'params error'
        })

    move = human.play(board,position)
    board.do_move(move)

    move = mcts_player.get_action(board)
    board.do_move(move)

    #发送 绘画指令
    global deviceid
    topic = conn.select_topic(deviceid)
    logging.info(f"topic: {topic}")
    arg = str(move+1)
    if move+1 <= 9:
        arg = '0'+str(move+1)
    push_json = {
        'type': 2,
        'deviceid': deviceid,
        'message': {
            'arg': "bwa"+arg,
            'url': HOST + '/dat/' + '8_8_board_bw/' + "bwa"+arg
        }
    }
    
    logging.info(push_json)
    code = send_message_to_topic(topic,push_json)
    logging.info(f"mqtt send sucess: {code}")

    res = {
        'success':True,
        'data':{
            "states" : map_to_json(board.states),
            "state_shapes" : board.state_shapes.tolist()
        }
    }
    return jsonify(res)

# 上传图片下棋
@server.route('/getHumanPlay', methods=['POST'])
def getHumanPlay():
    file = request.data
    binary_array = np.frombuffer(file, dtype=np.uint8)

    # 使用 OpenCV 解码二进制数据为图像
    im_bgr = cv2.imdecode(binary_array, cv2.IMREAD_COLOR)
    # cv2.imwrite('temp/temp_origin.jpg',im_bgr)
    # 圈叉
    # states = get_state_from_image(im_bgr)
    # 黑白
    # go = GoPhase('temp/temp_origin.jpg')
    # states = go.phase
    cv2.imwrite('temp/im_bgr.jpg',im_bgr)
    states = get_state_from_image_bw(im_bgr)

    # 将图片所见状态，转换为程序输入状态
    rotated_states_180 = np.rot90(states, k=2)

    # 获取人类的落子
    loc = get_last_play(board.state_shapes,rotated_states_180)
    print(f"loc:{loc}")

    # 人类下棋
    move = human.play(board,(board_shape - loc[0] - 1,loc[1]))
    board.do_move(move)

    # 机器下棋
    robot_move = mcts_player.get_action(board)
    board.do_move(robot_move)

    #发送 绘画指令
    global deviceid
    topic = conn.select_topic(deviceid)
    logging.info(f"topic: {topic}")
    arg = str(robot_move+1)
    if robot_move+1 <= 9:
        arg = '0'+str(robot_move+1)
    push_json = {
        'type': 2,
        'deviceid': deviceid,
        'message': {
            'arg': "bwa"+arg,
            'url': HOST + '/dat/' + '8_8_board_bw/' + "bwa"+arg
        }
    }

    logging.info(push_json)
    code = send_message_to_topic(topic,push_json)
    logging.info(f"mqtt send sucess: {code}")

    res = {
        'success':True,
        'data':{
            "states" : map_to_json(board.states),
            "state_shapes" : board.state_shapes.tolist()
        }
    }
    return jsonify(res)

@server.route('/getImageStates', methods=['POST'])
def getImageStates():
    file = request.data
    binary_array = np.frombuffer(file, dtype=np.uint8)

    # 使用 OpenCV 解码二进制数据为图像
    im_bgr = cv2.imdecode(binary_array, cv2.IMREAD_COLOR)  # 彩色图像
    # cv2.imwrite('temp/temp_origin.jpg',im_bgr)
    # 圈叉
    # states = get_state_from_image(im_bgr)
    # 黑白
    # go = GoPhase('temp/temp_origin.jpg')
    # states = go.phase
    cv2.imwrite('temp/im_bgr.jpg',im_bgr)
    states = get_state_from_image_bw(im_bgr)

    res = {
        'success':True,
        'data':{
            "imageStates" : states.tolist(), 
        }
    }
    return jsonify(res)


@server.route('/getBoardState')
def getBoardState():

    res = {
        'success':True,
        'data':{
            "states" : map_to_json(board.states),
            "state_shapes" : board.state_shapes.tolist()
        }
    }
    return jsonify(res)




@server.route('/mqtt/drawCircle', methods=['POST'])
def drawCircle():

    deviceid = request.headers.get('deviceid')

    if not deviceid:

        return jsonify({
            "msg":'params error'
        })


    logging.info('画小宇设备id: '+deviceid)

    push_json = {
        'type': 2,
        'deviceid': deviceid,
        'message': {
            'arg': '34_1',
            'url': HOST + '/dat/' + '34_1' 
        }
    }

    logging.info(push_json)

    topic = conn.select_topic(deviceid)
    print(topic)

    code = send_message_to_topic(topic,push_json)

    return jsonify({
        "code": code,
        "msg":"sucess"
    })

@server.route('/mqtt/testDat', methods=['POST'])
def testDat():

    deviceid = request.headers.get('deviceid')
    move = request.headers.get('move')

    if not deviceid and not move:

        return jsonify({
            "msg":'params error'
        })

    move = int(move)
    arg = str(move+1)
    logging.info('画小宇设备id: '+deviceid)

    if move+1 <= 9:
        arg = '0'+str(move+1)
    push_json = {
        'type': 2,
        'deviceid': deviceid,
        'message': {
            'arg': "bwa"+arg,
            'url': HOST + '/dat/' + '8_8_board_bw/' + "bwa"+arg
        }
    }

    logging.info(push_json)

    topic = conn.select_topic(deviceid)
    print(topic)

    code = send_message_to_topic(topic,push_json)

    return jsonify({
        "code": code,
        "msg":"sucess"
    })


@server.route('/end', methods=['GET'])
def end():
    #清除状态
    board.clean_state()

    return jsonify({
        "code": '0',
        "msg":"sucess"
    })


def map_to_json(intMap):
     # 手动转换不可序列化对象
    return {str(key):str(value) for key, value in intMap.items()}

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

# 寻找交叉点
def find_cross(shape,contours,states):
    h,w = shape
    cross_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10.0 or area > 1000.0:
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

        # 圆形排除
        loc_x, loc_y = get_location_by_point((x,y),(0,0),(w-1,h-1))
        if states[loc_y][loc_x] == 1:
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
#   1 2 3
#   4 5 6
#   7 8 9
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

# 判断一个点 是否 居于 格子中间
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


# 根据棋盘大小 初始化 棋盘
def init_state(board_shape):
    # 0 未下子 1 圆圈 2 叉叉
    return np.zeros((board_shape,board_shape),dtype=int)

# 画 十字点
def draw_point(img,p):
    cv2.line(img, (p[0]-10,p[1]), (p[0]+10,p[1]), (0,0,255), 1)
    cv2.line(img, (p[0],p[1]-10), (p[0],p[1]+10), (0,0,255), 1)

    return img

# 从 图片中获取状态
def get_state_from_image(im_bgr):
    im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY) # 转灰度图像
    im_gray = cv2.GaussianBlur(im_gray, (3,3), 0) # 灰度图像滤波降噪
    im_edge = cv2.Canny(im_gray, 30, 50) # 边缘检测获得边缘图像

    #找到棋盘顶点
    rect = _find_chessboard(im_edge)
    boxes = orderPoints(np.float32(rect))
    #得到只有棋盘的图片
    img_bgr = warpImage(im_bgr,boxes)
    cv2.imwrite("temp/temp_board.jpg",img_bgr)
    h,w = img_bgr.shape[:2]

    # 寻找所有轮廓
    img_bgr = cv2.imread('temp/temp_board.jpg', cv2.IMREAD_COLOR)  # 彩色图像
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) # 转灰度图像
    img_gray = cv2.GaussianBlur(img_gray, (3,3), 0) # 灰度图像滤波降噪
    img_edge = cv2.Canny(img_gray, 30, 50) # 边缘检测获得边缘图像
    contours, _ = cv2.findContours(img_edge, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    
    # 画出所有轮廓
    img_all = np.copy(img_bgr)
    cv2.drawContours(img_all, contours, -1, (0, 0, 255), 2)
    cv2.imwrite("temp/temp_all.jpg",img_all)

    # 初始化棋盘状态
    states = init_state(board_shape=board_shape)

    # 找到圆圆
    img_circle = np.copy(img_bgr)
    # circle_contours = find_circle(contours)
    # points = [get_center_point(contour) for contour in circle_contours ]
    # cv2.drawContours(img_circle, circle_contours, -1, (0, 0, 255), 2)#画圆
    points = get_circle_point(img_edge)
    for point in points:#画圆心
        img_circle = draw_point(img_circle,point)
    cv2.imwrite("temp/temp_circle.jpg",img_circle)
    states = get_state((h,w),states,points,1)
    
    # 找到叉叉
    img_cross = np.copy(img_bgr)
    cross_contours = find_cross((h,w),contours,states)
    points = [get_center_point(contour) for contour in cross_contours ]
    cv2.drawContours(img_cross, cross_contours, -1, (0, 0, 255), 2)#画叉
    for point in points:#画叉的中心
        img_cross = draw_point(img_cross,point)
    cv2.imwrite("temp/temp_cross.jpg",img_cross)
    states = get_state((h,w),states,points,2)

    return states

# 从 图片中获取状态 黑白棋子
def get_state_from_image_bw(im_bgr):
    im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY) # 转灰度图像
    im_gray = cv2.GaussianBlur(im_gray, (3,3), 0) # 灰度图像滤波降噪
    cv2.imwrite("temp/im_gray.jpg",im_gray)
    im_edge = cv2.Canny(im_gray, 30, 50) # 边缘检测获得边缘图像
    cv2.imwrite("temp/im_edge.jpg",im_edge)

    board_gray = None # 棋盘灰度图
    board_bgr = None # 棋盘彩色图
    board_edge = None # 棋盘边缘图
    rect = None # 棋盘四个角的坐标，顺序为lt/lb/rt/rb
    phase = None # 用以表示围棋局面的二维数组

    """找到棋盘"""
    rect = _find_chessboard(im_edge)
    boxes = orderPoints(np.float32(rect))
    #得到只有棋盘的图片
    board_gray = warpImage(im_gray,boxes)
    cv2.imwrite("temp/board_gray.jpg",board_gray)
    board_bgr = warpImage(im_bgr,boxes)
    cv2.imwrite("temp/board_bgr.jpg",board_bgr)
    board_edge = warpImage(im_edge,boxes)
    cv2.imwrite("temp/board_edge.jpg",board_edge)

    h,w = board_bgr.shape[:2]


    """识别棋子"""
    points = get_circle_point(board_edge)
    # for point in points:#画圆心
    #     board_bgr = draw_point(board_bgr,point)
    # cv2.imwrite("temp/board_bgr_cirlce.jpg",board_bgr)

    """获取棋盘状态"""
    states = init_state(board_shape=board_shape)
    board_bgr = cv2.imread('temp/board_bgr.jpg', cv2.IMREAD_COLOR)

    for point in points:
        row = point[1]
        col = point[0]
        
        bgr_ = board_bgr[row-5:row+5, col-5:col+5]
        cv2.imwrite("temp/board_bgr_cirlce_temp.jpg",bgr_)

        board_bgr = draw_point(board_bgr,point)
        cv2.imwrite("temp/board_bgr_cirlce.jpg",board_bgr)

        b = np.mean(bgr_[:,:,0])
        g = np.mean(bgr_[:,:,1])
        r = np.mean(bgr_[:,:,2])

        location = get_location_by_point((col, row),(0,0),(w-1,h-1))
        print(f"location(x:{location[0]},y:{location[1]})")

        total = b + g + r
        if total < 151:
            states[location[1]][location[0]] = 1 # 黑棋
        else:
            states[location[1]][location[0]] = 2 # 蓝棋

    return states


def get_last_play(pre_states,next_states):
    pre = pre_states.flatten()
    nex = next_states.flatten()

    for i in range(board_shape*board_shape):
        if pre[i] != nex[i]:
            return (i//board_shape,i%board_shape)
    
    return None

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

# 根据边缘检测图 输出圆心坐标
def get_circle_point(edge_img):
    points = []
    # 使用霍夫变换检测圆形
    circles = cv2.HoughCircles(edge_img, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=10, maxRadius=50)

    # 如果找到圆形，则绘制出来
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:

            points.append((i[0], i[1]))
    return points

# def write_text:

if __name__ == '__main__':
    server.run(host='0.0.0.0',port=5003)