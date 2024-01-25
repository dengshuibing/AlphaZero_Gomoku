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

server = Flask(__name__)

@server.route('/')
def hello():
    res = {
        'success':True,
        'data':"hello world!"
    }
    return jsonify(res)


n = 5
width, height = 8, 8
board = Board(width=width, height=height, n_in_row=n)
game = Game(board)

param_theano = pickle.load(open('best_policy_8_8_5.model', 'rb'),encoding='bytes')
best_policy = PolicyValueNet(width, height, param_theano=param_theano)
mcts_player = MCTSPlayer(best_policy.policy_value_fn,
                            c_puct=5,
                            n_playout=400)  # set larger n_playout for better performance
# human player, input your move in the format: 2,3
human = Human()

@server.route('/start')
def start():
    # set start_player=0 for human first
    data = game.start_play_api(human, mcts_player, start_player=1)
    json_data = {}
    # 手动转换不可序列化对象
    for key, value in data.items():
        json_data[str(key)] = str(value)

    res = {
        'success':True,
        'data':{
            "states" : json_data,
            "state_shapes" : board.state_shapes.tolist()
        }
    }
    return jsonify(res)

@server.route('/humanPlay', methods=['POST'])
def humanPlay():
    position = request.form.get('position',None)

    move = human.play(board,position)
    board.do_move(move)

    move = mcts_player.get_action(board)
    board.do_move(move)
    json_data = {}
    # 手动转换不可序列化对象
    for key, value in board.states.items():
        json_data[str(key)] = str(value)

    res = {
        'success':True,
        'data':{
            "states" : json_data,
            "state_shapes" : board.state_shapes.tolist()
        }
    }
    return jsonify(res)

@server.route('/getBoardState')
def getBoardState():
    
    json_data = {}
    # 手动转换不可序列化对象
    for key, value in board.states.items():
        json_data[str(key)] = str(value)

    res = {
        'success':True,
        'data':{
            "states" : json_data,
            "state_shapes" : board.state_shapes.tolist()
        }
    }
    return jsonify(res)


if __name__ == '__main__':
    server.run(host='0.0.0.0',port=5003)