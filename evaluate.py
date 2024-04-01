from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
import time
from collections import defaultdict, deque
from game import Board, Game
from policy_value_net_pytorch import PolicyValueNet  # Pytorch
import pickle

n_playout = 1200  # num of simulations for each move
c_puct = 5
# num of simulations used for the pure mcts, which is used as
# the opponent to evaluate the trained policy
pure_mcts_playout_num = 1000

board_width = 14
board_height = 14
n_in_row = 5
board = Board(width=board_width, height=board_height, n_in_row=n_in_row)
game = Game(board)

init_model = './temp/current_policy_50.model'
# param_theano = pickle.load(open(init_model, 'rb'),encoding='bytes')
# policy_value_net = PolicyValueNet(board_width,board_height,param_theano=param_theano)
policy_value_net = PolicyValueNet(board_width,board_height,model_file=init_model)

best_win_ratio = 0.0

def policy_evaluate(n_games=10):
    """
    Evaluate the trained policy by playing against the pure MCTS player
    Note: this is only for monitoring the progress of training
    """
    current_mcts_player = MCTSPlayer(policy_value_net.policy_value_fn,
                                        c_puct=c_puct,
                                        n_playout=n_playout)
    pure_mcts_player = MCTS_Pure(c_puct=5,
                                    n_playout=pure_mcts_playout_num)
    win_cnt = defaultdict(int)
    for i in range(n_games):
        time_start=time.time()
        winner = game.start_play(current_mcts_player,
                                        pure_mcts_player,
                                        start_player=i % 2,
                                        is_shown=0)
        time_end=time.time()
        print('time cost ',time_end-time_start,' s')

        win_cnt[winner] += 1
    win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
    print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            pure_mcts_playout_num,
            win_cnt[1], win_cnt[2], win_cnt[-1]))
    return win_ratio




if __name__ == '__main__':
    count = 0
    while True:
        win_ratio = policy_evaluate()
        if win_ratio > best_win_ratio:
            count = 0
            best_win_ratio = win_ratio
            if (best_win_ratio == 1.0 and
                    pure_mcts_playout_num < 20000):
                pure_mcts_playout_num += 1000
                best_win_ratio = 0.0
        else:
            # 连续评估三次都没有达到当前 num_playouts 的最高胜率，停止循环
            count += 1
        
        if count == 3:
            break
