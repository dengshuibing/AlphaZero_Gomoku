from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_tensorflow import PolicyValueNet # Tensorflow
from game import Board, Game
import variables

from collections import defaultdict, deque
from multiprocessing import Pool
import time
import os


def policy_evaluate(n_games=10):
    """
    Evaluate the trained policy by playing against the pure MCTS player
    Note: this is only for monitoring the progress of training
    """
    current_mcts_player = MCTSPlayer(agent.policy_value_fn,c_puct=c_puct,n_playout=n_playout)
    pure_mcts_player = MCTS_Pure(c_puct=5, n_playout=pure_mcts_playout_num)
    win_cnt = defaultdict(int)
    for i in range(n_games):
        a = time.time()
        winner = game.start_play(current_mcts_player,
                                        pure_mcts_player,
                                        start_player=i % 2,
                                        is_shown=0)
        print(f"game {i} cost time: "+str(time.time() - a)+" s")
        win_cnt[winner] += 1
    win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
    print(f"pid=={os.getpid()}=="+"num_playouts:{}, win: {}, lose: {}, tie:{}".format(
        pure_mcts_playout_num,
        win_cnt[1], win_cnt[2], win_cnt[-1]))
    return win_ratio


c_puct = 5
n_games = 10
best_win_ratio=0.0
n_playout = 600  # num of simulations for each move 深度mcst模拟次数
pure_mcts_playout_num=1000

n_in_row = 5
board = Board(width=variables.board_width,
                           height=variables.board_height,
                           n_in_row=n_in_row)
game = Game(board)
agent = PolicyValueNet(board_width=variables.board_width,board_height=variables.board_height,model_file=variables.eval_model)


if __name__=='__main__':
    a = time.time()
    win_ratio = policy_evaluate()
    print("eval cost time: "+str(time.time() - a)+" s")
    if win_ratio > best_win_ratio:
        print("New best policy!!!!!!!!")
        best_win_ratio = win_ratio
        # update the best_policy
        agent.save_model('./best_policy.model')
        if (best_win_ratio == 1.0 and
                pure_mcts_playout_num < 5000):
            pure_mcts_playout_num += 1000
            best_win_ratio = 0.0

