# -*- coding: utf-8 -*-

"""
根据围棋局面，计算对局结果

使用NumPy二维ubyte数组存储局面，约定：
0 - 空
1 - 黑子
2 - 白子
3 - 黑空
4 - 白空
5 - 黑子统计标识
6 - 白子统计标识
7 - 黑残子
8 - 白残子
9 - 单官/公气/统计标识
"""

import numpy as np
import os

os.system('')

def show_phase(phase):
    """显示局面"""
    
    for i in range(19):
        for j in range(19):
            if phase[i,j] == 1: 
                chessman = chr(0x25cf)
            elif phase[i,j] == 2:
                chessman = chr(0x25cb)
            elif phase[i,j] == 9:
                chessman = chr(0x2606)
            else:
                if i == 0:
                    if j == 0:
                        chessman = '%s '%chr(0x250c)
                    elif j == 18:
                        chessman = '%s '%chr(0x2510)
                    else:
                        chessman = '%s '%chr(0x252c)
                elif i == 18:
                    if j == 0:
                        chessman = '%s '%chr(0x2514)
                    elif j == 18:
                        chessman = '%s '%chr(0x2518)
                    else:
                        chessman = '%s '%chr(0x2534)
                elif j == 0:
                    chessman = '%s '%chr(0x251c)
                elif j == 18:
                    chessman = '%s '%chr(0x2524)
                else:
                    chessman = '%s '%chr(0x253c)
            print('\033[0;30;43m' + chessman + '\033[0m', end='')
        print()
        
def find_blank(phase, cell):
    """找出包含cell的成片的空格"""
    
    def _find_blank(phase, result, cell):
        i, j = cell
        phase[i,j] = 9
        result['cross'].add(cell)
        
        if i-1 > -1:
            if phase[i-1,j] == 0:
                _find_blank(phase, result, (i-1,j))
            elif phase[i-1,j] == 1:
                result['b_around'].add((i-1,j))
            elif phase[i-1,j] == 2:
                result['w_around'].add((i-1,j))
        if i+1 < 19:
            if phase[i+1,j] == 0:
                _find_blank(phase, result, (i+1,j))
            elif phase[i+1,j] == 1:
                result['b_around'].add((i+1,j))
            elif phase[i+1,j] == 2:
                result['w_around'].add((i+1,j))
        if j-1 > -1:
            if phase[i,j-1] == 0:
                _find_blank(phase, result, (i,j-1))
            elif phase[i,j-1] == 1:
                result['b_around'].add((i,j-1))
            elif phase[i,j-1] == 2:
                result['w_around'].add((i,j-1))
        if j+1 < 19:
            if phase[i,j+1] == 0:
                _find_blank(phase, result, (i,j+1))
            elif phase[i,j+1] == 1:
                result['b_around'].add((i,j+1))
            elif phase[i,j+1] == 2:
                result['w_around'].add((i,j+1))
    
    result = {'cross':set(), 'b_around':set(), 'w_around':set()}
    _find_blank(phase, result, cell)
    
    return result

def find_blanks(phase):
    """找出所有成片的空格"""
    
    blanks = list()
    while True:
        cells = np.where(phase==0)
        if cells[0].size == 0:
            break
        
        blanks.append(find_blank(phase, (cells[0][0], cells[1][0])))
    
    return blanks

def stats(phase):
    """统计结果"""
    
    temp = np.copy(phase)
    for item in find_blanks(np.copy(phase)):
        if len(item['w_around']) == 0:
            v = 3 # 黑空
        elif len(item['b_around']) == 0:
            v = 4 # 白空
        else:
            v = 9 # 单官或公气
        
        for i, j in item['cross']:
            temp[i, j] = v
    
    black = temp[temp==1].size + temp[temp==3].size
    white = temp[temp==2].size + temp[temp==4].size
    common = temp[temp==9].size
    
    return black, white, common

if __name__ == '__main__':
    phase = np.array([
        [0,0,2,1,1,0,1,1,1,2,0,2,0,2,1,0,1,0,0],
        [0,0,2,1,0,1,1,1,2,0,2,0,2,2,1,1,1,0,0],
        [0,0,2,1,1,0,0,1,2,2,0,2,0,2,1,0,1,0,0],
        [0,2,1,0,1,1,0,1,2,0,2,2,2,0,2,1,0,1,0],
        [0,2,1,1,0,1,1,2,2,2,2,0,0,2,2,1,0,1,0],
        [0,0,2,1,1,1,1,2,0,2,0,2,0,0,2,1,0,0,0],
        [0,0,2,2,2,2,1,2,2,0,0,0,0,0,2,1,0,0,0],
        [2,2,2,0,0,0,2,1,1,2,0,2,0,0,2,1,0,0,0],
        [1,1,2,0,0,0,2,2,1,2,0,0,0,0,2,1,0,0,0],
        [1,0,1,2,0,2,1,1,1,1,2,2,2,0,2,1,1,1,1],
        [0,1,1,2,0,2,1,0,0,0,1,2,0,2,2,1,0,0,1],
        [1,1,2,2,2,2,2,1,0,0,1,2,2,0,2,1,0,0,0],
        [2,2,0,2,2,0,2,1,0,0,1,2,0,2,2,2,1,0,0],
        [0,2,0,0,0,0,2,1,0,1,1,2,2,0,2,1,0,0,0],
        [0,2,0,0,0,2,1,0,0,1,0,1,1,2,2,1,0,0,0],
        [0,0,2,0,2,2,1,1,1,1,0,1,0,1,1,0,0,0,0],
        [0,2,2,0,2,1,0,0,0,0,1,0,0,0,0,1,1,0,0],
        [0,0,2,0,2,1,0,1,1,0,0,1,0,1,0,1,0,0,0],
        [0,0,0,2,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0]
    ], dtype=np.ubyte)
    
    show_phase(phase)
    black, white, common = stats(phase)
    print('--------------------------------------')
    print('黑方：%d，白方：%d，公气：%d'%(black, white, common))

