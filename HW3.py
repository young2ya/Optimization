##########################################
# 202421087 박윤영
# 조정최적화 H.W.3 Simulated Annealing
#########################################
import os
import sys
import time
import copy
from imp import reload

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.constants as sc

import HW3_func as f

sys.path.append(os.getcwd() + "/HW2")
# pd.options.display.flot_format = '{:.6f}'.format
ROOT_DIR = os.getcwd()

###########[ P1 ]##############
# best_quality, Best_sol, init_point, hist = f.P1_SA(search_loop=30, change_num= 1, alpha=0.99,
#                                                    start_temp=10000, stop_temp=1e-4)
# # c1
# c1_best_quality, c1_Best_sol, c1_init_point, c1_hist = f.P1_C1_SA(search_loop=30, change_num= 1,
#                                                                   start_temp=10000, stop_temp=1e-2)
# # c2
# c2_best_quality, c2_Best_sol, c2_init_point, c2_hist = f.P1_SA(search_loop=30, change_num= 1, alpha=0.99,
#                                                                   start_temp=10000, stop_temp=1e-6)
# # c3
# bq_list = []
# cnt_list = []
# for x in range(5):
#     c3_best_quality, c3_Best_sol, c3_init_point, c3_hist = f.P1_SA(search_loop=30, change_num= 1, alpha=0.99,
#                                                                   start_temp=10000, stop_temp=1e-2)
#     bq_list.append(c3_best_quality)
#     cnt_list.append(len(hist[0]))
# print(f"Quality: {min(bq_list)}")
# print(f"cnt : {min(cnt_list)}")
#
# # c4
# init_point = f.P1_initial_point()
# bq_list = []
# cnt_list = []
# for x in range(10):
#     np.random.seed(x)
#     c4_best_quality, c4_Best_sol, c4_init_point, c4_hist = f.P1_C4_SA(init_point=init_point, search_loop=30, change_num= 1,
#                                                                       alpha=0.99, start_temp=10000, stop_temp=1e-2)
#     bq_list.append(c4_best_quality)
#     cnt_list.append(len(hist[0]))
# print(f"Quality : {min(bq_list)}")
# print(f"cnt : {min(cnt_list)}")
#
# plt.clf()
#
# plt.plot(hist[0], label='')
# plt.plot(hist[1], label = 'best')
#
# plt.legend()
# plt.show()
###########[ P2 ]##############
def P2_Initial_point(): # x: -3~3   y: -2~2 랜덤하게 생성
    x = np.random.rand() * 6 - 3
    y = np.random.rand() * 4 - 2
    return [round(x, 5), round(y, 5)]

def P2_Assess(X):
    x, y = X
    Z = (4 - 2.1 * (x**2) + (x**4) / 3) * (x**2) + x * y + (-4 + 4 * (y**2)) * (y**2)
    return round(Z, 5)

def P2_Encoding(X): # real --> binary
    x, y = X
    # x_min, x_max, y_min, y_max = min_max_range
    x_min, x_max, y_min, y_max = -3, 3, -2, 2
    bits = 16 # x, y --> 16 bits 사용

    x_binary = (x - x_min) * (2**bits - 1) / (x_max - x_min)
    x_encoded = bin(round(x_binary))[2:]
    if bits - len(x_encoded) > 0: # 16 bits로 zero padding
        zero_pad = '0' * (bits - len(x_encoded))
        x_encoded = zero_pad + x_encoded

    y_binary = (y - y_min) * (2 ** bits - 1) / (y_max - y_min)
    y_encoded = bin(round(y_binary))[2:]
    if bits - len(y_encoded) > 0:  # 16 bits로 zero padding
        zero_pad = '0' * (bits - len(y_encoded))
        y_encoded = zero_pad + y_encoded

    return [x_encoded, y_encoded]

def P2_Decoding(Encoded):   # binary --> real
    x_encoded, y_encoded = Encoded
    x_min, x_max, y_min, y_max = -3, 3, -2, 2
    bits = 16

    x_binary = int(x_encoded, 2)
    x = round(x_min + x_binary * (x_max - x_min) / (2**bits - 1), 5)

    y_binary = int(y_encoded, 2)
    y = round(y_min + y_binary * (y_max - y_min) / (2**bits - 1), 5)

    return [x, y]

def P2_Tweak_binary(Encoded, tweak_num):
    x, y = Encoded
    bits = 16

    # binary string에서 tweak할 index 선택
    x_random_number = np.random.choice(np.arange(0, bits), size=tweak_num, replace=False)
    x_list = list(x)
    for num in x_random_number:
        if x_list[num] == '0':
            x_list[num] = '1'
        else:
            x_list[num] = '0'
    x_new = "".join(x_list)

    y_random_number = np.random.choice(np.arange(0, bits), size=tweak_num, replace=False)
    y_list = list(y)
    for num in y_random_number:
        if y_list[num] == '0':
            y_list[num] = '1'
        else:
            y_list[num] = '0'
    y_new = "".join(y_list)

    return [x_new, y_new]

def P2_SA(search_loop, change_num, alpha, start_temp, stop_temp):
    print(f"Start temperature: {start_temp}")
    T = start_temp

    X = P2_Initial_point()
    Best = copy.deepcopy(X)
    init_point = copy.deepcopy(X)

    hist_q = [P2_Assess(X)]
    hist_b = [P2_Assess(Best)]
    hist_temp = [T]

    while T > stop_temp:
        for rep in range(search_loop):

            # Tweak
            X_encoded = P2_Encoding(X)
            X_encoded_tweaked = P2_Tweak_binary(X_encoded, change_num)
            X_tweaked = P2_Decoding(X_encoded_tweaked)

            delta = P2_Assess(X_tweaked) - P2_Assess(X)

            if delta < 0:   # improving
                X = copy.deepcopy(X_tweaked)
            else:   # non-improving
                if np.random.rand() < np.exp(-delta / (sc.Boltzmann* T)):
                    X = copy.deepcopy(X_tweaked)

            # Update best
            if P2_Assess(X) < P2_Assess(Best):
                Best = copy.deepcopy(X)

            # History
            hist_q.append(P2_Assess(X))
            hist_b.append(P2_Assess(Best))

        T = T * alpha

        hist_temp.append(T)
        if len(hist_temp)%100 == 0:
            print(f"Temperature lowered {len(hist_temp)} times\nCurrent temperature: {round(T, 4)}\nCurrent Best: {P2_Assess(Best)}")
    print(f"Total quality: {P2_Assess(Best)}")
    print(f"Count: {len(hist_q)}")

    return P2_Assess(Best), Best, init_point, [hist_q, hist_b]
best_quality, Best_sol, init_point, hist = P2_SA(search_loop=30, change_num= 1, alpha=0.99,
                                                   start_temp=10000, stop_temp=1e-2)

# best_quality, Best_sol, init_point, hist = f.P2_SA(search_loop=30, change_num= 1, alpha=0.99,
#                                                    start_temp=10000, stop_temp=1e-2)
# # c1
# c1_best_quality, c1_Best_sol, c1_init_point, c1_hist = f.P2_C1_SA(search_loop=30, change_num= 1,
#                                                                   start_temp=10000, stop_temp=1e-2)
# # c2
# c2_best_quality, c2_Best_sol, c2_init_point, c2_hist = f.P2_SA(search_loop=30, change_num= 1, alpha=0.99,
#                                                                   start_temp=10000, stop_temp=1e-6)
# # c3
# bq_list = []
# cnt_list = []
# for x in range(5):
#     c3_best_quality, c3_Best_sol, c3_init_point, c3_hist = f.P2_SA(search_loop=30, change_num= 1, alpha=0.99,
#                                                                   start_temp=10000, stop_temp=1e-2)
#     bq_list.append(c3_best_quality)
#     cnt_list.append(len(hist[0]))
# print(f"Quality: {min(bq_list)}")
# print(f"cnt : {min(cnt_list)}")
#
# # c4
# init_point = f.P2_Initial_point()
# c4_bq_list = []
# c4_cnt_list = []
# for x in range(10):
#     np.random.seed(x)
#     c4_best_quality, c4_Best_sol, c4_init_point, c4_hist = f.P2_C4_SA(init_point=init_point, search_loop=30, change_num= 1,
#                                                                       alpha=0.99, start_temp=10000, stop_temp=1e-2)
#     c4_bq_list.append(c4_best_quality)
#     c4_cnt_list.append(len(c4_hist[0]))
# print(f"Quality : {min(c4_bq_list)}")
# print(f"cnt : {min(c4_cnt_list)}")

