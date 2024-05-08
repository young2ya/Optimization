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

######### Problem 1 ######################
def P1_initial_point():
    return np.random.choice(np.arange(15), size=15, replace=False)  # index --> 위치, value --> 부서

def P1_Assess(X):
    mat = np.array([[0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6],
[10, 0, 1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4, 5],
[0, 1, 0, 1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 3, 4],
[5, 3, 10, 0, 1, 4, 3, 2, 1, 2, 5, 4, 3, 2, 3],
[1, 2, 2, 1, 0, 5, 4, 3, 2, 1, 6, 5, 4, 3, 2],
[0, 2, 0, 1, 3, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5],
[1, 2, 2, 5, 5, 2, 0, 1, 2, 3, 2, 1, 2, 3, 4],
[2, 3, 5, 0, 5, 2, 6, 0, 1, 2, 3, 2, 1, 2, 3],
[2, 2, 4, 0, 5, 1, 0, 5, 0, 1, 4, 3, 2, 1, 2],
[2, 0, 5, 2, 1, 5, 1, 2, 0, 0, 5, 4, 3, 2, 1],
[2, 2, 2, 1, 0, 0, 5, 10, 10, 0, 0, 1, 2, 3, 4],
[0, 0, 2, 0, 3, 0, 5, 0, 5, 4, 5, 0, 1, 2, 3],
[4, 10, 5, 2, 0, 2, 5, 5, 10, 0, 0, 3, 0, 1, 2],
[0, 5, 5, 5, 5, 5, 1, 0, 0, 0, 5, 3, 10, 0, 1],
[0, 0, 5, 0, 5, 10, 0, 0, 2, 5, 0, 0, 2, 4, 0]])
    cost_list = []

    length = X.shape[0]
    for i in range(length): # i 위치의 부서와 (i+1, i+2, ..., 15) 위치의 부서와의 flow cost
        for j in range(i+1, length):
            loc = sorted([X[i], X[j]])  # i, j 위치의 부서 오름차순 정렬
            cost_list.append(mat[loc[0], loc[1]] * mat[j,i])    # distance * flow

    return sum(cost_list)

def P1_tweak(X, change_num):    # change_num: total number of tweak
    for change in range(change_num):
        two_idx = np.random.choice(len(X), size=2, replace=False)   # pick 2 locations randomly (인접하지 않아도 선택)

        # interchange 2 value
        dep = X[two_idx[0]]
        X[two_idx[0]] = X[two_idx[1]]
        X[two_idx[1]] = dep
    return X

def P1_SA(search_loop, change_num, alpha, start_temp, stop_temp):
    print(f"start temperature: {start_temp}")
    T = start_temp

    X = P1_initial_point()
    Best = copy.deepcopy(X)
    init_point = copy.deepcopy(X)

    hist_q = [P1_Assess(X)]
    hist_b = [P1_Assess(Best)]
    hist_temp = [T]

    while T > stop_temp:
        for rep in range(search_loop):

            # tweak
            R = P1_tweak(X, change_num) # R --> tweak location
            delta = P1_Assess(R) - P1_Assess(X)

            if delta < 0:   # improving
                X = copy.deepcopy(R)
            else:   # non-improving
                if np.random.rand() < np.exp(-delta / sc.Boltzmann * T):
                    X = copy.deepcopy(R)

            # update Best
            if P1_Assess(X) < P1_Assess(Best):
                Best = copy.deepcopy(X)

            # history
            hist_q.append(P1_Assess(X))
            hist_b.append(P1_Assess(Best))

        T = T * alpha

        hist_temp.append(T)
        if len(hist_temp)%100 == 0:
            print(f"Temperature lowered {len(hist_temp)} times\nCurrent temperature: {round(T, 4)}\nCurrent Best: {P1_Assess(Best)}")

    print(f"Final quality: {P1_Assess(Best)}")
    print(f"Total repeat: {len(hist_q)}")
    return P1_Assess(Best), Best, init_point, [hist_q, hist_b]

def P1_C1_SA(search_loop, change_num, start_temp, stop_temp):   # change the annealing schedule --> start_temp/100
    print(f"start temperature: {start_temp}")
    T = start_temp

    X = P1_initial_point()
    Best = copy.deepcopy(X)
    init_point = copy.deepcopy(X)

    hist_q = [P1_Assess(X)]
    hist_b = [P1_Assess(Best)]
    hist_temp = [T]

    while T > stop_temp:
        for rep in range(search_loop):

            # tweak
            R = P1_tweak(X, change_num) # R --> tweak location
            delta = P1_Assess(R) - P1_Assess(X)

            if delta < 0:   # improving
                X = copy.deepcopy(R)
            else:   # non-improving
                if np.random.rand() < np.exp(-delta / sc.Boltzmann * T):
                    X = copy.deepcopy(R)

            # update Best
            if P1_Assess(X) < P1_Assess(Best):
                Best = copy.deepcopy(X)

            # history
            hist_q.append(P1_Assess(X))
            hist_b.append(P1_Assess(Best))

        T = T - start_temp/100

        hist_temp.append(T)
        if len(hist_temp)%100 == 0:
            print(f"Temperature lowered {len(hist_temp)} times\nCurrent temperature: {round(T, 4)}\nCurrent Best: {P1_Assess(Best)}")

    print(f"Final quality: {P1_Assess(Best)}")
    print(f"Total repeat: {len(hist_q)}")
    return P1_Assess(Best), Best, init_point, [hist_q, hist_b]

def P1_C4_SA(init_point, search_loop, change_num, alpha, start_temp, stop_temp):
    print(f"start temperature: {start_temp}")
    T = start_temp

    X = init_point
    Best = copy.deepcopy(X)
    init_point = copy.deepcopy(X)

    hist_q = [P1_Assess(X)]
    hist_b = [P1_Assess(Best)]
    hist_temp = [T]

    while T > stop_temp:
        for rep in range(search_loop):

            # tweak
            R = P1_tweak(X, change_num) # R --> tweak location
            delta = P1_Assess(R) - P1_Assess(X)

            if delta < 0:   # improving
                X = copy.deepcopy(R)
            else:   # non-improving
                if np.random.rand() < np.exp(-delta / sc.Boltzmann * T):
                    X = copy.deepcopy(R)

            # update Best
            if P1_Assess(X) < P1_Assess(Best):
                Best = copy.deepcopy(X)

            # history
            hist_q.append(P1_Assess(X))
            hist_b.append(P1_Assess(Best))

        T = T * alpha

        hist_temp.append(T)
        if len(hist_temp)%100 == 0:
            print(f"Temperature lowered {len(hist_temp)} times\nCurrent temperature: {round(T, 4)}\nCurrent Best: {P1_Assess(Best)}")

    print(f"Final quality: {P1_Assess(Best)}")
    print(f"Total repeat: {len(hist_q)}")
    return P1_Assess(Best), Best, init_point, [hist_q, hist_b]

######### Problem 2 ######################
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

def P2_C1_SA(search_loop, change_num, start_temp, stop_temp):
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

        T = T - start_temp/100

        hist_temp.append(T)
        if len(hist_temp)%100 == 0:
            print(f"Temperature lowered {len(hist_temp)} times\nCurrent temperature: {round(T, 4)}\nCurrent Best: {P2_Assess(Best)}")
    print(f"Total quality: {P2_Assess(Best)}")
    print(f"Count: {len(hist_q)}")

    return P2_Assess(Best), Best, init_point, [hist_q, hist_b]

def P2_C4_SA(init_point, search_loop, change_num, alpha, start_temp, stop_temp):
    print(f"Start temperature: {start_temp}")
    T = start_temp

    X = init_point
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

            # History
            hist_q.append(P2_Assess(X))
            hist_b.append(P2_Assess(Best))

            if delta < 0:   # improving
                X = copy.deepcopy(X_tweaked)
            else:   # non-improving
                if np.random.rand() < np.exp(-delta / (sc.Boltzmann* T)):
                    X = copy.deepcopy(X_tweaked)


            # Update best
            if P2_Assess(X) < P2_Assess(Best):
                Best = copy.deepcopy(X)

        T = T * alpha

        hist_temp.append(T)
        if len(hist_temp)%100 == 0:
            print(f"Temperature lowered {len(hist_temp)} times\nCurrent temperature: {round(T, 4)}\nCurrent Best: {P2_Assess(Best)}")
    print(f"Total quality: {P2_Assess(Best)}")
    print(f"Count: {len(hist_q)}")

    return P2_Assess(Best), Best, init_point, [hist_q, hist_b]






