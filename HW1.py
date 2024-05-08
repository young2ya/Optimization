##########################################
# 202421087 박윤영
# 조정최적화 H.W.1 Local Search Algorithms
#########################################
import os
from imp import reload
import numpy as np
import pandas as pd
import copy

import HW1_func as f

pd.options.display.float_format = '{:.6f}'.format
os.makedirs('figure', exist_ok=True)

##### 1. Experiment parameter

# Common parameter
Threshold = 1e-3    # Treshold 이하일 경우 ideal solution에 도달했다고 판단함
Runtime = 120       # Runtime을 초과할 경우 탐색을 중단 (sec)
Figure_path = os.path.join(os.getcwd(), 'figure')
Repeat_time = 5

# Gradient Descent & Newton's Method
Alpha_set_GD = [0.0001, 0.00001, 0.000001]      # 의사결정변수를 업데이트하는 정도 (alpha)
Alpha_set_NM = [0.01, 0.001, 0.0001]

# Hill-Climbing
Noise_range = [0.1, 0.01, 0.001]                # half range of noise --> tweak 정도
Tweak_prob = [1, 0.5, 0.3]                      # tweak 발생 확률 (1 이면 항상 tweak)


##### 2. Local Search Algorithms

# Gradient Descent Experiment
result_GD = []
for alpha in Alpha_set_GD:
    repeat_result = []
    for repeat in range(Repeat_time):
        best_X, quality, running_time, hist, cnt, init_point = f.Gradient_Descent(Runtime, alpha, Threshold)
        repeat_result.append([alpha, quality, best_X, running_time, hist, cnt, init_point])
    result_GD.append(repeat_result)

# Newton's Method
result_NM = []
for alpha in Alpha_set_NM:
    repeat_result = []
    for repeat in range(Repeat_time):
        best_X, quality, running_time, hist, cnt, init_point = f.Newton_Method(Runtime,alpha, Threshold)
        repeat_result.append([alpha, quality, best_X, running_time, hist, cnt, init_point])
    result_NM.append(repeat_result)

# Hill-Climbing (p=1)
result_HC = []
for noise in Noise_range:
    repeat_result = []
    for repeat in range(Repeat_time):
        best_X, quality, running_time, hist, cnt, init_point = f.Hill_Climbing(Runtime, Threshold, noise, 1)
        repeat_result.append([noise, quality, best_X, running_time, hist, cnt, init_point])
    result_HC.append(repeat_result)

##### 3. Analysis on the result

GD_df = [pd.DataFrame(each, columns=['alpha', 'quality', 'best_X', 'running_time', 'hist', 'count', 'init_point']) for each in result_GD]
NM_df = [pd.DataFrame(each, columns=['alpha', 'quality', 'best_X', 'running_time', 'hist', 'count', 'init_point']) for each in result_NM]
HC_df = [pd.DataFrame(each, columns=['noise', 'quality', 'best_X', 'running_time', 'hist', 'count', 'init_point']) for each in result_HC]

# f.Analysis_result(df_list=GD_df, algo='GD', figure_path=Figure_path)
# f.Analysis_result(df_list=NM_df, algo='NM', figure_path=Figure_path)
# f.Analysis_result(df_list=HC_df, algo='HC', figure_path=Figure_path)

