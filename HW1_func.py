import random
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os

def Generate_Random_Value():
    return (np.random.rand(2) * 6 - 3).astype('float64')


def Assess(X):
    term1 = (1-X[0])**2
    term2 = 100*(X[1]-X[0]**2)**2
    return term1 + term2


def Derivate(X):
    derivate_x = 400*(X[0]**3) + 2*X[0]*(1 - 200*X[1]) - 2
    derivate_y = 200*X[1] - 200*(X[0]**2)
    return np.array([derivate_x, derivate_y])


def Hessian_Matrix_Inverse(X):
    hessian = np.array([[  1200*(X[0]**2) + 2 - 400*X[1],   -400*X[0]   ],
                       [  -400*X[0]                     ,    200         ]])
    return np.linalg.inv(hessian)   # 역행렬


def Tweak_Bounded_Uniform_Convolution(X, r, p):
    for i in range(2):  # 의사결정변수 2개에 독립적으로 tweak 여부 결정
        if p >= np.random.rand():
            while True:
                noise = np.random.uniform(-r, r)
                if -3 <= X[i] + noise < 3:
                    break
            X[i] = X[i] + noise
    return X


def Plot_history(hist, name, path, save):
    cnt = list(range(len(hist[0])))
    plt.ioff()
    fig, axs = plt.subplots(3, 1, constrained_layout=True, sharex=True)
    fig.suptitle(f'{name}\'s History Figure')

    axs[0].plot(cnt, hist[0], 'r')
    axs[0].set_title('history of x')
    axs[0].set_ylabel('x variable')
    axs[0].grid(True)

    axs[1].plot(cnt, hist[1], 'b')
    axs[1].set_title('history of y')
    axs[1].set_ylabel('y variable')
    axs[1].grid(True)

    axs[2].plot(cnt, hist[2], 'g')
    axs[2].set_title('history of quality')
    axs[2].set_ylabel('quality f(x,y)')
    axs[2].grid(True)

    if save:
        plt.savefig(os.path.join(path, f"{name}.png"))
    else:
        plt.show()
    return


# Search Algorithms
def Gradient_Descent(due_time, alpha, threshold):
    print(f"\nGradient_Descent alpha={alpha}")
    start_time = time.time()

    # initial random point
    X = Generate_Random_Value()
    init_point = copy.deepcopy(X)

    hist_x = [X[0]]
    hist_y = [X[1]]
    hist_q = [Assess(X)]

    cnt = 0
    while time.time() - start_time < due_time:
        cnt += 1

        # update decision variable
        X = X - alpha*Derivate(X)   # step size -> alpha 만큼 이동

        # history of searching
        hist_x.append(X[0])
        hist_y.append(X[1])
        hist_q.append(Assess(X))

        # ideal solution
        if Assess(X) <= threshold:
            break

    total_time = time.time()-start_time
    print(f"x: {X[0]}, y: {X[1]}\nf(x,y): {Assess(X)}\nrunning_time: {total_time}")
    print(f"count:{cnt}")

    return X, Assess(X), total_time, [hist_x, hist_y, hist_q], cnt, init_point


def Newton_Method(due_time, alpha, threshold):
    print(f"\nNewton_Method alpha={alpha}")
    start_time = time.time()

    # initial random point
    X = Generate_Random_Value()
    init_point = copy.deepcopy(X)

    hist_x = [X[0]]
    hist_y = [X[1]]
    hist_q = [Assess(X)]

    cnt = 0
    while time.time() - start_time < due_time:
        cnt += 1

        # update decision variable
        X = X - alpha*Hessian_Matrix_Inverse(X).dot(Derivate(X).reshape(2, 1)).reshape(2,)

        # history of searching
        hist_x.append(X[0])
        hist_y.append(X[1])
        hist_q.append(Assess(X))

        # ideal solution
        if Assess(X) <= threshold:
            break

    total_time = time.time()-start_time
    print(f"x: {X[0]}, y: {X[1]}\nf(x,y): {Assess(X)}\nrunning_time: {total_time}")
    print(f"count:{cnt}")

    return X, Assess(X), total_time, [hist_x, hist_y, hist_q], cnt, init_point


def Hill_Climbing(due_time, threshold, noise_range, tweak_probability):
    print(f"\nHill-Climbing noise_range={noise_range}")

    start_time = time.time()
    X = Generate_Random_Value()
    init_point = copy.deepcopy(X)
    hist_x = [X[0]]
    hist_y = [X[1]]
    hist_q = [Assess(X)]

    cnt = 0
    while time.time() - start_time < due_time:
        cnt += 1

        # Tweak
        R = Tweak_Bounded_Uniform_Convolution(copy.deepcopy(X), noise_range, tweak_probability)
        if Assess(R) < Assess(X):
            X = R

        # history of updating decision variable
        hist_x.append(X[0])
        hist_y.append(X[1])
        hist_q.append(Assess(X))

        # ideal solution
        if Assess(X) <= threshold:
            break

    total_time = time.time() - start_time
    print(f"x: {X[0]}, y: {X[1]}\nf(x,y): {Assess(X)}\nrunning_time: {total_time}")
    print(f"count:{cnt}")

    return X, Assess(X), total_time, [hist_x, hist_y, hist_q], cnt, init_point


# def Distance_to_11(arr):
#     return np.sqrt(np.power(arr - np.ones(2, ), 2).sum())


# def Analysis_result(df_list, algo, figure_path):
#     repeat_time = df_list[0].shape[0]
#     for idx, df in enumerate(df_list):
#         if algo == 'GD' or algo == 'NM':
#             alpha = df['alpha'][0]
#             print(f"Average of Repeatation for each alpha={alpha}")
#
#         elif algo == 'HC':
#             noise = df['noise'][0]
#             print(f"Average of Repeatation for each noise={noise}")
#
#         # Average of Repeatation
#         print(f"{df[['quality', 'running_time', 'count']].mean()}")
#
#         # Relationship between Initial Point <--> Running Time
#         print(f"Relationship between Initial Point <--> Running_Time\n"
#               f"{df[['init_point', 'running_time']]}")
#         for x in range(repeat_time):
#             print(f"distance_to_11 : {Distance_to_11(df['init_point'][x])}")
#
#         # Ploting Search_history
#         if algo == 'GD' or algo == 'NM':
#             for x in range(repeat_time):
#                 Plot_history(hist=df.iloc[x, 4], name=f"{algo}_{alpha}_{x + 1}", path=figure_path, save=True)
#         elif algo == 'HC':
#             for x in range(repeat_time):
#                 Plot_history(hist=df.iloc[x, 4], name=f"{algo}_{noise}_{x + 1}", path=figure_path, save=True)
#
#     return print(f"Analysis on {algo} is done.")