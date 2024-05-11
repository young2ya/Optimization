import os
import sys
import numpy as np
from imp import reload

def Random_Point(precision=4):  # 소수점 아래 4자리까지 x: -3~3, y: -2~2
    x = np.random.rand() * 6 - 3
    y = np.random.rand() * 4 - 2
    return np.array([round(x, precision), round(y, precision)])

def Get_Population(pop_size):   # pop_size : search할 군집 내 좌표 수
    return [Random_Point() for x in range(pop_size)]

def Evaluate(X):
    x = X[0]
    y = X[1]
    obj_func = (4 - 2.1 * (x**2) + (x**4) / 3) * (x**2) + x*y + (-4 + 4 * (y**2)) * (y**2)
    return obj_func

def Evaluate_Population(population):
    return [Evaluate(pop) for pop in population]

def Get_Best_from_Population(population):   # global best

    eval_list = Evaluate_Population(population)
    best = min(eval_list)
    best_idx = eval_list.index(best)
    best_chrom = population[best_idx]

    return best_chrom

def Get_Ring_Best(population, idx): # neighbor best -> idx 기준 양옆
    nei_score = []
    if idx == len(population) - 1:  # idx = 49
        for x in range(-1, 1):
            nei_score.append(Evaluate(population[idx + x])) # population[48, 49]
    else:
        for x in range(-1, 2):
            nei_score.append(Evaluate(population[idx + x])) # population[idx-1, idx, idx+1]

    best_idx = nei_score.index(min(nei_score)) - 1  # best_idx = best x
    ring_best = population[idx + best_idx]

    return ring_best

def Get_Ring_Best_Selfless(population, idx):    # neighbor best에 자기 자신은 포함하지 않는 버전 -> best일 경우 랜덤한 값을 best로 가져감
    nei_score = []

    if idx == len(population) - 1:
        for x in range(-1, 1):
            nei_score.append(Evaluate(population[idx + x]))
        best_idx = nei_score.idx(min(nei_score)) - 1

        if best_idx == 0:   # 자기 자신이 best면 random point를 personal best로 간주
            ring_best = Random_Point()
        else:
            ring_best = population[idx + best_idx]

    else:
        for x in range(-1, 2):
            nei_score.append(Evaluate(population[idx + x]))
        best_idx = nei_score.index(min(nei_score)) - 1

        if best_idx == 0:    # 자기 자신이 best면 random point를 personal best로 간주
            ring_best = Random_Point()
        else:
            ring_best = population[idx + best_idx]

    return  ring_best

# Basic PSO
def PSO_Ring(repeat_time, pop_size, C1, C2):

    population = Get_Population(pop_size=pop_size)  # 초기 랜덤 모집단
    velocity = [np.zeros(2) for x in range(pop_size)]   # 입자별 초기 속도는 0으로 설정
    particle_best = [np.copy(population[x]) for x in range(pop_size)]   # 입자별 베스트를 담을 리스트

    for rep in range(repeat_time):
        for part_idx in range(pop_size):

            # 이웃 베스트
            ring_best = Get_Ring_Best(population, part_idx)

            # velocity caculation
            r1, r2 = np.random.rand(2)  # 인지요소, 사회요소 rand 생성

            # 각 입자의 속도 업데이트
            velocity[part_idx] = (velocity[part_idx] + C1 * r1 * (particle_best[part_idx] - population[part_idx])
                                                     + C2 * r2 * (ring_best - population[part_idx]))

            # 각 입자의 위치 변경
            population[part_idx] = population[part_idx] + velocity[part_idx]

            # best update
            if Evaluate(population[part_idx]) < Evaluate(particle_best[part_idx]):
                particle_best[part_idx] = np.copy(population[part_idx])

    global_best = Get_Best_from_Population(particle_best)

    print(f"Best Chromosome is {global_best}")
    print(f"Evaluation of Best Chromosome is {Evaluate(global_best)}")

    return global_best, Evaluate(global_best)
