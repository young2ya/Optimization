import sys
import os
import time
import random
import copy
from imp import reload

import matplotlib.pyplot as plt
import scipy.constants as sc
import numpy as np
import pandas as pd

# Problem 1
def Initial_Point():
    return np.random.choice(np.arange(15), size=15, replace=False)

def Get_Random_Population(pop_size):    # 초기 15개의 chromosome 생성
    # param : population 속 chromosome의 개수
    # return : type-list
    population = []
    for chrom_idx in range(pop_size):
        population.append(Initial_Point())
    return population

def Evaluate(X):
    fd_mat = np.array([[0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6],
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

    length = X.shape[0]
    cost = 0
    for i in range(length - 1):
        for j in range(i+1, length):
            loc = sorted([X[i], X[j]])
            cost += fd_mat[loc[0], loc[1]] * fd_mat[j, i]   # distance * flow

    return cost # evaluation score

def Evaluate_Population(population):
    eval_list = []
    for chrom in population:
        eval_list.append(Evaluate(chrom))
    return np.array(eval_list)

def Get_Best_Chromosome(population):
    eval_list = []
    for chrom in population:
        eval_list.append(Evaluate(chrom))
    best_idx = eval_list.index(min(eval_list))

    return population[best_idx]

def Crossover(population, crossover_rate):
    chrom_length = len(population)
    all_index = set(range(chrom_length))
    offspring_list = []

    for chrom_idx, chrom in enumerate(population):
        if np.random.rand() < crossover_rate:
            couple = population[random.choice(list(all_index - {chrom_idx}))]   # 짝 부모해를 하나 선정
            offspring_list.append(Non_Duplicate_Interchange(chrom, couple)) # crossover한 offspring을 list에 저장

    return offspring_list

def Non_Duplicate_Interchange(A,B): # 부모해 A, B
    length = A.shape[0]
    new = np.where(A == B, A, -1)   # A, B가 중복되지않으면 -1로 표기

    for i in range(length):
        if new[i] < 0:  # new가 -1을 가질때
            candidate = np.array([A[i], B[i]])
            isduplicate = np.isin(candidate, new)

            # 중복되지 않도록 new 업데이트
            if isduplicate.all() == True: # 두 요소 모두 기존 new에서 중복
                pass
            elif isduplicate.any() == True: # 둘 중 하나는 중복되지 않으면
                for j in range(len(isduplicate)):
                    if isduplicate[j] == False:
                        new[i] = np.copy(candidate[j])
                # new[i] = np.copy(candidate[~isduplicate])
            else: # 중복 없음
                new[i] = np.where(np.random.rand() < 0.5, A[i], B[i])

    # 채워지지않은 index 채우기
    unused_values = []
    unused_index = 0
    for i in range(15):
        if i not in new:
            unused_values.append(i)

    for i in range(len(new)):
        if new[i] == -1:
            new[i] = unused_values[unused_index]
            unused_index += 1

    # new[new == -1] = np.arange(15)[~np.isin(np.arange(15), new)]    # 못 채워진 index 채우기

    return new

def Mutation(population, mutation_rate):
    chromosome_length = len(population[0])
    offspring_list = []

    for chrom_idx, chrom in enumerate(population):
        mutation_indices = np.where((np.random.rand(chromosome_length) < mutation_rate) == True)[0]

        # mutation이 하나도 일어나지 않는 경우
        if mutation_indices.shape[0] == 0:
            pass

        # mutation이 한 개 이상 일어나는 경우
        else:
            offspring_list.append(Interchange_Location(chrom, mutation_indices))

    return offspring_list

def Interchange_Location(chrom, mutation_indices):  # 랜덤하게 선택된 두 index switch
    for mu_idx in mutation_indices:

        couple_idx = np.random.choice(chrom.shape[0])

        tmp = chrom[mu_idx]
        chrom[mu_idx] = chrom[couple_idx]
        chrom[couple_idx] = tmp

    return chrom

def Selection(pool, pop_size):
    pool_size = len(pool)

    eval_arr = Evaluate_Population(pool)

    # minimization
    emax = eval_arr.max()
    emin = eval_arr.min()
    fitness_arr = emax - eval_arr + 0.00001
    fitness = fitness_arr / fitness_arr.sum()   # eval 값이 작을수록 확률 up

    # cumulative pobability
    cumulative_prob = np.zeros(shape=pool_size)

    sum_prob = 0
    for prob_idx, prob in enumerate(fitness):
        sum_prob += prob
        cumulative_prob[prob_idx] = sum_prob

    # selection
    selected = []
    for idx in range(pop_size):
        random_num = np.random.rand()
        selected.append(pool[np.sum(cumulative_prob < random_num)]) # random_num 보다 작은 확률 값

    return selected

def Genetic_Algorithm(repeatation, pop_size, crossover_rate, mutation_rate):
    # 초기 random population 생성
    population = Get_Random_Population(pop_size=pop_size)
    # random population 중 best min 값 저장
    init_chrom = Get_Best_Chromosome(population)

    best_chrom = copy.deepcopy(init_chrom)
    best_score = Evaluate(best_chrom)

    for rep in range(repeatation):

        # Crossover
        offspring_crossover = Crossover(population, crossover_rate)

        # Mutation
        offspring_mutation = Mutation(population, mutation_rate)

        # Pool decode
        pool = population + offspring_crossover + offspring_mutation

        # Selection
        population = Selection(pool, pop_size)

        # Save Best
        temp_chrom = Get_Best_Chromosome(population)
        temp_score = Evaluate(temp_chrom)

        if temp_score < best_score:
            print(f"{best_score} --> {temp_score}")
            best_score = copy.deepcopy(temp_score)
            best_chrom = copy.deepcopy(temp_chrom)
        if (rep + 1) % 1000 == 0:
            print(f"{(rep + 1) // 1000}번째 best_score: {best_score}")
        if best_score == 575:
            break

    print(f"best_score = {best_score}")
    print(f"best_chrom = {best_chrom}")

    return best_score, best_chrom

