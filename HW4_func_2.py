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

def Initial_Point(precision=4): # x: -3 ~ 3 범위 값, y: -2 ~ 2 범위 값
    x = np.random.rand() * 6 - 3
    y = np.random.rand() * 4 - 2
    return [round(x, precision), round(y, precision)]

def Get_Random_Population(pop_size):
    # param : population 속 chromosome의 개수
    # return : type-list

    population =[]
    for chrom_idx in range(pop_size):
        population.append(Initial_Point())

    return population

def Evaluate(X):
    x, y = X
    objective_func = (4 - 2.1 * (x**2) + (x**4) / 3) * (x**2) + x*y + (-4 + 4 * (y**2))* (y**2)
    return objective_func

def Encoding(X):
    x, y = X

    # x_min, x_max, y_min, y_max = min_max_range
    x_min, x_max, y_min, y_max = -3, 3, -2, 2
    bits = 16

    x_binary_step = (x - x_min) * (2 ** bits - 1) / (x_max - x_min)
    x_encoded = bin(round(x_binary_step))[2:]
    if bits - len(x_encoded) > 0:
        zero_pad = '0' * (bits - len(x_encoded))
        x_encoded = zero_pad + x_encoded

    y_binary_step = (y - y_min) * (2 ** bits - 1) / (y_max - y_min)
    y_encoded = bin(round(y_binary_step))[2:]
    if bits - len(y_encoded) > 0:
        zero_pad = '0' * (bits - len(y_encoded))
        y_encoded = zero_pad + y_encoded

    return x_encoded + y_encoded

def Decoding(X_encoded, precision=4):
    x_encoded = X_encoded[:16]
    y_encoded = X_encoded[16:]

    x_min, x_max, y_min, y_max = -3, 3, -2, 2
    bits = 16

    x_binary_step = int(x_encoded, 2)
    x = round(x_min + x_binary_step * (x_max - x_min) / (2 ** bits - 1), precision)

    y_binary_step = int(y_encoded, 2)
    y = np.round(y_min + y_binary_step * (y_max - y_min) / (2 ** bits - 1), precision)

    return [x, y]

def Encoding_Population(population):
    encoded_population = []
    for pop_idx, pop in enumerate(population):
        encoded_population.append(Encoding(pop))
    return encoded_population

def Decoding_Population(population):
    decoded_population = []
    for pop_idx, pop in enumerate(population):
        decoded_population.append(Decoding(pop))
    return decoded_population

def Evaluate_population(population):
    score_list = []
    for pop in population:
        score_list.append(Evaluate(pop))
    return np.array(score_list)

def Get_Best_Chromosome(population):
    eval_list = []
    for chrom in population:
        eval_list.append(Evaluate(chrom))
    best_idx = eval_list.index(min(eval_list))

    return population[best_idx]

def Crossover(population, crossover_rate):
    chromosome_length = len(population[0])

    offspring_list = []
    for pop_idx, pop in enumerate(population):
        if np.random.rand() < crossover_rate:   # cross-over 실행
            couple = random.choice(population)
            pos = np.random.randint(1, chromosome_length - 1)   # 1~32 사이에에서 하나 선택

            offspring_list.append(pop[:pos] + couple[pos:]) # 두 개를 pos만큼 잘라서 섞음
            offspring_list.append(couple[:pos] + pop[pos:])

    return offspring_list

def Mutation(population, mutation_rate):
    chromosome_length = len(population[0])
    offspring_list = []

    for pop_idx, pop in enumerate(population):
        # 랜덤하게 mutation 진행할 idx 추출 (32개의 랜덤값이 각각 mutation보다 작을 경우 idx추출)
        mu_idx = np.where((np.random.rand(chromosome_length) < mutation_rate) == True)[0]

        # mutation이 하나도 일어나지 않는 경우
        if mu_idx.shape[0] == 0:
            pass

        # mutation이 한 개 이상 일어나는 경우
        else:
            offspring_list.append(Interchange_binary(pop, mu_idx))

    return offspring_list

def Interchange_binary(binary_value, change_indices):
    b_list = list(binary_value)

    for idx in change_indices:
        if b_list[idx] == '0':
            b_list[idx] = '1'

        elif b_list[idx] == '1':
            b_list[idx] = '0'

        else:
            print('binary 값이 아님')

    mutated = "".join(b_list)

    return mutated

def Selection(pool, pop_size):
    pool_size = len(pool)

    eval_arr = Evaluate_population(pool)
    emax = eval_arr.max()
    emin = eval_arr.min()
    fitness_arr = (emax - eval_arr + 0.00001)   # 0이 되지 않도록 함
    fitness = fitness_arr / fitness_arr.sum()   # 각 확률 값

    cumulative_prob = np.zeros(shape=pool_size)

    # cumulative pobability
    sum_prob = 0
    for prob_idx, prob in enumerate(fitness):
        sum_prob += prob
        cumulative_prob[prob_idx] = sum_prob

    selected = []
    for idx in range(pop_size):
        random_num = np.random.rand()
        selected.append(pool[np.sum(cumulative_prob < random_num)]) # random_num 보다 작은 확률 값

    return selected

def Genetic_Algorithm(repeatation, pop_size, crossover_rate, mutation_rate):
    # 초기 random population 생성
    population = Get_Random_Population(pop_size=pop_size)
    init_chrom = Get_Best_Chromosome(population)

    best_chrom = copy.deepcopy(init_chrom)
    best_score = Evaluate(best_chrom)

    for rep in range(repeatation):
        # Encoding
        population_encoded = Encoding_Population(population)

        # Crossover
        offspring_crossover = Crossover(population_encoded, crossover_rate)

        # Mutation
        offspring_mutation = Mutation(population_encoded, mutation_rate)

        # Pool decode
        pool = population_encoded + offspring_crossover + offspring_mutation
        pool_decoded = Decoding_Population(pool)

        # Selection
        population = Selection(pool_decoded, pop_size)

        # Save Best
        temp_chrom = Get_Best_Chromosome(population)
        temp_score = Evaluate(temp_chrom)

        if temp_score < best_score:
            print(f"{best_score} --> {temp_score}")
            best_score = copy.deepcopy(temp_score)
            best_chrom = copy.deepcopy(temp_chrom)
        if (rep + 1) % 1000 == 0:
            print(f"{(rep + 1) // 1000} 번째 best score: {best_score}")
        if best_score == -1.0316:
            print(f"최적해 발견 {best_score}")
            break

    print(f"best_score = {best_score}")
    print(f"best_chrom = {best_chrom}")

    return best_score, best_chrom
