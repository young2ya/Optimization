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

def Initial_Point():
    return np.random.rand(15)

def Get_Population(pop_size):
    population = []
    for chrom_idx in range(pop_size):
        population.append(Initial_Point())
    return population

def Assign_idx(X):
    sorted_X = sorted(X)
    indexes = {num: index for index, num in enumerate(sorted_X)}
    return [indexes[num] for num in X]

def Assign_idx_population(population):
    idx_population = []
    for chrom in population:
        idx_population.append(Assign_idx(chrom))
    return np.array(idx_population)

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

    idx_X = Assign_idx(X)
    length = X.shape[0]
    cost = 0
    for i in range(length - 1):
        for j in range(i+1, length):
            loc = sorted([idx_X[i], idx_X[j]])
            cost += fd_mat[loc[0], loc[1]] * fd_mat[j,i]   # distance * flow

    return cost # evaluation score

def Evaluate_Population(population):
    eval_list = []
    for chrom in population:
        eval_list.append(Evaluate(chrom))
    return np.array(eval_list)


pop_size = 50
population = Get_Population(pop_size)
pop_cost = Evaluate_Population(population)
