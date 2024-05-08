##########################################################
# 202421087 박윤영
# 조정최적화 H.W.4 GA
##########################################################
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

import HW4_func_1 as p1
import HW4_func_2 as p2


reload(p1)
# # # Problem 1 – Combinatorial Optimization: QAP problem

# Basic GA
print('1')
eval_basic, chrom_basic = p1.Genetic_Algorithm(repeatation=1000, pop_size=50, crossover_rate=0.3, mutation_rate=0.03)

# Change the initial starting points 10 times
print('1-1')
eval1_list = []
chrom1_list = []
for x in range(10):
    eval1, chrom1 = p1.Genetic_Algorithm(repeatation=1000, pop_size=50, crossover_rate=0.3, mutation_rate=0.03)
    eval1_list.append(eval1)
    chrom1_list.append(chrom1)

# Change the mutation probability twice
print('1-2')
eval2_1, chrom2_1 = p1.Genetic_Algorithm(repeatation=1000, pop_size=50, crossover_rate=0.3, mutation_rate=0.2)
eval2_2, chrom2_2 = p1.Genetic_Algorithm(repeatation=1000, pop_size=50, crossover_rate=0.3, mutation_rate=0.005)

# Change the population size twice
print('1-3')
eval3_1, chrom3_1 = p1.Genetic_Algorithm(repeatation=1000, pop_size=20, crossover_rate=0.3, mutation_rate=0.03)
eval3_2, chrom3_2 = p1.Genetic_Algorithm(repeatation=1000, pop_size=100, crossover_rate=0.3, mutation_rate=0.03)

# Change the stopping criteria twice
print('1-4')
eval4_1, chrom4_1 = p1.Genetic_Algorithm(repeatation=10000, pop_size=50, crossover_rate=0.3, mutation_rate=0.03)
eval4_2, chrom4_2 = p1.Genetic_Algorithm(repeatation=5000, pop_size=50, crossover_rate=0.3, mutation_rate=0.03)

# # # Problem 2 – Continuous optimization

# Basic GA
reload(p2)
print('2')
p2_eval_basic, p2_chrom_basic = p2.Genetic_Algorithm(repeatation=1000, pop_size=50, crossover_rate=0.3, mutation_rate=0.03)

# Change the initial starting points 10 times
print('2-1')
p2_eval1_list = []
p2_chrom1_list = []
for x in range(10):
    p2_eval1, p2_chrom1 = p2.Genetic_Algorithm(repeatation=1000, pop_size=50, crossover_rate=0.3, mutation_rate=0.03)
    p2_eval1_list.append(p2_eval1)
    p2_chrom1_list.append(p2_chrom1)

# Change the mutation probability twice
print('2-2')
p2_eval2_1, p2_chrom2_1 = p2.Genetic_Algorithm(repeatation=1000, pop_size=50, crossover_rate=0.3, mutation_rate=0.2)
p2_eval2_2, p2_chrom2_2 = p2.Genetic_Algorithm(repeatation=1000, pop_size=50, crossover_rate=0.3, mutation_rate=0.005)

# Change the population size twice
print('2-3')
p2_eval3_1, p2_chrom3_1 = p2.Genetic_Algorithm(repeatation=1000, pop_size=20, crossover_rate=0.3, mutation_rate=0.03)
p2_eval3_2, p2_chrom3_2 = p2.Genetic_Algorithm(repeatation=1000, pop_size=100, crossover_rate=0.3, mutation_rate=0.03)

# Change the stopping criteria twice
print('2-4')
p2_eval4_1, p2_chrom4_1 = p2.Genetic_Algorithm(repeatation=10000, pop_size=50, crossover_rate=0.3, mutation_rate=0.03)
p2_eval4_2, p2_chrom4_2 = p2.Genetic_Algorithm(repeatation=5000, pop_size=50, crossover_rate=0.3, mutation_rate=0.03)


p21mean = sum(p2_eval1_list) / len(p2_eval1_list)
p21best = min(p2_eval1_list)











# Test
# pop = p2.Get_Random_Population(5)
# encoded_pop = p2.Encoding_Population(pop)
# cross = p2.Crossover(encoded_pop, 0.3)
# muta = p2.Mutation(encoded_pop, 0.03)
#
# pool = encoded_pop + cross + muta
# pool_decoded = p2.Decoding_Population(pool)
#
# new_pop = p2.Selection(pool_decoded, 5)
# c = p2.Get_Best_Chromosome(new_pop)
# cs = p2.Evaluate(c)
#
# for pop in new_pop:
#     score = p2.Evaluate(pop)
#     print(f"score:{score}")
#
# X = p2.Initial_Point()
# X_en = p2.Encoding(X)
# X_de = p2.Decoding(X_en)
# print(f"{X}\n{X_en}\n{X_de}")
