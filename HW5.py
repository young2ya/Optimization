import os
import sys
import numpy as np
from imp import reload
import HW5_func_2 as f

reload(f)
# 반복실험 횟수
REPEAT = 10

# Basic PSO
list_0 = []
for x in range(REPEAT):
    best_0, eval_0 = f.PSO_Ring(repeat_time=1000, pop_size=50, C1=2, C2=2)
    list_0.append(eval_0)
print(f"Mean : {sum(list_0) / REPEAT}")
print(f"Best : {min(list_0)}")