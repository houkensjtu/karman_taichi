import taichi as ti
import time
import numpy as np
from numba import njit

ti.init(arch=ti.cpu, default_fp=ti.f64)

n = 819200
v1 = ti.field(dtype=float, shape = n)
v2 = ti.field(dtype=float, shape = n)

@ti.kernel
def init():
    for i in range(n):
        v1[i] = 1.0
        v2[i] = 2.0

@ti.kernel
def reduce_ti()->ti.f32:
    n = v1.shape[0]
    sum = 0.0
    # ti.block_dim(32) # Adaptive block_dim
    for i in range(n):
        sum += v1[i]*v2[i]
    return sum

@njit
def reduce_nb():
    sum = 0.0
    n = v1np.shape[0]
    for i in range(n):
        sum += v1np[i] * v2np[i]
    return sum

num_runs = 1000
print('Initializing...')
init()
v1np = v1.to_numpy()
v2np = v2.to_numpy()
reduce_ti() # Skip the first run to avoid compilation time
reduce_nb() # Skip the first run to avoid compilation time

print('Reducing in Taichi scope with a parallelized kernel...')
start = time.perf_counter()
for _ in range(num_runs):
    reduce_ti()
print(time.perf_counter() - start)

print('Reducing in Numba with @njit...')
start = time.perf_counter()
for _ in range(num_runs):
    reduce_nb()
print(time.perf_counter() - start)
