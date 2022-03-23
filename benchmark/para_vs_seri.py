# See related issue discussion on Github:
# https://github.com/taichi-dev/taichi/issues/4541
# Important conlusions:
# 1. Skip first kernel run to avoid compilation time
# 2. Tweak block_dim for cpu to maximize perf
# 3. Problem size needs to be large enough for para to outperform seri

import taichi as ti
import time

ti.init(arch=ti.cpu, default_fp=ti.f32)

n = 8192 # n needs to be large enough for cpu to unleash its parallel power
v1 = ti.field(dtype=float, shape = n)
v2 = ti.field(dtype=float, shape = n)

@ti.kernel
def init():
    for i in range(n):
        v1[i] = 1.0
        v2[i] = 2.0

@ti.kernel
def reduce_para()->ti.f32:
    n = v1.shape[0]
    sum = 0.0
    ti.block_dim(32) # larger block_dim leads to less overhead; default dim = 32
    for i in range(n):
        sum += v1[i]*v2[i]
    return sum

@ti.kernel
def reduce_seri()->ti.f32:
    n = v1.shape[0]
    sum = 0.0
    ti.block_dim(32)    
    for _ in range(1):
        for i in range(n):
            sum += v1[i]*v2[i]
    return sum

print('Initializing...')
init()
reduce_para() # Skip the first run to avoid compilation time
reduce_seri() # Skip the first run to avoid compilation time

print('Reducing in Taichi scope with a parallel kernel...')
start = time.perf_counter()
for _ in range(1000):
    reduce_para()
print(time.perf_counter() - start)

print('Reducing in Taichi scope with serial kernel...')
start = time.perf_counter()
for _ in range(1000):
    reduce_seri()
print(time.perf_counter() - start)
