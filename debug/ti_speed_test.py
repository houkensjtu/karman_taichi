import taichi as ti
import time

ti.init(arch=ti.gpu, kernel_profiler=True)

n = 1024*1024
v1 = ti.field(dtype=ti.f64, shape = n)
v2 = ti.field(dtype=ti.f64, shape = n)

@ti.kernel
def init():
    for i in range(n):
        v1[i] = 1.0
        v2[i] = 2.0

@ti.kernel
def reduce_para(v1:ti.template(), v2:ti.template())->ti.f64:
    n = v1.shape[0]
    # sum = ti.cast(0.0, ti.f64)
    sum = 0.0
    for i in range(n):
        sum += v1[i]*v2[i]
    return sum

@ti.kernel
def reduce_seri(v1:ti.template(), v2:ti.template())->ti.f64:
    n = v1.shape[0]
    # sum = ti.cast(0.0, ti.f64)
    sum = 0.0    
    for _ in range(1):
        for i in range(n):
            sum += v1[i]*v2[i]
    return sum

def reduce(v1, v2, n):
    sum = 0.0
    for i in range(n):
        sum += v1[i]*v2[i]
    return sum

print('Initializing...')
init()

# print('Reducing in Python scope...')
# start = time.time()
# print(reduce(v1, v2, n))
# print(time.time() - start)

print('Reducing in Taichi scope parallel...')
start = time.time()
print(reduce_para(v1, v2))
print(time.time() - start)
ti.profiler.print_kernel_profiler_info('trace')
ti.profiler.clear_kernel_profiler_info()


print('Reducing in Taichi scope serial...')
start = time.time()
print(reduce_seri(v1, v2))
print(time.time() - start)
ti.profiler.print_kernel_profiler_info('trace')
ti.profiler.clear_kernel_profiler_info()
