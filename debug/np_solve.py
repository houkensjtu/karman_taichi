import taichi as ti
import numpy as np
from bicg import bicgstab
ti.init(arch=ti.cpu)

n = 8
A = np.random.rand(n, n)
b = np.random.rand(n)
A_ti = ti.field(dtype=ti.f32)
ti.root.dense(ti.ij, (n, n)).place(A_ti)

# Copy matrix from A to A_ti; 
A_ti.from_numpy(A)

# 5-diag A and A_ti
for i in range(n):
    for j in range(n):
        if i == j or abs(i-j)==1 or abs(i-j)==3:
            pass
        else:
            A[i,j] = 0
            A_ti[i,j] = 0

x = np.linalg.solve(A, b)
print(np.allclose(np.dot(A,x), b))
