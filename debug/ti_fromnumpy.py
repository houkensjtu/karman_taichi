import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

n = 3
A = np.random.rand(n, n)
A_ti = ti.field(dtype=ti.f32)
ti.root.pointer(ti.ij, (n, n)).place(A_ti)

# Copy matrix from A to A_ti; 
A_ti.from_numpy(A)

print(A) # Random vals generated in numpy
print(A_ti.to_numpy()) # All zero...?
