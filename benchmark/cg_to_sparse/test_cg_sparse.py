from cgsolver import CGSolver
from spsolver import SparsePoissonSolver
import taichi as ti
import numpy as np
import time

ti.init(arch=ti.x64, kernel_profiler=True)

# Solve a 128x128 2-D Poisson Equation
# There are 128^2 unknowns in total
psize = 128
# Solve in Taichi using custom CG
# A is implicitly represented in compute_Ap()
now = time.time()
print('Solving using CGSolver...')
solver = CGSolver(psize, 1e-16, quiet=False) # quiet=False to print residual
solver.solve()
ti.profiler.print_kernel_profiler_info()
print('>>> Time spent using CG:', time.time() - now, 'sec')

# Solve in Taichi using SparseMatrixSolver
now = time.time()
# print('Solving using SparseSolver...')
spsolver = SparsePoissonSolver(psize, 'LU')
xsp = spsolver.solve()
print('>>> Time spent using SparseMatrixSolver:', time.time() - now, 'sec')

# Compare the results
print('Fetching results...') # Building matrix in Numpy is very slow...
xti = solver.build_x()       # Fetch solution from CG solver
a = solver.build_A()         # Build the lfs of the equation
b = solver.build_b()         # Build the rfs of the equation
print('Comparing the results...')
print('Residual cg:', np.linalg.norm(b - np.dot(a, xti)))
print('Residual sp:', np.linalg.norm(b - np.dot(a, xsp)))
print('np.linalg.norm(xsp-xti)=', np.linalg.norm(xti-xsp))
