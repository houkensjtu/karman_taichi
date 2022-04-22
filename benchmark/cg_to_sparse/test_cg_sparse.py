from cgsolver import CGSolver
import taichi as ti
import numpy as np
import time

ti.init(arch=ti.x64, kernel_profiler=True)

# Solve a 128x128 2-D Poisson Equation
# There are 128^2 unknowns in total

# Solve in Taichi using custom CG
# A is implicitly represented in compute_Ap()
now = time.time()
print('Solving using CGSolver...')
solver = CGSolver(128, 1e-16, quiet=False) # quiet=False to print residual
solver.solve()
ti.profiler.print_kernel_profiler_info()
print('>>> Time spent using CG:', time.time() - now, 'sec')

# Solve in Taichi using SparseMatrixSolver
now = time.time()
print('Building sparse matrix...')
asp = solver.build_ASparse()
b = solver.build_b()
print('Solving using SparseSolver...')
spsolver = ti.linalg.SparseSolver(solver_type='LU')
spsolver.analyze_pattern(asp)
spsolver.factorize(asp)
xsp = spsolver.solve(b)
isSuccess = spsolver.info()
print('Sparse solver succeded?', isSuccess)
print('>>> Time spent using SparseMatrixSolver:', time.time() - now, 'sec')

# Compare the results
print('Fetching results...') # Building matrix in Numpy is very slow...
xti = solver.build_x()       # Fetch solution from CG solver
a = solver.build_A()         # Build the lfs of the equation
print('Comparing the results...')
print('Residual cg:', np.linalg.norm(b - np.dot(a, xti)))
print('Residual sp:', np.linalg.norm(b - np.dot(a, xsp)))
print('np.linalg.norm(xsp-xti)=', np.linalg.norm(xti-xsp))
