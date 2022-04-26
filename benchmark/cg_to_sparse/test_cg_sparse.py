from cgsolver import CGPoissonSolver, CGPoissonSolverYM
from spsolver import SparsePoissonSolver
import taichi as ti
import numpy as np
import time

ti.init(arch=ti.x64)

# Solve a 128x128 2-D Poisson Equation
# There are 128^2 unknowns in total
psize = 256

# Solve in Taichi using custom CG
# A is implicitly represented in compute_Ap()
now = time.time()
print('Solving using CGPoissonSolver...')
solver = CGPoissonSolver(psize, 1e-16, quiet=True) # quiet=False to print residual
solver.solve()
print('>>> Time spent using CG:', time.time() - now, 'sec')

# Solve in Taichi using custom CG (YM version with global sum[None])
now = time.time()
print('Solving using CGPoissonSolverYM...')
ymsolver = CGPoissonSolverYM(psize, 1e-16, quiet=True) # quiet=False to print residual
ymsolver.solve()
print('>>> Time spent using CGYM:', time.time() - now, 'sec')

# Solve in Taichi using SparseMatrixSolver
now = time.time()
# print('Solving using SparseSolver...')
spsolver = SparsePoissonSolver(psize, 'LU')
xsp = spsolver.solve()
print('>>> Time spent using SparseMatrixSolver:', time.time() - now, 'sec')

# Compare the residuals: norm(r) where r = Ax - b
print('Fetching results...') # Building matrix in Numpy is very slow...
xti = solver.build_x()       # Fetch solution from CG solver
a = solver.build_A()         # Build the lfs of the equation
b = solver.build_b()         # Build the rfs of the equation
print('Comparing the results...')
print('Residual cg (using Numpy norm):', np.linalg.norm(b - np.dot(a, xti)))
print('Residual cg (using Taichi internal check):', solver.check_solution())
print('Residual sp:', np.linalg.norm(b - np.dot(a, xsp)))
print('np.linalg.norm(xsp-xti)=', np.linalg.norm(xti-xsp))
