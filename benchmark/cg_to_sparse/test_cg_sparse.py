from cgsolver import CGPoissonSolver
from bicgsolver import BICGPoissonSolver
from spsolver import SparsePoissonSolver
import taichi as ti
import numpy as np
import time

ti.init(arch=ti.x64)

# Solve a 128x128 2-D Poisson Equation
# There are 128^2 unknowns in total
psize = 128

# Solve in Taichi using custom CG
# A is implicitly represented in compute_Ap()
now = time.time()
print('>>> Solving using CGPoissonSolver...')
cgsolver = CGPoissonSolver(psize, 1e-16, quiet=True) # quiet=False to print residual
cgsolver.solve()
print('>>> Time spent using CGPoissonSolver:', time.time() - now, 'sec')

# Solve in Taichi using custom BICG
now = time.time()
print('>>> Solving using BICGPoissonSolver...')
bicgsolver = BICGPoissonSolver(psize, 1e-16, quiet=True)
bicgsolver.solve()
print('>>> Time spent using BICGPoissonSolver:', time.time() - now, 'sec')

# Solve in Taichi using SparseMatrixSolver
now = time.time()
print('>>> Solving using SparseSolver...')
spsolver = SparsePoissonSolver(psize, 'LU')
spsolver.solve()
print('>>> Time spent using SparseMatrixSolver:', time.time() - now, 'sec')

# Compare the residuals: norm(r) where r = Ax - b
print('>>> Comparing the residual norm(Ax-b)...')
print('>>> Residual CGPoissonSolver:', cgsolver.check_solution())
print('>>> Residual BICGPoissonSolver:', bicgsolver.check_solution())
print('>>> Residual SparsePoissonSolver:', spsolver.check_solution())
