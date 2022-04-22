from cgsolver import CGSolver
import taichi as ti
import numpy as np
import time

ti.init(arch=ti.x64, kernel_profiler=True)

# Solve in Taichi using custom CG
now = time.time()
print('Solving using CGSolver...')
solver = CGSolver(128, 1e-30, False)
solver.solve()
ti.profiler.print_kernel_profiler_info()
print(time.time() - now, 'sec')

# Solve in Taichi using SparseMatrixSolver
now = time.time()
print('Building matrix in Numpy...')
asp = solver.build_ASparse()
b = solver.build_b()
print('Solving using SparseSolver...')
spsolver = ti.linalg.SparseSolver(solver_type='LLT')
spsolver.analyze_pattern(asp)
spsolver.factorize(asp)
xsp = spsolver.solve(b)
isSuccess = spsolver.info()
print('Sparse solver succeded?', isSuccess)
print(time.time() - now, 'sec')

# Compare the results
xti = solver.build_x() # Fetch solution from CG solver
a = solver.build_A()
print('Type xti', type(xti), xti.shape)
print('Type xsp', type(xsp), xsp.shape)
print('Type a', type(a), a.shape)
print('Type asp', type(asp))
print('Type b', type(b), b.shape)

print('Residual cg:', np.linalg.norm(b - np.dot(a, xti)))
print('Residual sp:', np.linalg.norm(b - np.dot(a, xsp)))
print(np.linalg.norm(xti-xsp))
