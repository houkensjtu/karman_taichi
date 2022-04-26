import taichi as ti
from cgsolver import CGPoissonSolver

ti.init(arch=ti.x64)
psize = 32
cgsolver = CGPoissonSolver(psize, 1e-16, 0.1, quiet=True) # quiet=False to print residual
cgsolver.solve()
cgsolver.build_A()
cgsolver.build_b()

