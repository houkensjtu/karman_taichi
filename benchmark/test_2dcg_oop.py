from cgsolver import CGSolver
import taichi as ti

ti.init(arch=ti.cpu, kernel_profiler=True)

solver = CGSolver(256, 1e-20)
solver.solve()
ti.profiler.print_kernel_profiler_info()
