import taichi as ti
import numpy as np
from cgsolver import CGPoissonSolver

@ti.data_oriented
class BICGPoissonSolver(CGPoissonSolver):
    def __init__(self, n, eps, quiet):
        super().__init__(n, eps, quiet)
