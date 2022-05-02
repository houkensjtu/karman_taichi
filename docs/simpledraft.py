# 2022.5.2
# This file is a quick sketch of what I want to implement in the SIMPLE Solver.
# Development steps:
# 1. Setup u, v, p; test GUI visualization utilities
# 2. Solve momentum equation based on p
# 3. Solve p correction
# 4. Add obstacle

import taichi as ti

ti.init(arch=ti.cpu, default_fp=ti.f64)

@ti.data_oriented
class SimpleSolver:
    def __init__(self, nx, ny):
        self.u  = ti.field(dtype=ti.f64, shape=(nx, ny))
        self.v  = ti.field(dtype=ti.f64, shape=(nx, ny))
        self.p  = ti.field(dtype=ti.f64, shape=(nx, ny))
        self.ct = ti.field(dtype=ti.f64, shape=(nx, ny))   # Cell type

        self.coef_u = ti.field(dtype=ti.f64, shape=(nx, ny, 5))
        self.b_u    = ti.field(dtype=ti.f64, shape=(nx, ny, 5))           
        self.coef_v = ti.field(dtype=ti.f64, shape=(nx, ny, 5))
        self.b_v    = ti.field(dtype=ti.f64, shape=(nx, ny, 5))           
        self.coef_p = ti.field(dtype=ti.f64, shape=(nx, ny, 5))

        self.au = ti.field(dtype=ti.f64, shape=(nx, ny, 5)) # Matrix-vector product
        self.av = ti.field(dtype=ti.f64, shape=(nx, ny, 5))
        self.ap = ti.field(dtype=ti.f64, shape=(nx, ny, 5))

    @ti.kernel
    def init(self):
        for i,j in u:
            ...

    @ti.kernel
    def compute_coef_u(self):
        for i,j in coef_u:
            self.coef_u[i,j,1] = self.u[i,j]*dy...
            self.coef_u[i,j,2] = self.u[i,j]*dy...
            ...
            
    def solve_momentum(self): # No inner iteration needed; one-step solution
        self.compute_moment_coef_u() # Use u and v to compute coef_u
        self.compute_moment_coef_v()
        self.compute_moment_b_u()
        self.compute_moment_b_v()                
        u_solver = usolver(self.u, self.coef_u, self.b_u) # coef_u will be used to compute_matvec
        usolver.solve() # => u*
        ...
            
    def solve_pcorrection(self):  # Solving p-correction is challenging for bicgstab because A is not full rank
        ...
        
@ti.data_oriented
class CG:
    def __init__(self, m, n):
        self.nx = m
        self.ny = n
        
    @ti.kernel
    def init(self):
        pass
    
    @ti.kernel
    def compute_matvec():
        pass

    @ti.kernel
    def compute_rhs():
        pass
    
    def solve():
        pass

@ti.data_oriented    
class usolver(CG):
    def __init__(self, u, v, p):
        super().__init__()
        self.u = u
        self.v = v
        self.p = p
        self.coef_u =

    @ti.kernel
    def compute_matvec(au:ti.template(), u:ti.template(), coef_u:ti.template()):
        self.au[i,j] = coef_u[i,j,1] * u[i,j] + coef_u[i,j,2] * u[i-1, j] + ...
    
    @ti.kernel
    def compute_rhs(bu:ti.template(), p:ti.template()):
        self.bu[i,j] = (p[i,j] - p[i-1,j]) * ...
    
