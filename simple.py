import taichi as ti
import numpy as np
from display import Display
from cgsolver import CGSolver
from bicgsolver import BICGSolver

ti.init(arch=ti.cpu, default_fp=ti.f64)

@ti.data_oriented
class SIMPLESolver:
    def __init__(self, lx, ly, nx, ny):
        self.lx = lx
        self.ly = ly
        self.nx = nx
        self.ny = ny
        self.dx = self.lx / self.nx
        self.dy = self.ly / self.ny
        self.rho= 1.00
        self.mu = 0.01
        self.dt = 10000
        self.real = ti.f64
        
        self.u  = ti.field(dtype=self.real, shape=(nx+3, ny+2))
        self.v  = ti.field(dtype=self.real, shape=(nx+2, ny+3))
        self.u0 = ti.field(dtype=self.real, shape=(nx+3, ny+2)) # Previous time step
        self.v0 = ti.field(dtype=self.real, shape=(nx+2, ny+3))
        
        self.p  = ti.field(dtype=self.real, shape=(nx+2, ny+2))
        self.bc = {'w': [1.0, 0.0], 'e': [1.0, 0.0], 'n': [1.0, 0.0], 's': [1.0, 0.0] }
        self.ct = ti.field(dtype=self.real, shape=(nx+2, ny+2))   # Cell type

        self.coef_u = ti.field(dtype=self.real, shape=(nx+3, ny+2, 5))
        self.b_u    = ti.field(dtype=self.real, shape=(nx+3, ny+2))           
        self.coef_v = ti.field(dtype=self.real, shape=(nx+2, ny+3, 5))
        self.b_v    = ti.field(dtype=self.real, shape=(nx+2, ny+3))           
        self.coef_p = ti.field(dtype=self.real, shape=(nx+2, ny+2, 5))
        self.b_p    = ti.field(dtype=self.real, shape=(nx+2, ny+2))        
        
        self.disp = Display(self)

    def dump_matrix(self, step): # Save u,v,p at step to csv files
        for name,val in {'u':self.u, 'v':self.v, 'p':self.p, 'bu':self.b_u}.items():
            np.savetxt(f'log/{name}-{step:06}.csv', val.to_numpy(), delimiter=',')

    @ti.kernel
    def compute_coef_u(self):
        nx, ny, dx, dy, dt, rho, mu = self.nx, self.ny, self.dx, self.dy, self.dt, self.rho, self.mu
        for i,j in ti.ndrange((2,nx+1), (1,ny+1)):
            self.coef_u[i,j,1] =  -mu * dy / dx - rho * 0.5 * (self.u[i,j] + self.u[i-1,j]) * dy      # aw
            self.coef_u[i,j,2] =  -mu * dy / dx + rho * 0.5 * (self.u[i,j] + self.u[i+1,j]) * dy      # ae
            self.coef_u[i,j,3] =  -mu * dx / dy + rho * 0.5 * (self.v[i-1,j] + self.v[i,j]) * dx      # an
            self.coef_u[i,j,4] =  -mu * dx / dy - rho * 0.5 * (self.v[i-1,j+1] + self.v[i,j+1]) * dx  # as
            self.coef_u[i,j,0] =  -(self.coef_u[i,j,1] + self.coef_u[i,j,2] + self.coef_u[i,j,3] +\
                                  self.coef_u[i,j,4]) + rho * dx * dy / dt                            # ap
            self.b_u[i,j] = (self.p[i-1,j] - self.p[i,j]) * dy + rho * dx * dy / dt * self.u0[i, j]   # rhs
            # Verified with the following input, results close enough to the original bicg solver.
            # xl = (i - 1) / (self.nx + 1)
            # yl = (j - 1) / self.ny
            # self.b_u[i,j] = ti.sin(2.0 * np.pi * xl) * ti.sin(2.0 * np.pi * yl)
            
    @ti.kernel
    def compute_coef_v(self):
        nx, ny, dx, dy, dt, rho, mu = self.nx, self.ny, self.dx, self.dy, self.dt, self.rho, self.mu
        for i,j in ti.ndrange((1,nx+1),(1,ny+2)):
            self.coef_v[i,j,1] = -mu * dy / dx - rho * 0.5 * (self.u[i,j] + self.u[i,j-1]) * dy       # aw
            self.coef_v[i,j,2] = -mu * dy / dx + rho * 0.5 * (self.u[i+1,j-1] + self.u[i+1,j]) * dy   # ae            
            self.coef_v[i,j,3] = -mu * dx / dy + rho * 0.5 * (self.v[i,j-1] + self.v[i,j]) * dx       # an
            self.coef_v[i,j,4] = -mu * dx / dy - rho * 0.5 * (self.v[i,j+1] + self.v[i,j]) * dx       # as
            self.coef_v[i,j,0] = -(self.coef_v[i,j,1] + self.coef_v[i,j,2] + self.coef_v[i,j,3] +\
                                 self.coef_v[i,j,4]) + rho * dx * dy / dt                             # ap
            self.b_v[i,j] = (self.p[i,j] - self.p[i,j-1]) * dx + rho * dx * dy / dt * self.v0[i, j]   # rhs
            
    @ti.kernel
    def compute_coef_p(self):
        pass

    @ti.kernel
    def set_bc(self):
        nx, ny, bc = self.nx, self.ny, self.bc
        for j in range(1,ny+1):
            # u bc for w
            self.b_u[2,j] += - self.coef_u[2,j,1] * bc['w'][0]
            self.coef_u[2,j,1] = 0.0
            self.u[1,j] = bc['w'][0]
            # u bc for e
            self.b_u[nx,j] += - self.coef_u[nx,j,2] * bc['e'][0]            
            self.coef_u[nx,j,2] = 0.0
            self.u[nx+1,j] = bc['e'][0]
            
        for i in range(1,nx+1):
            # v bc for s
            self.b_v[i,2] += - self.coef_v[i,2,4] * bc['s'][0]            
            self.coef_v[i,2,4] = 0.0            
            # v bc for n
            self.b_v[i,ny] += - self.coef_v[i,ny,3] * bc['n'][0]
            self.coef_v[i,ny,3] = 0.0

            
    def solve_momentum_eqn(self):
        self.u0 = self.u
        self.v0 = self.v
        self.compute_coef_u()
        self.compute_coef_v()
        self.set_bc()
        u_momentum_solver = BICGSolver(self.coef_u, self.b_u)
        #u_momentum_solver = CGSolver(self.coef_u, self.b_u)        
        v_momentum_solver = BICGSolver(self.coef_v, self.b_v)
        #v_momentum_solver = CGSolver(self.coef_v, self.b_v)        
        
        u_momentum_solver.solve(eps=1e-6, quiet=False)
        v_momentum_solver.solve(eps=1e-6, quiet=False)

        self.u = u_momentum_solver.x # Problematic?
        self.v = v_momentum_solver.x
        
    def solve_pcorrection_eqn(self):
        pass
        
    @ti.kernel
    def init(self):
        for i,j in self.u:
            self.u[i,j] = 0.0
        for i,j in self.v:
            self.v[i,j] = 0.0 
        for i,j in self.p:
            self.p[i,j] = 0.0 

    def solve(self):
        self.init()
        for step in range(1):
            self.solve_momentum_eqn()
            self.solve_pcorrection_eqn()
            self.disp.display(f'log/{step:06}.png')
            self.dump_matrix(step)

# Lid-driven Cavity            
ssolver = SIMPLESolver(1.0, 1.0, 128, 128) # lx, ly, nx, ny
ssolver.bc['w'][0] = 1.0
ssolver.bc['e'][0] = 1.0
ssolver.bc['n'][0] = 1.0
ssolver.bc['s'][0] = 1.0
ssolver.solve()
