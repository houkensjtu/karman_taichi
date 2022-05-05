import taichi as ti
import numpy as np
from display import Display

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
        self.dt = 0.00001
        
        self.real = ti.f64
        
        self.u  = ti.field(dtype=self.real, shape=(nx+3, ny+2))
        self.v  = ti.field(dtype=self.real, shape=(nx+2, ny+3))
        self.p  = ti.field(dtype=self.real, shape=(nx+2, ny+2))
        self.bc = {'w': [0.0, 0.0], 'e': [0.0, 0.0], 'n': [0.0, 0.0], 's': [0.0, 0.0] }
        self.ct = ti.field(dtype=self.real, shape=(nx+2, ny+2))   # Cell type

        self.coef_u = ti.field(dtype=self.real, shape=(nx+3, ny+2, 5))
        self.b_u    = ti.field(dtype=self.real, shape=(nx+3, ny+2, 5))           
        self.coef_v = ti.field(dtype=self.real, shape=(nx+2, ny+3, 5))
        self.b_v    = ti.field(dtype=self.real, shape=(nx+2, ny+3, 5))           
        self.coef_p = ti.field(dtype=self.real, shape=(nx+2, ny+2, 5))
        self.b_p    = ti.field(dtype=self.real, shape=(nx+2, ny+2, 5))        
        
        self.disp = Display(self)

    def dump_matrix(self, step): # Save u,v,p at step to csv files
        for k,v in {'u':self.u, 'v':self.v, 'p':self.p}.items():
            np.savetxt(f'log/{k}-{step:06}.csv', v.to_numpy(), delimiter=',')

    @ti.kernel
    def compute_coef_u(self):
        u, v, dx, dy, rho, mu = self.u, self.v, self.dx, self.dy, self.rho, self.mu
        for i,j in ti.ndrange((1,nx+2), (1,ny+1)):
            coef_u[i,j,1] =  -mu * dy / dx - rho * 0.5 * (u[i,j] + u[i-1,j]) * dy      # aw
            coef_u[i,j,2] =  -mu * dy / dx + rho * 0.5 * (u[i, j] + u[i + 1, j]) * dy  # ae
            coef_u[i,j,3] =  -mu * dx / dy + rho * 0.5 * (v[i-1,j] + v[i,j]) * dx      # an
            coef_u[i,j,4] =  -mu * dx / dy - rho * 0.5 * (v[i-1,j+1] + v[i,j+1]) * dx  # as
            coef_u[i,j,0] =  coef_u[i,j,1] + coef_u[i,j,2] + coef_u[i,j,3] + coef_u[i,j,4] +\
                             rho * dx * dy / dt                                        # ap
        
    @ti.kernel
    def compute_coef_v(self):
        pass

    @ti.kernel
    def compute_coef_p(self):
        pass

    @ti.kernel
    def set_bc(self):
        pass
        
    @ti.kernel
    def init(self):
        for i,j in self.u:
            self.u[i,j] = - i
        for i,j in self.v:
            self.v[i,j] = - j 
        for i,j in self.p:
            self.p[i,j] = - i * j

    def solve(self):
        self.init()
        self.set_bc()
        for step in range(3):
            self.disp.display(f'log/{step:06}.png')
            self.dump_matrix(step)

# Lid-driven Cavity            
ssolver = SIMPLESolver(1.0, 1.0, 128, 128) # lx, ly, nx, ny
ssolver.bc['n'][1] = 1.0
ssolver.solve()
