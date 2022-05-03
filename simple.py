import taichi as ti
import numpy as np
from display import Display

ti.init(arch=ti.cpu, default_fp=ti.f64)

@ti.data_oriented
class SIMPLESolver:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.real = ti.f64
        self.u  = ti.field(dtype=self.real, shape=(nx+3, ny+2))
        self.v  = ti.field(dtype=self.real, shape=(nx+2, ny+3))
        self.p  = ti.field(dtype=self.real, shape=(nx+2, ny+2))
        self.ct = ti.field(dtype=self.real, shape=(nx+2, ny+2))   # Cell type
        self.disp = Display(self)


    def dump_matrix(self, step):
        for k,v in {'u':self.u, 'v':self.v, 'p':self.p}.items():
            np.savetxt(f'log/{k}-{step:06}.csv', v.to_numpy(), delimiter=',')

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
        for step in range(3):
            self.disp.display(f'log/{step:06}.png')
            self.dump_matrix(step)

ssolver = SIMPLESolver(128, 128)
ssolver.solve()
