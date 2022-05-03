import taichi as ti

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
        self.disp = ti.field(dtype=self.real, shape=(3*(nx+2), 3*(ny+2)))

    def display(self):
        print('Displaying the flow field...')

ssolver = SIMPLESolver(64, 320)
ssolver.display()
