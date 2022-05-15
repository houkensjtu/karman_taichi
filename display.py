import taichi as ti

@ti.data_oriented
class Display:
    def __init__(self, SIMPLESolver, *args):
        self.solver = SIMPLESolver
        self.nx = self.solver.nx
        self.ny = self.solver.ny
        self.real = self.solver.real
        self.udisp = ti.field(dtype=self.solver.real, shape=((self.nx+2), (self.ny+2)))
        self.vdisp = ti.field(dtype=self.solver.real, shape=((self.nx+2), (self.ny+2)))
        self.pdisp = ti.field(dtype=self.solver.real, shape=((self.nx+2), (self.ny+2)))
        self.pcordisp = ti.field(dtype=self.solver.real, shape=((self.nx+2), (self.ny+2)))
        self.mdivdisp = ti.field(dtype=self.solver.real, shape=((self.nx+2), (self.ny+2)))                

    @ti.func
    def scale_field(self, f):
        f_max = -1.0e9
        f_min = 1.0e9
        for i,j in f:
            if f[i,j] > f_max:
                f_max = f[i,j]
            if f[i,j] < f_min:
                f_min = f[i,j]
        for i,j in f:
            f[i,j] = (f[i,j] - f_min) / (f_max - f_min + 1.0e-9)
        
    @ti.kernel
    def post_process_field(self):
        for i,j in ti.ndrange(self.nx+2, self.ny+2):
            self.udisp[i,j] = 0.5 * (self.solver.u[i,j] + self.solver.u[i+1,j])
            self.vdisp[i,j] = 0.5 * (self.solver.v[i,j] + self.solver.v[i,j+1])
            self.pdisp[i,j] = self.solver.p[i,j]
            self.pcordisp[i,j] = self.solver.pcor[i,j]
            self.mdivdisp[i,j] = self.solver.mdiv[i,j]            
        self.scale_field(self.udisp)
        self.scale_field(self.vdisp)
        self.scale_field(self.pdisp)
        self.scale_field(self.pcordisp)
        self.scale_field(self.mdivdisp)        
            
    def display(self, filename):
        import numpy as np        
        self.post_process_field()
        gui = ti.GUI("SIMPLESolver", ((self.nx+2),5*(self.ny+2)), show_gui=False)
        img = np.concatenate((self.udisp.to_numpy(), self.vdisp.to_numpy(), self.pdisp.to_numpy(), \
                              self.pcordisp.to_numpy(), self.mdivdisp.to_numpy()), axis=1)
        gui.set_image(img)
        gui.show(filename)

