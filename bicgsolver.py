import taichi as ti
import numpy as np
from cgsolver import CGSolver

@ti.data_oriented
class BICGSolver(CGSolver):
    def __init__(self, coef, b, x):
        super().__init__(coef, b, x)
        self.r_tld = ti.field(dtype=self.real)
        self.p_hat = ti.field(dtype=self.real)
        self.s = ti.field(dtype=self.real)
        self.s_hat = ti.field(dtype=self.real)
        self.t = ti.field(dtype=self.real)
        self.Ashat = ti.field(dtype=self.real)        
        ti.root.dense(ti.ij, (self.NX, self.NY)).place(self.r_tld, self.p_hat, self.s, self.t, self.s_hat, self.Ashat)
        self.omega = ti.field(dtype=self.real)
        self.rho = ti.field(dtype=self.real)
        self.rho_1 = ti.field(dtype=self.real)
        ti.root.place(self.omega, self.rho, self.rho_1)

    @ti.kernel
    def init(self):
        for i, j in ti.ndrange((self.N_ext, self.NX-self.N_ext), (self.N_ext, self.NY-self.N_ext)):
            # r[0] = b - Ax, where x = 0; therefore r[0] = b
            self.r[i, j] = self.b[i,j]
            self.r_tld[i, j] = self.b[i,j]
            self.Ap[i, j] = 0.0
            self.Ax[i, j] = 0.0
            self.Ashat[i, j] = 0.0            
            self.p[i, j] = 0.0
            self.x[i, j] = 0.0            
            self.omega[None] = 1.0
            self.alpha[None] = 1.0
            self.beta[None] = 1.0
            self.rho_1[None] = 1.0
            self.rho[None] = 0.0

    @ti.kernel
    def copy(self, orig:ti.template(), dest:ti.template()):
        for I in ti.grouped(orig):
            dest[I] = orig[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            self.p[I] = self.r[I] + self.beta[None]*(self.p[I] - self.omega[None] * self.Ap[I])

    @ti.kernel
    def update_phat(self):
        for I in ti.grouped(self.p_hat):
            # self.p_hat[I] = 1.0 / self.coef[I,0] * self.p[I]
            self.p_hat[I] = self.p[I]            

    @ti.kernel
    def update_shat(self):
        for I in ti.grouped(self.s_hat):
            # self.s_hat[I] = 1.0 / self.coef[I,0] * self.s[I]
            self.s_hat[I] = self.s[I]            
            
    @ti.kernel
    def update_s(self):
        for I in ti.grouped(self.s):
            self.s[I] = self.r[I] - self.alpha[None] * self.Ap[I]

    @ti.kernel
    def compute_As(self):
        for i, j in ti.ndrange((self.N_ext, self.NX-self.N_ext), (self.N_ext, self.NY-self.N_ext)):
            self.Ashat[i,j] = self.coef[i,j,0] * self.s_hat[i,j] + self.coef[i,j,1] * self.s_hat[i-1,j] + self.coef[i,j,2] * self.s_hat[i+1,j] +\
                              self.coef[i,j,3] * self.s_hat[i,j+1] + self.coef[i,j,4] * self.s_hat[i,j-1]
            

    @ti.kernel
    def compute_Ap(self):
        for i, j in ti.ndrange((self.N_ext, self.NX-self.N_ext), (self.N_ext, self.NY-self.N_ext)):
            self.Ap[i,j] = self.coef[i,j,0] * self.p_hat[i,j] + self.coef[i,j,1] * self.p_hat[i-1,j] + self.coef[i,j,2] * self.p_hat[i+1,j] +\
                           self.coef[i,j,3] * self.p_hat[i,j+1] + self.coef[i,j,4] * self.p_hat[i,j-1]
        
    @ti.kernel
    def update_x(self):
        for I in ti.grouped(self.x):
            self.x[I] += self.alpha[None] * self.p_hat[I] + self.omega[None] * self.s_hat[I]
    
    @ti.kernel
    def update_r(self):
        for I in ti.grouped(self.r):
            self.r[I] = self.s[I] - self.omega[None] * self.t[I]
        
    def solve(self, eps=1e-8, quiet=True):
        self.init()
        initial_rTr = self.reduce(self.r, self.r)
        if not quiet:
            print('>>> Initial residual =', ti.sqrt(initial_rTr))
        # self.history.append(f'{ti.sqrt(initial_rTr):e}\n')
        for i in range(self.steps):
            self.rho[None] = self.reduce(self.r, self.r_tld)
            if self.rho[None] == 0.0:
                print('>>> BICG failed at first place...')
                break
            if i == 0:
                self.copy(self.r, self.p)
            else:
                self.beta[None] = (self.rho[None] / self.rho_1[None]) * (self.alpha[None]/self.omega[None])
                self.update_p()
            self.update_phat()
            
            self.compute_Ap()
            alpha_lower = self.reduce(self.r_tld, self.Ap)
            self.alpha[None] = self.rho[None] / alpha_lower

            self.update_s()
            self.update_shat()

            self.compute_As()
            self.copy(self.Ashat, self.t)

            omega_upper = self.reduce(self.t, self.s)
            omega_lower = self.reduce(self.t, self.t)
            self.omega[None] = omega_upper / omega_lower

            self.update_x()
            self.update_r()

            rTr = self.reduce(self.r, self.r)
            
            #self.history.append(f'{ti.sqrt(rTr):e}\n') # Write converge history; i+1 because starting from 1.
            
            if not quiet:
                print('>>> Iter =', i+1, ' Residual =', ti.sqrt(rTr))

            if ti.sqrt(rTr / initial_rTr) < eps:
                if not quiet:
                    print('>>> BICG Converged...')
                break

            self.rho_1[None] = self.rho[None]
        
        
        
