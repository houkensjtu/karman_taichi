import taichi as ti
import numpy as np
from display import Display
from cgsolver import CGSolver
from bicgsolver import BICGSolver
import time


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
        self.dt = 100000
        self.real = ti.f64
        
        self.alpha_u = 1.0 # Used in Jacobian computation
        
        self.u  = ti.field(dtype=self.real, shape=(nx+3, ny+2))
        self.v  = ti.field(dtype=self.real, shape=(nx+2, ny+3))
        
        self.u_mid  = ti.field(dtype=self.real, shape=(nx+3, ny+2))
        self.v_mid  = ti.field(dtype=self.real, shape=(nx+2, ny+3))
        
        self.u0 = ti.field(dtype=self.real, shape=(nx+3, ny+2)) # Previous time step
        self.v0 = ti.field(dtype=self.real, shape=(nx+2, ny+3))
        
        self.p  = ti.field(dtype=self.real, shape=(nx+2, ny+2))
        self.pcor = ti.field(dtype=self.real, shape=(nx+2, ny+2))
        self.pcor_mid = ti.field(dtype=self.real, shape=(nx+2, ny+2))
        
        self.mdiv = ti.field(dtype=self.real, shape=(nx+2, ny+2))                
        self.bc = {'w': [0.0, 0.0], 'e': [0.0, 0.0], 'n': [0.0, 0.0], 's': [0.0, 0.0] }
        self.ct = ti.field(dtype=self.real, shape=(nx+2, ny+2))   # Cell type

        self.coef_u = ti.field(dtype=self.real, shape=(nx+3, ny+2, 5))
        self.b_u    = ti.field(dtype=self.real, shape=(nx+3, ny+2))           
        self.coef_v = ti.field(dtype=self.real, shape=(nx+2, ny+3, 5))
        self.b_v    = ti.field(dtype=self.real, shape=(nx+2, ny+3))           
        self.coef_p = ti.field(dtype=self.real, shape=(nx+2, ny+2, 5))
        self.b_p    = ti.field(dtype=self.real, shape=(nx+2, ny+2))        
        
        self.disp = Display(self)

    def dump_matrix(self, step, msg): # Save u,v,p at step to csv files
        for name,val in {'u':self.u, 'v':self.v, 'p':self.p, 'mdiv':self.mdiv, 'pcor':self.pcor}.items():
            np.savetxt(f'log/{step:06}-{name}-{msg}.csv', val.to_numpy(), delimiter=',')

    def dump_coef(self, step, msg):
        np.savetxt(f'log/{step:06}-apu-{msg}.csv', self.coef_u.to_numpy()[:,:,0], delimiter=',')
        np.savetxt(f'log/{step:06}-awu-{msg}.csv', self.coef_u.to_numpy()[:,:,1], delimiter=',')
        np.savetxt(f'log/{step:06}-aeu-{msg}.csv', self.coef_u.to_numpy()[:,:,2], delimiter=',')
        np.savetxt(f'log/{step:06}-anu-{msg}.csv', self.coef_u.to_numpy()[:,:,3], delimiter=',')
        np.savetxt(f'log/{step:06}-asu-{msg}.csv', self.coef_u.to_numpy()[:,:,4], delimiter=',')
        np.savetxt(f'log/{step:06}-bu -{msg}.csv', self.b_u.to_numpy(),           delimiter=',')
        
        np.savetxt(f'log/{step:06}-apv-{msg}.csv', self.coef_v.to_numpy()[:,:,0], delimiter=',')
        np.savetxt(f'log/{step:06}-awv-{msg}.csv', self.coef_v.to_numpy()[:,:,1], delimiter=',')
        np.savetxt(f'log/{step:06}-aev-{msg}.csv', self.coef_v.to_numpy()[:,:,2], delimiter=',')
        np.savetxt(f'log/{step:06}-anv-{msg}.csv', self.coef_v.to_numpy()[:,:,3], delimiter=',')
        np.savetxt(f'log/{step:06}-asv-{msg}.csv', self.coef_v.to_numpy()[:,:,4], delimiter=',')        
        np.savetxt(f'log/{step:06}-bv -{msg}.csv', self.b_v.to_numpy(),           delimiter=',')

        np.savetxt(f'log/{step:06}-app-{msg}.csv', self.coef_p.to_numpy()[:,:,0], delimiter=',')
        np.savetxt(f'log/{step:06}-awp-{msg}.csv', self.coef_p.to_numpy()[:,:,1], delimiter=',')
        np.savetxt(f'log/{step:06}-aep-{msg}.csv', self.coef_p.to_numpy()[:,:,2], delimiter=',')
        np.savetxt(f'log/{step:06}-anp-{msg}.csv', self.coef_p.to_numpy()[:,:,3], delimiter=',')
        np.savetxt(f'log/{step:06}-asp-{msg}.csv', self.coef_p.to_numpy()[:,:,4], delimiter=',')        
        np.savetxt(f'log/{step:06}-bp -{msg}.csv', self.b_p.to_numpy(),           delimiter=',')        
        

    @ti.kernel
    def compute_coef_u(self):
        nx, ny, dx, dy, dt, rho, mu = self.nx, self.ny, self.dx, self.dy, self.dt, self.rho, self.mu
        for i,j in ti.ndrange((2,nx+1), (1,ny+1)):
            self.coef_u[i,j,1] =  -(mu * dy / dx + 0.5 * rho * 0.5 * (self.u[i,j] + self.u[i-1,j]) * dy) # aw
            self.coef_u[i,j,2] =  -(mu * dy / dx - 0.5 * rho * 0.5 * (self.u[i,j] + self.u[i+1,j]) * dy) # ae
            self.coef_u[i,j,3] =  -(mu * dx / dy - 0.5 * rho * 0.5 * (self.v[i-1,j+1] + self.v[i,j+1]) * dx)# an
            self.coef_u[i,j,4] =  -(mu * dx / dy + 0.5 * rho * 0.5 * (self.v[i-1,j] + self.v[i,j]) * dx) # as
            self.coef_u[i,j,0] =  -(self.coef_u[i,j,1] + self.coef_u[i,j,2] + self.coef_u[i,j,3] +\
                                    self.coef_u[i,j,4]) +\
                                    rho * 0.5 * (self.u[i,j] + self.u[i+1,j]) * dy -\
                                    rho * 0.5 * (self.u[i,j] + self.u[i-1,j]) * dy +\
                                    rho * 0.5 * (self.v[i-1,j+1] + self.v[i,j+1]) * dx -\
                                    rho * 0.5 * (self.v[i-1,j] + self.v[i,j]) * dx +\
                                    rho * dx * dy / dt                            # ap
            self.b_u[i,j] = (self.p[i-1,j] - self.p[i,j]) * dy + rho * dx * dy / dt * self.u0[i, j]   # rhs
        
            
    @ti.kernel
    def compute_coef_v(self):
        nx, ny, dx, dy, dt, rho, mu = self.nx, self.ny, self.dx, self.dy, self.dt, self.rho, self.mu
        for i,j in ti.ndrange((1,nx+1),(2,ny+1)):
            self.coef_v[i,j,1] = -(mu * dy / dx + 0.5 * rho * 0.5 * (self.u[i,j] + self.u[i,j-1]) * dy)       # aw
            self.coef_v[i,j,2] = -(mu * dy / dx - 0.5 * rho * 0.5 * (self.u[i+1,j-1] + self.u[i+1,j]) * dy)   # ae            
            self.coef_v[i,j,3] = -(mu * dx / dy - 0.5 * rho * 0.5 * (self.v[i,j+1] + self.v[i,j]) * dx)       # an
            self.coef_v[i,j,4] = -(mu * dx / dy + 0.5 * rho * 0.5 * (self.v[i,j-1] + self.v[i,j]) * dx)       # as
            self.coef_v[i,j,0] = -(self.coef_v[i,j,1] + self.coef_v[i,j,2] + self.coef_v[i,j,3] +\
                                 self.coef_v[i,j,4]) +\
                                 rho * 0.5 * (self.u[i+1,j-1] + self.u[i+1,j]) * dy -\
                                 rho * 0.5 * (self.u[i,j] + self.u[i,j-1]) * dy +\
                                 rho * 0.5 * (self.v[i,j+1] + self.v[i,j]) * dx -\
                                 rho * 0.5 * (self.v[i,j-1] + self.v[i,j]) * dx +\
                                 rho * dx * dy / dt                             # ap
            self.b_v[i,j] = (self.p[i,j-1] - self.p[i,j]) * dx + rho * dx * dy / dt * self.v0[i, j]   # rhs

            
    @ti.kernel
    def compute_coef_p(self):
        nx, ny, dx, dy, rho = self.nx, self.ny, self.dx, self.dy, self.rho
        for i,j in ti.ndrange((1,nx+1),(1,ny+1)):  # [1,nx], [1,ny]
            self.mdiv[i,j] = rho * (self.u[i,j] - self.u[i+1,j]) * dy + rho * (self.v[i,j] - self.v[i,j+1]) * dx
            self.b_p[i,j] = self.mdiv[i,j]
            self.coef_p[i,j,1] =  -rho * dy * dy / self.coef_u[i,j,0]                 # aw
            self.coef_p[i,j,2] =  -rho * dy * dy / self.coef_u[i+1,j,0]               # ae
            self.coef_p[i,j,3] =  -rho * dx * dx / (self.coef_v[i,j+1,0] + 1e-30)               # an
            self.coef_p[i,j,4] =  -rho * dx * dx / (self.coef_v[i,j,0] + 1e-30)                # as

            if i == 1:
                self.coef_p[i,j,1] = 0.0
            if i == nx:
                self.coef_p[i,j,2] = 0.0
            if j == 1:
                self.coef_p[i,j,4] = 0.0
            if j == ny:
                self.coef_p[i,j,3] = 0.0
                
            self.coef_p[i,j,0] = - (self.coef_p[i,j,1] + self.coef_p[i,j,2] + self.coef_p[i,j,3] + self.coef_p[i,j,4])
            
        self.coef_p[1,1,1] = 0.0
        self.coef_p[1,1,2] = 0.0
        self.coef_p[1,1,3] = 0.0
        self.coef_p[1,1,4] = 0.0


            
    @ti.kernel
    def set_bc(self):
        nx, ny, bc = self.nx, self.ny, self.bc
        # u - [nx+3, ny+2] - i E [0,nx+2], j E [0,ny+1]
        # v - [nx+2, ny+3] - i E [0,nx+1], j E [0,ny+2]
        for j in range(1,ny+1):
            # u bc for w
            self.b_u[2,j] += - self.coef_u[2,j,1] * bc['w'][0]       # b += aw * u_inlet
            self.coef_u[2,j,1] = 0.0                                 # aw = 0
            self.u[1,j] = bc['w'][0]                                 # u_inlet
            
            # u bc for e
            self.b_u[nx,j] += - self.coef_u[nx,j,2] * bc['e'][0]     # b += ae * u_outlet
            self.coef_u[nx,j,2] = 0.0                                # ae = 0
            self.u[nx+1,j] = bc['e'][0]                              # u_outlet
            
        for i in range(1,nx+1):
            # v bc for s
            self.b_v[i,2] += - self.coef_v[i,2,4] * bc['s'][0]       # b += as * v_inlet
            self.coef_v[i,2,4] = 0.0                                 # as = 0
            self.v[i,1] = bc['s'][0]                                 # v_inlet            

            # v bc for n
            self.b_v[i,ny] += - self.coef_v[i,ny,3] * bc['n'][0]     # b += an * v_outlet
            self.coef_v[i,ny,3] = 0.0                                # an = 0
            self.v[i,ny+1] = bc['n'][0]                              # v_outlet

        for i in range(2,nx+1):
            self.b_u[i,1] += 2 * self.mu * bc['s'][1] * self.dx / self.dy # South sliding wall
            self.coef_u[i,1,0] += (self.coef_u[i,1,4] + 2 * self.mu * self.dx / self.dy)
            self.coef_u[i,1,4] = 0.0
            # ap = ap - as + 2mudx/dy
            
            self.b_u[i,ny] += 2 * self.mu * bc['n'][1] * self.dx / self.dy # North sliding wall
            self.coef_u[i,ny,0] += (self.coef_u[i,ny,3] + 2 * self.mu * self.dx / self.dy)
            self.coef_u[i,ny,3] = 0.0            
            # ap = ap - an + 2mudx/dy

        for j in range(2,ny+1):
            self.b_v[1,j] += 2 * self.mu * bc['w'][1] * self.dy / self.dx  # West sliding wall
            self.coef_v[1,j,0] += (self.coef_v[1,j,1] + 2 * self.mu * self.dy / self.dx)
            self.coef_v[1,j,1] = 0.0

            self.b_v[nx,j] += 2 * self.mu * bc['e'][1] * self.dy / self.dx  # East sliding wall
            self.coef_v[nx,j,0] += (self.coef_v[nx,j,2] + 2 * self.mu * self.dy / self.dx)
            self.coef_v[nx,j,2] = 0.0

    @ti.kernel
    def jacobian_solve_u(self)->ti.f64:
        nx, ny, dx, dy = self.nx, self.ny, self.dx, self.dy
        residual_max_udiv = 0.0
        for i,j in ti.ndrange((2,nx+1),(1,ny+1)):
            self.u_mid[i,j] = (- self.coef_u[i,j,1] * self.u[i-1,j] \
                           - self.coef_u[i,j,2] * self.u[i+1,j] \
                           - self.coef_u[i,j,3] * self.u[i,j+1] \
                           - self.coef_u[i,j,4] * self.u[i,j-1] \
                           + self.b_u[i,j] ) / self.coef_u[i,j,0]
            if ti.abs(self.u_mid[i,j]-self.u[i,j]) > residual_max_udiv:
                residual_max_udiv = ti.abs(self.u_mid[i,j]-self.u[i,j])
            self.u[i,j] = (1 - self.alpha_u) * self.u[i,j] + self.alpha_u * self.u_mid[i,j]
        return residual_max_udiv

    def jacob_solve_momentum_eqn(self, n_iter):
        self.compute_coef_u()
        self.compute_coef_v()                            
        self.set_bc()
        for i in range(n_iter):
            eps = self.jacobian_solve_u()
            print('>>> Momentum eps =', eps)

    def bicg_solve_pcorrection_eqn(self, eps):
        self.compute_coef_p()
        p_correction_solver = CGSolver(self.coef_p, self.b_p, self.pcor)
        p_correction_solver.solve(eps, quiet=False)
         
    @ti.kernel
    def jacobian_solve_p(self)->ti.f64:
        nx, ny = self.nx, self.ny
        residual = 0.0
        for i,j in ti.ndrange((1,nx+1),(1,ny+1)):
            self.pcor_mid[i,j] = (- self.coef_p[i,j,1] * self.pcor[i-1,j] \
                              - self.coef_p[i,j,2] * self.pcor[i+1,j] \
                              - self.coef_p[i,j,3] * self.pcor[i,j+1] \
                              - self.coef_p[i,j,4] * self.pcor[i,j-1] \
                              + self.mdiv[i,j]) / self.coef_p[i,j,0]
            if ti.abs(self.pcor[i,j]-self.pcor_mid[i,j]) > residual:
                residual = ti.abs(self.pcor[i,j]-self.pcor_mid[i,j])
            self.pcor[i,j] = self.pcor_mid[i,j]
        return residual
    
    def jacob_solve_pcorrection_eqn(self, eps):
        res = 1.0
        while res > eps:
            res = self.jacobian_solve_p()
            print('>>> Pressure correction residual =', res)
    
    @ti.kernel
    def init(self):
        for i,j in self.u:
            self.u[i,j] = 0.0
            self.u_mid[i,j] = 0.0            
        for i,j in self.v:
            self.v[i,j] = 0.0
            self.v_mid[i,j] = 0.0             
        for i,j in self.p:
            self.p[i,j] = 1.0
            #self.pcor[i,j] = 0.0
            #self.pcor_mid[i,j] = 0.0                        

    def solve(self):
        step = 0
        self.init()
        print('Solving momentum using Jacobian...')
        self.jacob_solve_momentum_eqn(1000)
        self.compute_coef_p()

        # Jacobian solve p correction
        print('Solving pressure using Jacobian...')        
        now = time.time()
        self.jacob_solve_pcorrection_eqn(1e-6)
        ts1  = time.time() - now
        
        self.disp.display(f'log/{step:06}-jacob.png')
        self.dump_coef(step, 'jacob')            
        self.dump_matrix(step, 'jacob')

        # BICG solve p correction
        print('Solving pressure using BICG...')                
        now = time.time()        
        self.bicg_solve_pcorrection_eqn(1e-6)
        ts2  = time.time() - now
        
        self.disp.display(f'log/{step:06}-bicg.png')
        self.dump_coef(step, 'bicg')            
        self.dump_matrix(step, 'bicg')
        print('>>> Jacobian p correction took', ts1, 'sec.')        
        print('>>> BICG p correction took', ts2, 'sec.')        
            

# Lid-driven Cavity            
ssolver = SIMPLESolver(1.0, 1.0, 50, 50) # lx, ly, nx, ny

# Boundary conditions
#ssolver.bc['w'][0] = 1.0    # West Normal velocity               
#ssolver.bc['w'][1] = 1.0    # West Tangential velocity

#ssolver.bc['e'][0] = 1.0    # East Normal velocity
# ssolver.bc['e'][1] = 0.0    # East Tangential velocity

# ssolver.bc['n'][0] = 0.0    # North Normal velocity
ssolver.bc['n'][1] = 1.0    # North Tangential velocity

# ssolver.bc['s'][0] = 0.0    # South Normal velocity
# ssolver.bc['s'][1] = 0.0    # South Tangential velocity

ssolver.solve()
