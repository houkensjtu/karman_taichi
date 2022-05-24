import taichi as ti
import numpy as np

@ti.data_oriented
class CGSolver:
    def __init__(self, coef, b, x):
        self.coef = coef
        self.b = b
        self.x = x
        self.NX = coef.shape[0]
        self.NY = coef.shape[1]
        self.real = ti.f64
        self.N_ext = 0
        self.steps = self.NX * self.NY # Cg should converge within the size of the vector
        self.history = []                # Convergence history data
        
        # -- Conjugate gradient variables -- 
        self.r = ti.field(dtype=self.real) # residual
        self.p = ti.field(dtype=self.real) # conjugate gradient
        self.Ap = ti.field(dtype=self.real)# matrix-vector product
        self.Ax = ti.field(dtype=self.real)# matrix-vector product
        self.alpha = ti.field(dtype=self.real)  # step size
        self.beta = ti.field(dtype=self.real)  # step size
        ti.root.place(self.alpha, self.beta) # , self.sum)

        ti.root.dense(ti.ij, (self.NX, self.NY)).place(self.p, self.Ap, self.r, self.Ax) # Dense data structure
        #ti.root.pointer(ti.ij, self.N_tot//16) \
        #       .dense(ti.ij, 16) \
        #       .place(self.x, self.p, self.Ap, self.r, self.Ax, self.b)
        
    @ti.kernel
    def init(self):
        for i, j in ti.ndrange((self.N_ext, self.NX-self.N_ext), (self.N_ext, self.NY-self.N_ext)):
            # r[0] = b - Ax, where x = 0; therefore r[0] = b
            self.r[i,j]  = self.b[i,j]
            self.Ap[i,j] = 0.0
            self.Ax[i,j] = 0.0
            self.p[i,j]  = 0.0
            self.x[i,j]  = 0.0

    @ti.kernel
    def update_coef(self, coef:ti.template(), b:ti.template(), x:ti.template()):
        for i,j,k in coef:
            self.coef[i,j,k] = coef[i,j,k]
        for i,j in b:
            self.b[i,j]    = b[i,j]
            self.x[i,j]    = x[i,j]

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template())->ti.f64:
        sum = 0.0
        for I in ti.grouped(p):
            sum += p[I] * q[I]
        return sum

    @ti.kernel
    def compute_Ap(self):
        for i, j in ti.ndrange((self.N_ext, self.NX-self.N_ext), (self.N_ext, self.NY-self.N_ext)):
            self.Ap[i,j] = self.coef[i,j,0] * self.p[i,j] + self.coef[i,j,1] * self.p[i-1,j] + self.coef[i,j,2] * self.p[i+1,j] +\
                           self.coef[i,j,3] * self.p[i,j+1] + self.coef[i,j,4] * self.p[i,j-1]

    @ti.kernel
    def compute_Ax(self):
        for i, j in ti.ndrange((self.N_ext, self.NX-self.N_ext), (self.N_ext, self.NY-self.N_ext)):
            self.Ax[i,j] = self.coef[i,j,0] * self.x[i,j] + self.coef[i,j,1] * self.x[i-1,j] + self.coef[i,j,2] * self.x[i+1,j] +\
                           self.coef[i,j,3] * self.x[i,j+1] + self.coef[i,j,4] * self.x[i,j-1]
            
    @ti.kernel
    def update_x(self):
        for I in ti.grouped(self.p):
            self.x[I] += self.alpha[None] * self.p[I]

    @ti.kernel
    def update_r(self):
        for I in ti.grouped(self.p):
            self.r[I] -= self.alpha[None] * self.Ap[I]

    @ti.kernel
    def update_p(self):
        for I in ti.grouped(self.p):
            self.p[I] = self.r[I] + self.beta[None] * self.p[I]

    def solve(self, eps=1e-8, quiet=True):
        self.init()
        initial_rTr = self.reduce(self.r, self.r) # Compute initial residual
        if not quiet:
            print('>>> Initial residual =', ti.sqrt(initial_rTr))
        self.history.append(f'{ti.sqrt(initial_rTr):e}\n')
        old_rTr = initial_rTr
        self.update_p() # Initial p = r + beta * p ( beta = 0 )
        # -- Main loop -- 
        for i in range(self.steps):
            # 1. Compute alpha
            self.compute_Ap()
            pAp = self.reduce(self.p, self.Ap)
            self.alpha[None] = old_rTr / pAp
            # 2. Update x and r using alpha
            self.update_x()
            self.update_r()
            # 3. Check for convergence
            new_rTr = self.reduce(self.r, self.r)
            if ti.sqrt(new_rTr) < ti.sqrt(initial_rTr) * eps:
                if not quiet:
                    print('>>> Conjugate Gradient method converged.')
                break
            # 4. Compute beta
            self.beta[None] = new_rTr / old_rTr
            # 5. Update p using beta
            self.update_p()
            old_rTr = new_rTr
            self.history.append(f'{ti.sqrt(new_rTr):e}\n') # Write converge history; i+1 because starting from 1.
            # Visualizations
            if not quiet:
                print(f'>>> Iter = {i+1:4}, Residual = {ti.sqrt(new_rTr):e}')

    def save_history(self):
        with open('convergence.txt', 'w') as f:
            for line in self.history:
                f.write(line)

    @ti.kernel
    def compute_residual(self): # compute r = Ax - b
        #for i,j in self.b:
        #    self.r[i,j] = self.b[i,j] - self.Ax[i,j]
        for I in ti.grouped(self.b):
            self.r[I] = self.b[I] - self.Ax[I]
        
    def check_solution(self):   # Return the norm of rTr as the residual
        # self.compute_Ax()
        # self.compute_residual()
        return np.sqrt(self.reduce(self.r, self.r)) # TODO: Is this equal to compute_residual()?

    # Build lfs of the matrix using Numpy (Slow!)
    def build_A(self):
        n = self.N * self.N
        A = (4.0+self.offset) * np.identity(n)
        for i in range(n):
            if i-1 >= 0 and i%self.N!=0:
                A[i, i-1] = -1.0
            if i-self.N >= 0:
                A[i, i-self.N] = -1.0
            if i+1 < n and i%self.N!=self.N-1:
                A[i, i+1] = -1.0
            if i+self.N < n:
                A[i, i+self.N] = -1.0
        np.savetxt('A.csv', A, delimiter=',') # For debugging purpose
        return A

    def build_b(self):
        bnp = self.b.to_numpy() # Convert to numpy ndarray
        bsp = bnp[self.N_ext:self.N_tot-self.N_ext, self.N_ext:self.N_tot-self.N_ext] # Slicing
        b = bsp.flatten(order='C') # Flatten the array to a 1D vector
        np.savetxt('b.csv', b, delimiter=',', fmt='%80.70f') # For debugging purpose
        return b
        
    def build_x(self):
        xnp = self.x.to_numpy() # Convert to numpy ndarray
        xsp = xnp[self.N_ext:self.N_tot-self.N_ext, self.N_ext:self.N_tot-self.N_ext] # Slicing
        x = xsp.flatten(order='C') # Flatten the array to a 1D vector
        np.savetxt('x.csv', x, delimiter=',') # For debugging purpose
        return x
