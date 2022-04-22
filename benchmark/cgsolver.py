import taichi as ti
import numpy as np


@ti.data_oriented
class CGSolver:
    def __init__(self, n=256, eps=1e-16, quiet=False):
        self.N = n
        real = ti.f64
        self.eps = eps
        self.N_tot = 2 * self.N
        self.N_ext = self.N // 2
        self.steps = self.N * self.N # Cg should converge within the size of the vector
        self.quiet = quiet
        # -- Conjugate gradient variables -- 
        self.r = ti.field(dtype=real) # residual
        self.b = ti.field(dtype=real) # residual
        self.x = ti.field(dtype=real) # solution
        self.p = ti.field(dtype=real) # conjugate gradient
        self.Ap = ti.field(dtype=real)# matrix-vector product
        self.Ax = ti.field(dtype=real)# matrix-vector product
        self.alpha = ti.field(dtype=real)  # step size
        self.beta = ti.field(dtype=real)  # step size
        ti.root.place(self.alpha, self.beta) # , self.sum)

        # ti.root.dense(ti.ij, (N_tot, N_tot)).place(x, p, Ap, r, Ax, b) # Dense data structure
        ti.root.pointer(ti.ij, self.N_tot//16) \
               .dense(ti.ij, 16) \
               .place(self.x, self.p, self.Ap, self.r, self.Ax, self.b)
        
    @ti.kernel
    def init(self):
        for i, j in ti.ndrange((self.N_ext, self.N_tot-self.N_ext), (self.N_ext, self.N_tot-self.N_ext)):
            # xl, yl, zl = [0,1]        
            xl = (i - self.N_ext) * 2.0 / self.N_tot
            yl = (j - self.N_ext) * 2.0 / self.N_tot
            # r[0] = b - Ax, where x = 0; therefore r[0] = b
            self.r[i, j] = ti.sin(2.0 * np.pi * xl) * ti.sin(2.0 * np.pi * yl)
            self.b[i, j] = ti.sin(2.0 * np.pi * xl) * ti.sin(2.0 * np.pi * yl)
            self.Ap[i, j] = 0.0
            self.Ax[i, j] = 0.0
            self.p[i, j] = 0.0

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template())->ti.f64:
        sum = 0.0
        for I in ti.grouped(p):
            sum += p[I] * q[I]
        return sum

    @ti.kernel
    def compute_Ap(self):
        for i, j in ti.ndrange((self.N_ext, self.N_tot-self.N_ext), (self.N_ext, self.N_tot-self.N_ext)):
            self.Ap[i,j] = 4.0 * self.p[i,j] - self.p[i+1,j] - self.p[i-1,j] - self.p[i,j+1] - self.p[i,j-1]
            
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

    def solve(self):
        self.init()
        initial_rTr = self.reduce(self.r, self.r) # Compute initial residual
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
            if new_rTr < initial_rTr * self.eps:
                print('>>> Conjugate Gradient method converged.')
                break
            # 4. Compute beta
            self.beta[None] = new_rTr / old_rTr
            # 5. Update p using beta
            self.update_p()
            old_rTr = new_rTr
            # Visualizations
            if not self.quiet:
                print(f'Iter = {i:4}, Residual = {new_rTr:e}') # Turn off residual display for perf testing.

    def build_ASparse(self):
        n = self.N * self.N
        k = ti.linalg.SparseMatrixBuilder(n,n,max_num_triplets=5*n) # Create the builder
        @ti.kernel
        def fill(A: ti.types.sparse_matrix_builder()): # Fill the builder with data
            for i in range(n):
                A[i, i] += 4.0
                if i-1 >= 0 and i%self.N !=0:
                    A[i, i-1] -= 1.0
                if i-self.N >= 0:
                    A[i, i-self.N] -= 1.0
                if i+1 < n and i%self.N!=self.N-1:
                    A[i, i+1] -= 1.0
                if i+self.N < n:
                    A[i, i+self.N] -= 1.0
        fill(k)
        A = k.build() # Build the matrix using the builder
        return A
    
    def build_A(self):
        n = self.N * self.N
        A = 4.0 * np.identity(n)
        for i in range(n):
            if i-1 >= 0 and i%self.N!=0:
                A[i, i-1] = -1.0
            if i-self.N >= 0:
                A[i, i-self.N] = -1.0
            if i+1 < n and i%self.N!=self.N-1:
                A[i, i+1] = -1.0
            if i+self.N < n:
                A[i, i+self.N] = -1.0
        np.savetxt('A.csv', A, delimiter=',')
        return A

    def build_b(self):
        bnp = self.b.to_numpy() # Convert to numpy ndarray
        bsp = bnp[self.N_ext:self.N_tot-self.N_ext, self.N_ext:self.N_tot-self.N_ext] # Slicing
        b = bsp.flatten(order='C') # Flatten the array to a 1D vector
        np.savetxt('b.csv', b, delimiter=',')        
        return b
        
    def build_x(self):
        xnp = self.x.to_numpy() # Convert to numpy ndarray
        xsp = xnp[self.N_ext:self.N_tot-self.N_ext, self.N_ext:self.N_tot-self.N_ext] # Slicing
        x = xsp.flatten(order='C') # Flatten the array to a 1D vector
        np.savetxt('x.csv', x, delimiter=',')
        return x
