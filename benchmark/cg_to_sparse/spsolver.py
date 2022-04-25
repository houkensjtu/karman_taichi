import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

@ti.data_oriented
class SparsePoissonSolver:
    def __init__(self, n=128, solver_type='LU'):
        self.N = n
        self.solver_type = solver_type
        
    # Build lfs of the equation using SparseMatrixBuilder
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

    def build_b(self):
        b = np.zeros(shape=(self.N,self.N))
        for i,j in np.ndindex(self.N, self.N):
            xl = i / self.N
            yl = j / self.N
            b[i, j] = ti.sin(2.0 * np.pi * xl) * ti.sin(2.0 * np.pi * yl)
        b = b.flatten(order='C')
        return b

    def solve(self):
        asp = self.build_ASparse()
        solver = ti.linalg.SparseSolver(solver_type=self.solver_type)
        solver.analyze_pattern(asp)
        solver.factorize(asp)
        b = self.build_b()
        return solver.solve(b)


sp = SparsePoissonSolver()
sp.solve()

