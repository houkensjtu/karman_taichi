import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

@ti.data_oriented
class SparsePoissonSolver:
    def __init__(self, n=128, offset=0.1, solver_type='LU'):
        self.N = n
        self.NN = n * n
        self.offset = 0.1
        self.solver_type = solver_type
        self.builder = ti.linalg.SparseMatrixBuilder(self.NN,self.NN,max_num_triplets=5*self.NN) # Create the builder        

    # Build lfs of the equation using SparseMatrixBuilder
    def build_ASparse(self):
        @ti.kernel
        def fill(A: ti.types.sparse_matrix_builder()): # Fill the builder with data
            for i in range(self.NN):
                A[i, i] += (4.0 + self.offset)
                if i-1 >= 0 and i%self.N !=0:
                    A[i, i-1] -= 1.0
                if i-self.N >= 0:
                    A[i, i-self.N] -= 1.0
                if i+1 < self.NN and i%self.N!=self.N-1:
                    A[i, i+1] -= 1.0
                if i+self.N < self.NN:
                    A[i, i+self.N] -= 1.0
        fill(self.builder)
        self.a = self.builder.build() # Build the matrix using the builder

    def build_b(self):
        self.b = np.zeros(shape=(self.N,self.N))
        for i,j in np.ndindex(self.N, self.N):
            xl = i / self.N
            yl = j / self.N
            self.b[i, j] = ti.sin(2.0 * np.pi * xl) * ti.sin(2.0 * np.pi * yl)
        self.b = self.b.flatten(order='C')

    def solve(self):
        self.build_ASparse()
        solver = ti.linalg.SparseSolver(solver_type=self.solver_type)
        solver.analyze_pattern(self.a)
        solver.factorize(self.a)
        self.build_b()
        self.x = solver.solve(self.b)

    def check_solution(self):
        return np.linalg.norm(self.a @ self.x - self.b)

sp = SparsePoissonSolver()
sp.solve()

