import taichi as ti
ti.init(arch=ti.cpu) # only CPU backend is supported for now

nx = 4
ny = 4
n = nx * ny
# step 1: create sparse matrix builder
K = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=10000)

@ti.kernel
def fill(A: ti.types.sparse_matrix_builder()):
    for i in range(n):
        A[i, i] += 4.0
        if i-1 >= 0:
            A[i, i-1] -= 1.0
        if i-ny >= 0:
            A[i, i-ny] -= 1.0
        if i+1 < n:
            A[i, i+1] -= 1.0
        if i+ny < n:
            A[i, i+ny] -= 1.0

# step 2: fill the builder with data.
fill(K)

print(">>>> K.print_triplets()")
K.print_triplets()
# outputs:
# >>>> K.print_triplets()
# n=4, m=4, num_triplets=4 (max=100)(0, 0) val=1.0(1, 1) val=1.0(2, 2) val=1.0(3, 3) val=1.0

# step 3: create a sparse matrix from the builder.
A = K.build()
print(">>>> A = K.build()")
print(A)
# outputs:
# >>>> A = K.build()
# [1, 0, 0, 0]
# [0, 1, 0, 0]
# [0, 0, 1, 0]
# [0, 0, 0, 1]
print(type(A))
