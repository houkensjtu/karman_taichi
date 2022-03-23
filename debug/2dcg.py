import numpy as np
import taichi as ti
#from visual import visual_matrix

# -- Taichi initialization --
real = ti.f32
ti.init(default_fp=real, arch=ti.cpu)


# -- Grid parameters --
N = 4096
N_gui = 512
steps = N # Conjugate gradient should converge within the size of the matrix


# -- Conjugate gradient variables -- 
r = ti.field(dtype=real) # residual
b = ti.field(dtype=real) # residual
z = ti.field(dtype=real) # M^-1 r
x = ti.field(dtype=real) # solution
p = ti.field(dtype=real) # conjugate gradient
Ap = ti.field(dtype=real)# matrix-vector product
Ax = ti.field(dtype=real)# matrix-vector product
alpha = ti.field(dtype=real)  # step size
beta = ti.field(dtype=real)  # step size
sum = ti.field(dtype=real)  # storage for reductions
pixels = ti.field(dtype=real, shape=(N_gui, N_gui))  # image buffer

ti.root.place(alpha, beta, sum)
# Place x,p,Ap,r,z (shape = N * N)
ti.root.dense(ti.ij, N).place(x, p, Ap, r, z, Ax, b)
# Alternative data structures...
# ti.root.pointer(ti.ij, N//4).dense(ti.ij, 4).place(x, p, Ap, r, z, Ax, b)


# -- Computation kernels --
@ti.kernel
def init():
    for i, j in ti.ndrange(N, N):
        xl = i / N
        yl = j / N
        # r[0] = b - Ax, where x = 0; therefore r[0] = b
        r[i, j] = ti.sin(2.0 * np.pi * xl) * ti.sin(2.0 * np.pi * yl)
        b[i, j] = ti.sin(2.0 * np.pi * xl) * ti.sin(2.0 * np.pi * yl)
        
        z[i, j] = 0.0
        Ap[i, j] = 0.0
        p[i, j] = 0.0
        x[i, j] = 0.0


@ti.kernel
def reduce(p: ti.template(), q: ti.template()):
    for I in ti.grouped(p):
        sum[None] += p[I] * q[I]


@ti.kernel
def compute_Ap():
    for i, j in Ap:
        # A is implicitly expressed as a 2-D laplace operator
        Ap[i,j] = 4.0 * p[i,j] - p[i+1,j] - p[i-1,j] - p[i,j+1] - p[i,j-1]

        
@ti.kernel
def compute_Ax():
    for i, j in Ax:
        # A is implicitly expressed as a 2-D laplace operator
        Ax[i,j] = 4.0 * x[i,j] - x[i+1,j] - x[i-1,j] - x[i,j+1] - x[i,j-1]


@ti.kernel
def update_x():
    for I in ti.grouped(p):
        x[I] += alpha[None] * p[I]


@ti.kernel
def update_r():
    for I in ti.grouped(p):
        r[I] -= alpha[None] * Ap[I]


@ti.kernel
def update_p():
    for I in ti.grouped(p):
        p[I] = r[I] + beta[None] * p[I]


# --- Conjugate gradient main loop ---
init()
sum[None] = 0.0
reduce(r, r)
initial_rTr = sum[None] # Compute initial residual
old_rTr = initial_rTr
update_p() # Initial p = r + beta * p ( beta = 0 )

for i in range(steps):
    # 1. Compute alpha
    compute_Ap()
    sum[None] = 0.0
    reduce(p, Ap)
    pAp = sum[None]
    alpha[None] = old_rTr / pAp

    # 2. Update x and r using alpha
    update_x()
    update_r()

    # 3. Check for convergence
    sum[None] = 0.0
    reduce(r, r)
    new_rTr = sum[None]
    if new_rTr < initial_rTr * 1.0e-16:
        print('Converged')
        break

    # 4. Compute beta
    beta[None] = new_rTr / old_rTr
    
    # 5. Update p using beta
    update_p()
    old_rTr = new_rTr
    print(f'Iter = {i:4}, Residual = {new_rTr:e}')

# --- Conjugate gradient main loop end here ---


# Check the solution Ax = b in numpy
compute_Ax()
print('Checking if Ax = b ...')
print(np.allclose(Ax.to_numpy(), b.to_numpy()))
