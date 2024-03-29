# This is a script performing the 2-D conjugate gradient method
# adapted from the 3-D mgpcg.py official example.
# No pre-conditioner or multi-grid used.

import numpy as np
import taichi as ti

# -- Taichi initialization --
real = ti.f32
ti.init(default_fp=real, arch=ti.cpu, kernel_profiler=True)


# -- Grid parameters --
N = 128
N_tot = 2 * N
N_ext = N // 2
N_gui = 512
steps = N * N # Cg should converge within the size of the vector


# -- Conjugate gradient variables -- 
r = ti.field(dtype=real) # residual
b = ti.field(dtype=real) # residual
x = ti.field(dtype=real) # solution
p = ti.field(dtype=real) # conjugate gradient
Ap = ti.field(dtype=real)# matrix-vector product
Ax = ti.field(dtype=real)# matrix-vector product
alpha = ti.field(dtype=real)  # step size
beta = ti.field(dtype=real)  # step size
sum = ti.field(dtype=real)  # storage for reductions
pixels = ti.field(dtype=real, shape=(N_gui, N_gui))  # image buffer

ti.root.place(alpha, beta, sum)

# ti.root.dense(ti.ij, N_tot).place(x, p, Ap, r, z, Ax, b) # Dense data structure
ti.root.pointer(ti.ij, N_tot//16).dense(ti.ij, 16).place(x, p, Ap, r, Ax, b)


# -- Computation kernels --
@ti.kernel
def init():
    for i, j in ti.ndrange((N_ext, N_tot-N_ext), (N_ext, N_tot-N_ext)):
        # xl, yl, zl = [0,1]        
        xl = (i - N_ext) * 2.0 / N_tot
        yl = (j - N_ext) * 2.0 / N_tot
        # r[0] = b - Ax, where x = 0; therefore r[0] = b
        r[i, j] = ti.sin(2.0 * np.pi * xl) * ti.sin(2.0 * np.pi * yl)
        b[i, j] = ti.sin(2.0 * np.pi * xl) * ti.sin(2.0 * np.pi * yl)
        Ap[i, j] = 0.0
        Ax[i, j] = 0.0
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
        # A = |4 -1 ... -1      ... ... |
        #     |-1 4 -1 ... -1      ...  |
        #     |. -1 4 -1 ... -1      ...|
        #     |          ...            |
        #     |-1 ... -1 4 -1 ... -1 ...|
        #     |          ...            |
        #     |          ...     -1 4 -1|
        #     |                     -1 4|
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


@ti.kernel
def paint():
    for i, j in pixels:
        ii = int(i * N / N_gui) + N_ext
        jj = int(j * N / N_gui) + N_ext
        pixels[i, j] = x[ii, jj] / N_tot
        
gui = ti.GUI("cg solution", res=(N_gui, N_gui))


# -- Conjugate gradient starts here --
init()
sum[None] = 0.0
reduce(r, r)
initial_rTr = sum[None] # Compute initial residual
old_rTr = initial_rTr
update_p() # Initial p = r + beta * p ( beta = 0 )
# -- Main loop -- 
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
    if new_rTr < initial_rTr * 1.0e-12:
        print('Converged')
        break

    # 4. Compute beta
    beta[None] = new_rTr / old_rTr
    
    # 5. Update p using beta
    update_p()
    old_rTr = new_rTr

    # Visualizations
    print(f'Iter = {i:4}, Residual = {new_rTr:e}')
    paint()
    gui.set_image(pixels) # Visualize the solution: x
    gui.show()

# -- Conjugate gradient main loop ends here --

# -- For visualizing the data with colorbar -- 
#def visual_matrix(mat):
#    import matplotlib.pyplot as plt
#    import matplotlib.cm as cm    
#    im = plt.imshow(mat, cmap = 'bone')
#    plt.colorbar(im)
#    plt.show()
# compute_Ax()
# visual_matrix(Ax.to_numpy())
# visual_matrix(b.to_numpy())

ti.print_kernel_profile_info()
