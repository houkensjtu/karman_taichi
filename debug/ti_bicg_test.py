import taichi as ti
import numpy as np
from linalg import bicgstab

ti.init(arch=ti.cpu)

# 2022.3.1 Test results
# Bicgstab produces correct results for most arrays
# As long as the diagnol is +=1.
# Otherwise, Bicgstab might lead to divergence.

nx, ny = 2, 2 # not used in current solution
              # Used in karman to acclerate iteration(skip)
n = 320
output = 1
eps = 1e-8

#
# -- Solving in Numpy --
#
Anp = np.random.rand(n,n)
bnp = np.random.rand(n)

# Generate a 5-diag matrix, all 0 otherwise
print('Generating Matrix in Numpy...')
for i in range(n):
    for j in range(n):
        if i == j:
            pass
        elif abs(i-j)==1 or abs(i-j)==3:
            # Off main diag elements are all minus
            Anp[i,j] *= -1
        else:
            Anp[i,j] = 0
# Main diag = - all other elements + rho*dx*dy/dt
for i in range(n):
    sum = 0.0
    for j in range(n):
        if i != j:
            sum += Anp[i,j]
    Anp[i,i] = - sum + 0.1
        
            
print('Solving in numpy...')
xnp = np.linalg.solve(Anp, bnp)
print('Checking the solution...')
print(np.allclose(np.dot(Anp,xnp), bnp))

#
# -- Solving in Taichi --
#
print('Generating Matrix in Taichi...')
A = ti.field(dtype=ti.f64)
M = ti.field(dtype=ti.f64)
ti.root.dense(ti.ij, (n,n)).place(A, M)

b     = ti.field(dtype=ti.f64)
x     = ti.field(dtype=ti.f64)
x_old = ti.field(dtype=ti.f64)
x_new = ti.field(dtype=ti.f64)

Ax     = ti.field(dtype=ti.f64)
Ap     = ti.field(dtype=ti.f64)
Ap_tld = ti.field(dtype=ti.f64)

r     = ti.field(dtype=ti.f64)
p     = ti.field(dtype=ti.f64)
z     = ti.field(dtype=ti.f64)
r_tld = ti.field(dtype=ti.f64)
p_tld = ti.field(dtype=ti.f64)
z_tld = ti.field(dtype=ti.f64)

p_hat = ti.field(dtype=ti.f64)
s     = ti.field(dtype=ti.f64)
s_hat = ti.field(dtype=ti.f64)
t     = ti.field(dtype=ti.f64)

ti.root.dense(ti.i, n).place(b, x, x_old, x_new)
ti.root.dense(ti.i, n).place(Ax, Ap, Ap_tld)
ti.root.dense(ti.i, n).place(r, p, z, r_tld, p_tld, z_tld)
ti.root.dense(ti.i, n).place(p_hat, s, s_hat, t)

#
# -- Copying data from Numpy to Taichi --
#
print('Copying Matrix from Numpy to Taichi...')
A.from_numpy(Anp)
M.from_numpy(Anp)
b.from_numpy(bnp)


#
# -- Compare results --
#
print('Solving in Taichi...')
bicgstab(A, b, x, A, Ax, r, r_tld, p, p_hat, Ap, s, s_hat, t, nx, ny, n, eps, output)
print('Checking Taichi solution compared to Numpy...')
print(np.allclose(x.to_numpy(), xnp))

print('Printing solutions')
#print(x.to_numpy())
#print(xnp)
#print(Anp)
#print(bnp)
